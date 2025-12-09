"""
Train adaptive halting transformer on AFIB dataset (Normal vs Abnormal).
Tracks halting depth, FLOPs, inference time, and per-class metrics for comparison with fixed model.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from thop import profile

from models.adaptive_transformer import AdaptiveCNNTransformer


def main():
    # -----------------------
    # config
    # -----------------------
    train_path = "data/db_train_afib.npz"
    val_path   = "data/db_val_afib.npz"

    # model hyperparameters
    seq_len     = 7500    # 30 seconds @ 250 Hz
    patch_len   = 75      # 7500 / 75 = 100 patches
    d_model     = 128
    n_heads     = 2
    num_layers  = 4
    dim_ff      = 256
    dropout     = 0.1
    num_classes = 2
    halt_epsilon = 0.05   # ACT epsilon

    # optimization hyperparameters
    batch_size      = 32
    num_epochs      = 20
    lr              = 3e-4
    weight_decay    = 1e-4
    max_grad_norm   = 1.0
    label_smoothing = 0.0
    alpha_p         = 5e-4   # ponder loss weight

    # scheduler config
    use_scheduler = False
    step_size     = 10
    gamma         = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)

    # -----------------------
    # load datasets
    # -----------------------
    print("Loading datasets...")
    train_data = np.load(train_path)
    X_train = train_data['segments'].astype(np.float32)
    y_train = train_data['labels'].astype(np.int64)

    val_data = np.load(val_path)
    X_val = val_data['segments'].astype(np.float32)
    y_val = val_data['labels'].astype(np.int64)

    print(f"Train: {X_train.shape[0]} samples, shape: {X_train.shape}")
    print(f"Val: {X_val.shape[0]} samples")

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")

    if X_train.ndim == 2:
        X_train = X_train[:, np.newaxis, :]
        X_val   = X_val[:,   np.newaxis, :]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    

    # -----------------------
    # model
    # -----------------------
    model = AdaptiveCNNTransformer(
        seq_len=seq_len,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        dim_feedforward=dim_ff,
        dropout=dropout,
        halt_epsilon=halt_epsilon,
    ).to(device)

    # -----------------------
    # loss, optimizer, scheduler
    # -----------------------
    class_weights = compute_class_weight('balanced', classes=unique_train, y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # -----------------------
    # FLOPs and metrics trackers
    # -----------------------
    flops_per_step = []
    flops_per_step_class0 = []  # Normal
    flops_per_step_class1 = []  # Abnormal
    acc_per_step = []  # Training accuracy per batch
    flops_full_cached = None
    inference_times_train = []
    

    # Model complexity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel complexity:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Max layers: {num_layers}")

    # -----------------------
    # training
    # -----------------------
    best_val_loss = float("inf")
    model_path = "checkpoints/selective_transformer_afib.pth"
    metrics_path = "checkpoints/selective_afib_metrics.pt"
    flops_path = "checkpoints/selective_flops_per_step_afib.pt"
    flops_path_classwise = "checkpoints/selective_flops_per_step_afib_classwise.pt"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # Profile full-depth FLOPs once
            if flops_full_cached is None:
                with torch.no_grad():
                    flops_full, params = profile(model, inputs=(x,), verbose=False)
                flops_full_cached = flops_full
                print(f"Full-depth FLOPs per forward: {flops_full_cached:.3e}")
                print(f"Number of parameters: {params}")

            optimizer.zero_grad()

            # Forward with rho for depth metrics
            start_time = time.time()
            logits, ponder_loss, rho = model(x, return_rho=True)
            batch_infer_time = time.time() - start_time
            inference_times_train.append(batch_infer_time)

            task_loss = criterion(logits, y)
            loss = task_loss + alpha_p * ponder_loss

            if torch.isnan(loss):
                print("NaN loss encountered, aborting training.")
                return

            # Effective FLOPs scaled by depth (per-sample rho is the ACT ponder term)
            avg_depth = rho.mean()
            flops_eff = flops_full_cached * (avg_depth.item() / num_layers)
            flops_per_step.append(flops_eff)

            # Class-wise FLOPs (scale class mean depth to full-depth FLOPs to match overall units)
            rho_cpu = rho.detach().cpu()
            labels_cpu = y.detach().cpu()
            depth_frac = rho_cpu / float(num_layers)
            class0_mask = labels_cpu == 0
            class1_mask = labels_cpu == 1
            if class0_mask.any():
                frac0 = depth_frac[class0_mask].mean().item()
                flops_per_step_class0.append(flops_full_cached * frac0)
            else:
                flops_per_step_class0.append(float('nan'))
            if class1_mask.any():
                frac1 = depth_frac[class1_mask].mean().item()
                flops_per_step_class1.append(flops_full_cached * frac1)
            else:
                flops_per_step_class1.append(float('nan'))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            running_correct += correct
            running_total += x.size(0)
            acc_per_step.append(correct / x.size(0))

            if (batch_idx + 1) % 20 == 0:
                batch_loss = running_loss / running_total
                batch_acc = running_correct / running_total
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Train Loss: {batch_loss:.4f} Train Acc: {batch_acc:.4f} "
                    f"LR: {current_lr:.2e} AvgDepth: {avg_depth.item():.3f}"
                )

        train_epoch_loss = running_loss / running_total
        train_epoch_acc  = running_correct / running_total

        # ====== VALIDATION ======
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_depths = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                logits_val, ponder_val, rho_val = model(x_val, return_rho=True)
                task_loss_val = criterion(logits_val, y_val)
                loss_val = task_loss_val + alpha_p * ponder_val

                val_loss += loss_val.item() * x_val.size(0)
                preds_val = logits_val.argmax(dim=1)
                val_correct += (preds_val == y_val).sum().item()
                val_total += x_val.size(0)
                val_depths.append(rho_val.mean().item())

        val_epoch_loss = val_loss / val_total
        val_epoch_acc  = val_correct / val_total
        avg_val_depth = np.mean(val_depths) if len(val_depths) > 0 else 0.0

        print(
            f"==> Epoch {epoch+1}/{num_epochs} "
            f"| Train Loss: {train_epoch_loss:.4f} Train Acc: {train_epoch_acc:.4f} "
            f"| Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f} "
            f"| Avg Val Depth: {avg_val_depth:.3f}"
        )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model to {model_path}")

        if scheduler is not None:
            scheduler.step()

    # -----------------------
    # Save training artifacts
    # -----------------------
    torch.save(model.state_dict(), "checkpoints/adaptive_transformer_afib_final.pth")
    torch.save(torch.tensor(flops_per_step, dtype=torch.float64), flops_path)
    torch.save(
        {
            "overall": torch.tensor(flops_per_step, dtype=torch.float64),
            "class0": torch.tensor(flops_per_step_class0, dtype=torch.float64),
            "class1": torch.tensor(flops_per_step_class1, dtype=torch.float64),
            "full_depth_flops": flops_full_cached,
            "num_layers": num_layers,
        },
        flops_path_classwise,
    )
    torch.save(torch.tensor(acc_per_step, dtype=torch.float32), "checkpoints/selective_acc_per_step_afib.pt")
    print(
        f"Saved FLOPs per step to {flops_path} and class-wise data to {flops_path_classwise}"
    )

    avg_train_infer_time = float(np.mean(inference_times_train)) if len(inference_times_train) else 0.0

    # -----------------------
    # Save training/validation metrics only
    # -----------------------
    metrics = {
        'best_val_loss': best_val_loss,
        'last_train_loss': train_epoch_loss,
        'last_train_acc': train_epoch_acc,
        'last_val_loss': val_epoch_loss,
        'last_val_acc': val_epoch_acc,
        'full_depth_flops': flops_full_cached,
        'avg_train_infer_time_ms': avg_train_infer_time * 1000,
        'num_layers': num_layers,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'alpha_p': alpha_p,
        'halt_epsilon': halt_epsilon,
        'model_config': {
            'seq_len': seq_len,
            'patch_len': patch_len,
            'd_model': d_model,
            'n_heads': n_heads,
            'num_layers': num_layers,
            'dim_ff': dim_ff,
            'dropout': dropout,
        },
    }
    torch.save(metrics, metrics_path)
    print(f"Saved training metrics to {metrics_path}")
    print("Training complete. Run test_adaptive_afib.py for test metrics.")


if __name__ == "__main__":
    main()
