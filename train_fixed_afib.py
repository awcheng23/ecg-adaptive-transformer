"""
Train fixed-depth transformer on AFIB dataset (Normal vs Abnormal classification).
Tracks FLOPs, computation time, and other metrics for comparison with adaptive model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

from models.fixed_transformer import CNNTransformer
from sklearn.utils.class_weight import compute_class_weight
from thop import profile


def main():
    # -----------------------
    # config
    # -----------------------
    train_path = "data/db_train_afib.npz"
    val_path   = "data/db_val_afib.npz"
    test_path  = "data/db_test_afib.npz"

    # model hyperparameters
    seq_len     = 7500    # 30 seconds @ 250 Hz
    patch_len   = 75      # 7500 / 75 = 100 patches
    d_model     = 128
    n_heads     = 2
    num_layers  = 4
    dim_ff      = 256
    dropout     = 0.1
    num_classes = 2

    # optimization hyperparameters
    batch_size     = 32
    num_epochs     = 20
    lr             = 3e-4
    weight_decay   = 1e-4
    max_grad_norm  = 1.0
    label_smoothing = 0.0

    # scheduler config
    use_scheduler = False
    step_size     = 10
    gamma         = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
    
    test_data = np.load(test_path)
    X_test = test_data['segments'].astype(np.float32)
    y_test = test_data['labels'].astype(np.int64)
    
    print(f"Train: {X_train.shape[0]} samples, shape: {X_train.shape}")
    print(f"Val: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Analyze class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
    
    # Add channel dimension if needed (B, seq_len) -> (B, 1, seq_len)
    if X_train.ndim == 2:
        X_train = X_train[:, np.newaxis, :]
        X_val = X_val[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]
    
    print(f"After adding channel: {X_train.shape}")
    
    # Create datasets and dataloaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------
    # model
    # -----------------------
    print(f"\nBuilding model with seq_len={seq_len}, patch_len={patch_len}...")
    model = CNNTransformer(
        seq_len=seq_len,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        dim_feedforward=dim_ff,
        dropout=dropout,
    ).to(device)

    # -----------------------
    # loss, optimizer, scheduler
    # -----------------------
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_train,
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    # -----------------------
    # FLOPs tracking and metrics
    # -----------------------
    flops_per_step = []
    flops_cached = None
    inference_times = []  # Track inference time per batch
    
    # Model complexity metrics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel complexity:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of transformer layers: {num_layers}")
    print(f"  Model depth (fixed): {num_layers} layers")

    # -----------------------
    # training + validation loops
    # -----------------------
    best_val_loss = float("inf")
    model_path = "checkpoints/fixed_transformer_afib.pth"
    metrics_path = "checkpoints/fixed_afib_metrics.pt"

    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)

    for epoch in range(num_epochs):
        # ====== TRAIN ======
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)  # (B, 1, 7500)
            y = y.to(device)  # (B,)

            # FLOPs: compute once on first train batch
            if flops_cached is None:
                with torch.no_grad():
                    flops, params = profile(model, inputs=(x,), verbose=False)
                flops_cached = flops
                print(f"Estimated FLOPs per forward pass: {flops_cached:.3e}")
                print(f"Number of parameters: {params}")

            flops_per_step.append(flops_cached)

            # Track inference time
            start_time = time.time()
            
            optimizer.zero_grad()
            logits = model(x)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            loss = criterion(logits, y)

            if torch.isnan(loss):
                print("NaN loss encountered, aborting training.")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            running_correct += correct
            running_total += x.size(0)

            if (batch_idx + 1) % 20 == 0:
                batch_loss = running_loss / running_total
                batch_acc = running_correct / running_total
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Train Loss: {batch_loss:.4f} Train Acc: {batch_acc:.4f} LR: {current_lr:.2e}"
                )

        train_epoch_loss = running_loss / running_total
        train_epoch_acc  = running_correct / running_total

        # ====== VALIDATION ======
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                logits_val = model(x_val)
                loss_val = criterion(logits_val, y_val)

                val_loss += loss_val.item() * x_val.size(0)
                preds_val = logits_val.argmax(dim=1)
                val_correct += (preds_val == y_val).sum().item()
                val_total += x_val.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc  = val_correct / val_total
        
        # Check class predictions
        model.eval()
        with torch.no_grad():
            sample_batch_x, sample_batch_y = next(iter(val_loader))
            sample_batch_x = sample_batch_x.to(device)
            sample_logits = model(sample_batch_x)
            sample_preds = sample_logits.argmax(dim=1).cpu().numpy()
            pred_counts = np.bincount(sample_preds, minlength=2)
        
        print(
            f"==> Epoch {epoch+1}/{num_epochs} "
            f"| Train Loss: {train_epoch_loss:.4f} Train Acc: {train_epoch_acc:.4f} "
            f"| Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}"
        )
        print(f"    Sample batch predictions: Class 0: {pred_counts[0]}, Class 1: {pred_counts[1]}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model to {model_path}")

        if scheduler is not None:
            scheduler.step()
    
    # -----------------------
    # test evaluation with detailed metrics
    # -----------------------
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    test_inference_times = []
    
    # Per-class metrics
    class_inference_times = {0: [], 1: []}  # Track inference time per class
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    
    with torch.no_grad():
        for x_test_batch, y_test_batch in test_loader:
            x_test_batch = x_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)
            
            # Track inference time for test set
            start_time = time.time()
            logits_test = model(x_test_batch)
            inference_time = time.time() - start_time
            test_inference_times.append(inference_time)
            
            # Track per-sample inference time and class
            per_sample_time = inference_time / x_test_batch.size(0)
            for label in y_test_batch.cpu().numpy():
                class_inference_times[int(label)].append(per_sample_time)
            
            loss_test = criterion(logits_test, y_test_batch)
            
            test_loss += loss_test.item() * x_test_batch.size(0)
            preds_test = logits_test.argmax(dim=1)
            test_correct += (preds_test == y_test_batch).sum().item()
            test_total += x_test_batch.size(0)
            
            # Per-class accuracy
            for pred, target in zip(preds_test.cpu().numpy(), y_test_batch.cpu().numpy()):
                class_total[int(target)] += 1
                if pred == target:
                    class_correct[int(target)] += 1
            
            all_preds.append(preds_test.cpu().numpy())
            all_targets.append(y_test_batch.cpu().numpy())
    
    test_epoch_loss = test_loss / test_total
    test_epoch_acc = test_correct / test_total
    
    # Compute average inference metrics
    avg_train_inference_time = np.mean(inference_times)
    avg_test_inference_time = np.mean(test_inference_times)
    
    # Compute per-class metrics
    class_0_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    class_1_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    class_0_time = np.mean(class_inference_times[0]) if len(class_inference_times[0]) > 0 else 0
    class_1_time = np.mean(class_inference_times[1]) if len(class_inference_times[1]) > 0 else 0
    
    print(f"\n{'='*60}")
    print("Final Results:")
    print(f"  Test Loss: {test_epoch_loss:.4f}")
    print(f"  Test Accuracy: {test_epoch_acc:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"\nPer-Class Performance:")
    print(f"  Class 0 (Normal):")
    print(f"    Accuracy: {class_0_acc:.4f} ({class_correct[0]}/{class_total[0]})")
    print(f"    Avg inference time: {class_0_time*1000:.3f} ms/sample")
    print(f"  Class 1 (Abnormal):")
    print(f"    Accuracy: {class_1_acc:.4f} ({class_correct[1]}/{class_total[1]})")
    print(f"    Avg inference time: {class_1_time*1000:.3f} ms/sample")
    print(f"  Time ratio (Abnormal/Normal): {class_1_time/class_0_time:.3f}x" if class_0_time > 0 else "")
    print(f"\nComputational Metrics:")
    print(f"  FLOPs per forward pass: {flops_cached:.3e}")
    print(f"  Avg train inference time: {avg_train_inference_time*1000:.2f} ms/batch")
    print(f"  Avg test inference time: {avg_test_inference_time*1000:.2f} ms/batch")
    print(f"  Fixed depth: {num_layers} layers (always)")
    print(f"  Total parameters: {total_params:,}")
    print(f"\nNote: Fixed transformer uses same computation for ALL samples.")
    print(f"      Time differences are due to batch effects, not model adaptivity.")
    
    # Save comprehensive metrics
    metrics = {
        'best_val_loss': best_val_loss,
        'test_loss': test_epoch_loss,
        'test_acc': test_epoch_acc,
        'flops_per_step': flops_cached,
        'avg_train_inference_time_ms': avg_train_inference_time * 1000,
        'avg_test_inference_time_ms': avg_test_inference_time * 1000,
        'num_layers': num_layers,
        'fixed_depth': num_layers,  # Always uses all layers
        'total_params': total_params,
        'trainable_params': trainable_params,
        # Per-class metrics
        'class_0_acc': class_0_acc,
        'class_1_acc': class_1_acc,
        'class_0_inference_time_ms': class_0_time * 1000,
        'class_1_inference_time_ms': class_1_time * 1000,
        'class_0_count': class_total[0],
        'class_1_count': class_total[1],
        'model_config': {
            'seq_len': seq_len,
            'patch_len': patch_len,
            'd_model': d_model,
            'n_heads': n_heads,
            'num_layers': num_layers,
            'dim_ff': dim_ff,
        }
    }
    torch.save(metrics, metrics_path)
    print(f"\nSaved metrics to {metrics_path}")
    
    flops_tensor = torch.tensor(flops_per_step, dtype=torch.float64)
    flops_path = "checkpoints/flops_per_step_afib.pt"
    torch.save(flops_tensor, flops_path)
    print(f"Saved {flops_path}")
    
    print("="*60)
    print("Training complete!")


if __name__ == "__main__":
    main()
