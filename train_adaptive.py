# train_adaptive.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_dir.datasets import ECGDataset
from models.adaptive_transformer import AdaptiveCNNTransformer  # adaptive halting version

from thop import profile


def main():
    # -----------------------
    # config
    # -----------------------
    train_path = "data/db_train_anomaly.npz"
    val_path   = "data/db_val_anomaly.npz"

    # model hyperparameters
    seq_len     = 5000
    patch_len   = 50      # 5000 / 50 = 100 patches
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
    # datasets & dataloaders
    # -----------------------
    train_ds = ECGDataset(train_path)
    val_ds   = ECGDataset(val_path)

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
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
    # FLOPs tracking (halting-aware)
    # -----------------------
    flops_per_step = []
    flops_full_cached = None  # full-depth FLOPs for one forward pass

    # -----------------------
    # best model tracking (by val loss)
    # -----------------------
    best_val_loss = float("inf")
    model_path = "checkpoints/adaptive_transformer.pth"

    # -----------------------
    # training + validation loops
    # -----------------------
    for epoch in range(num_epochs):
        # ====== TRAIN ======
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # profile full-depth FLOPs once on a representative batch
            if flops_full_cached is None:
                with torch.no_grad():
                    flops_full, params = profile(model, inputs=(x,), verbose=False)
                flops_full_cached = flops_full
                print(f"Full-depth FLOPs per forward: {flops_full_cached:.3e}")
                print(f"Number of parameters: {params}")

            optimizer.zero_grad()

            # ---- forward with halting ----
            logits, ponder_loss = model(x)            # logits: (B, 2)
            task_loss = criterion(logits, y)          # classification loss
            loss = task_loss + alpha_p * ponder_loss  # total loss

            if torch.isnan(loss):
                print("NaN loss encountered, aborting training.")
                return

            # ---- halting-aware FLOPs estimate ----
            # ponder_loss is E[N + r] over the batch; divide by num_layers
            avg_depth = ponder_loss.detach()
            L = model.num_layers
            flops_eff = flops_full_cached * (avg_depth.item() / L)
            flops_per_step.append(flops_eff)

            # ---- backward + clipping + step ----
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            # ---- stats ----
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            running_correct += correct
            running_total += x.size(0)

            if (batch_idx + 1) % 50 == 0:
                batch_loss = running_loss / running_total
                batch_acc = running_correct / running_total
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Train Loss: {batch_loss:.4f} "
                    f"Train Acc: {batch_acc:.4f} "
                    f"LR: {current_lr:.2e} "
                    f"AvgDepth: {avg_depth.item():.3f}"
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

                # we still use training forward here to include ponder loss
                logits_val, ponder_val = model(x_val)
                task_loss_val = criterion(logits_val, y_val)
                loss_val = task_loss_val + alpha_p * ponder_val

                val_loss += loss_val.item() * x_val.size(0)
                preds_val = logits_val.argmax(dim=1)
                val_correct += (preds_val == y_val).sum().item()
                val_total += x_val.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc  = val_correct / val_total

        print(
            f"==> Epoch {epoch+1}/{num_epochs} "
            f"| Train Loss: {train_epoch_loss:.4f} Train Acc: {train_epoch_acc:.4f} "
            f"| Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}"
        )

        # save best model based on validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), model_path)

        if scheduler is not None:
            scheduler.step()

    # -----------------------
    # after training
    # -----------------------
    final_ckpt_path = "checkpoints/adaptive_transformer_final.pth"
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Final model saved to {final_ckpt_path+"end"}")

    flops_tensor = torch.tensor(flops_per_step, dtype=torch.float64)
    flops_path = "checkpoints/adaptive_flops_per_step.pt"
    torch.save(flops_tensor, flops_path)
    print(f"Saved halting-aware FLOPs to {flops_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
