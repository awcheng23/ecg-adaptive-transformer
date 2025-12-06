import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from data_dir.datasets import ECGDataset
from models.fixed_transformer import CNNTransformer

from thop import profile   


def main():
    # -----------------------
    # config
    # -----------------------
    train_path = "data/db_train_anomaly.npz"
    val_path   = "data/db_val_anomaly.npz"
    batch_size = 32
    num_epochs = 20
    lr         = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------
    # datasets & dataloader
    # -----------------------
    train_ds = ECGDataset(train_path)
    val_ds   = ECGDataset(val_path)

    # combine train + val into one training dataset
    full_train_ds = ConcatDataset([train_ds, val_ds])

    train_loader = DataLoader(
        full_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # adjust if you want
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------
    # model, loss, optimizer
    # -----------------------
    model = CNNTransformer(
        seq_len=5000,
        patch_len=50,
        d_model=128,
        n_heads=2,
        num_layers=4,
        num_classes=2,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------
    # FLOPs tracking
    # -----------------------
    flops_per_step = []  # will store FLOPs for each training step (batch)

    # We can compute FLOPs once for a representative batch and then reuse,
    # since FLOPs are constant for given shapes & model.
    flops_cached = None

    # -----------------------
    # training loop
    # -----------------------
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)  # (B, 1, 5000)
            y = y.to(device)  # (B,)

            # ---- compute FLOPs for this step (once, then reuse) ----
            if flops_cached is None:
                # use a clone of the batch to avoid messing with autograd
                with torch.no_grad():
                    flops, params = profile(model, inputs=(x,), verbose=False)
                flops_cached = flops

            flops_per_step.append(flops_cached)

            # ---- forward ----
            optimizer.zero_grad()
            logits = model(x)              # (B, 2)
            loss = criterion(logits, y)

            # ---- backward ----
            loss.backward()
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
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {batch_loss:.4f} Acc: {batch_acc:.4f}"
                )

        epoch_loss = running_loss / running_total
        epoch_acc  = running_correct / running_total
        print(
            f"==> Epoch {epoch+1}/{num_epochs} "
            f"Finished | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
        )

    # -----------------------
    # after training
    # -----------------------
    torch.save(model.state_dict(), "checkpoints/fixed_transformer.pth")
    print("Model saved to fixed_transformer.pth")

    print("Training complete.")
    print(f"Recorded FLOPs for {len(flops_per_step)} steps.")

    # Example: convert to tensor or save to disk
    flops_tensor = torch.tensor(flops_per_step, dtype=torch.float64)
    torch.save(flops_tensor, "checkpoints/flops_per_step.pt")
    print("Saved flops_per_step.pt")


if __name__ == "__main__":
    main()
