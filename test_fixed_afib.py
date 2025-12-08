"""
Evaluate the fixed-depth transformer on the AFIB test set.
Loads the trained checkpoint, reports accuracy, per-class metrics, FLOPs, and inference time.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from thop import profile

from models.fixed_transformer import CNNTransformer


def main():
    # -----------------------
    # config
    # -----------------------
    train_path = "data/db_train_afib.npz"  # used only for class weights
    test_path = "data/db_test_afib.npz"

    checkpoint_path = "checkpoints/fixed_transformer_afib.pth"
    metrics_path = "checkpoints/fixed_afib_test_metrics.pt"

    # model hyperparameters (must match training)
    seq_len     = 7500
    patch_len   = 75
    d_model     = 128
    n_heads     = 2
    num_layers  = 4
    dim_ff      = 256
    dropout     = 0.1
    num_classes = 2

    label_smoothing = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    # -----------------------
    # load datasets
    # -----------------------
    print("Loading datasets...")
    train_data = np.load(train_path)
    y_train = train_data['labels'].astype(np.int64)

    test_data = np.load(test_path)
    X_test = test_data['segments'].astype(np.float32)
    y_test = test_data['labels'].astype(np.int64)

    print(f"Test: {X_test.shape[0]} samples, shape: {X_test.shape}")

    if X_test.ndim == 2:
        X_test = X_test[:, np.newaxis, :]

    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # -----------------------
    # model + weights
    # -----------------------
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
    state = torch.load(checkpoint_path, map_location=device)
    # Filter out profiling keys from thop
    state_filtered = {k: v for k, v in state.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_filtered)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    # -----------------------
    # loss (for reporting), class weights
    # -----------------------
    unique_train = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_train, y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # -----------------------
    # metrics containers
    # -----------------------
    flops_full_cached = None
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    inference_times = []

    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    class_time = {0: [], 1: []}

    # -----------------------
    # evaluation loop
    # -----------------------
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Profile full FLOPs once
            if flops_full_cached is None:
                flops_full, _ = profile(model, inputs=(x_batch,), verbose=False)
                flops_full_cached = flops_full
                print(f"Full FLOPs per forward: {flops_full_cached:.3e}")

            start_time = time.time()
            logits = model(x_batch)
            batch_time = time.time() - start_time
            inference_times.append(batch_time)

            loss = criterion(logits, y_batch)
            test_loss += loss.item() * x_batch.size(0)

            preds = logits.argmax(dim=1)
            test_correct += (preds == y_batch).sum().item()
            test_total += x_batch.size(0)

            # per-sample accounting
            per_sample_time = batch_time / x_batch.size(0)
            for pred, target in zip(preds.cpu().numpy(), y_batch.cpu().numpy()):
                class_total[int(target)] += 1
                if pred == target:
                    class_correct[int(target)] += 1
                class_time[int(target)].append(per_sample_time)

    # -----------------------
    # aggregate metrics
    # -----------------------
    test_loss /= max(test_total, 1)
    test_acc = test_correct / max(test_total, 1)

    class_0_acc = class_correct[0] / class_total[0] if class_total[0] else 0.0
    class_1_acc = class_correct[1] / class_total[1] if class_total[1] else 0.0

    class_0_time = float(np.mean(class_time[0])) if class_time[0] else 0.0
    class_1_time = float(np.mean(class_time[1])) if class_time[1] else 0.0

    avg_infer_time = float(np.mean(inference_times)) if inference_times else 0.0

    print("\nTest Results (Fixed):")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print("  Per-Class:")
    print(f"    Class 0 (Normal):   Acc={class_0_acc:.4f}, Time={class_0_time*1000:.3f} ms/sample")
    print(f"    Class 1 (Abnormal): Acc={class_1_acc:.4f}, Time={class_1_time*1000:.3f} ms/sample")
    if class_0_time > 0:
        print(f"  Time Ratio (Abnormal/Normal): {class_1_time/class_0_time:.3f}x")
    print("  Compute:")
    print(f"    FLOPs per forward pass: {flops_full_cached:.3e}")
    print(f"    Avg test inference time: {avg_infer_time*1000:.2f} ms/batch")
    print(f"    Fixed depth: {num_layers} layers (always)")
    print(f"\nNote: Fixed transformer uses same computation for ALL samples.")

    # -----------------------
    # persist metrics
    # -----------------------
    metrics = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'class_0_acc': class_0_acc,
        'class_1_acc': class_1_acc,
        'class_0_time_ms': class_0_time * 1000,
        'class_1_time_ms': class_1_time * 1000,
        'time_ratio_abn_norm': class_1_time / class_0_time if class_0_time > 0 else 0.0,
        'avg_test_infer_time_ms': avg_infer_time * 1000,
        'flops_per_forward': flops_full_cached,
        'num_layers': num_layers,
        'fixed_depth': num_layers,  # Always uses all layers
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
    print(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
