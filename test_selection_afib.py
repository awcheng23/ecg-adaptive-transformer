"""
Evaluate the adaptive selection transformer on the AFIB test set.
Reports accuracy, per-class compute (FLOPs/time), and compares against
fixed and adaptive-halting baselines using saved metrics files.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from thop import profile

from models.adaptive_selection_transformer import AdaptiveSelectionCNNTransformer


def load_metrics(path):
    if not os.path.exists(path):
        print(f"  WARNING: missing metrics at {path}")
        return None
    return torch.load(path, map_location="cpu")


def main():
    # -----------------------
    # config
    # -----------------------
    train_path = "data/db_train_afib.npz"  # class weights
    test_path = "data/db_test_afib.npz"

    checkpoint_path = "checkpoints/adaptive_selection_transformer_afib.pth"
    metrics_path = "checkpoints/adaptive_selection_afib_test_metrics.pt"

    # model hyperparameters (must match training)
    seq_len = 7500
    patch_len = 75
    d_model = 128
    n_heads = 2
    num_layers = 4
    dim_ff = 256
    dropout = 0.1
    num_classes = 2
    gumbel_tau = 1.0

    alpha_p = 5e-4  # compute usage regularization weight used in training
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
    y_train = train_data["labels"].astype(np.int64)

    test_data = np.load(test_path)
    X_test = test_data["segments"].astype(np.float32)
    y_test = test_data["labels"].astype(np.int64)

    print(f"Test: {X_test.shape[0]} samples, shape: {X_test.shape}")

    if X_test.ndim == 2:
        X_test = X_test[:, np.newaxis, :]

    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------
    # model + weights
    # -----------------------
    model = AdaptiveSelectionCNNTransformer(
        seq_len=seq_len,
        patch_len=patch_len,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        dim_feedforward=dim_ff,
        dropout=dropout,
        gumbel_tau=gumbel_tau,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    state_filtered = {k: v for k, v in state.items() if not k.endswith(("total_ops", "total_params"))}
    model.load_state_dict(state_filtered)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    # -----------------------
    # loss (for reporting), class weights
    # -----------------------
    unique_train = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=unique_train, y=y_train)
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
    compute_fracs = []

    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    class_time = {0: [], 1: []}
    class_compute_frac = {0: [], 1: []}

    # selection stats
    patch_keep = []
    head_keep = []
    block_keep = []

    # -----------------------
    # evaluation loop
    # -----------------------
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Profile dense FLOPs once (includes CNN + transformer)
            if flops_full_cached is None:
                flops_full, _ = profile(model, inputs=(x_batch,), verbose=False)
                flops_full_cached = flops_full
                print(f"Full-depth FLOPs per forward: {flops_full_cached:.3e}")

            start_time = time.time()
            logits, compute_fraction_batch, stats = model(x_batch, return_stats=True)
            batch_time = time.time() - start_time
            inference_times.append(batch_time)
            compute_fracs.append(float(compute_fraction_batch.item()))

            task_loss = criterion(logits, y_batch)
            loss = task_loss + alpha_p * compute_fraction_batch
            test_loss += loss.item() * x_batch.size(0)

            preds = logits.argmax(dim=1)
            test_correct += (preds == y_batch).sum().item()
            test_total += x_batch.size(0)

            # per-sample accounting
            per_sample_time = batch_time / x_batch.size(0)
            flops_per_sample_est = stats["flops_per_sample"].cpu().numpy()  # estimated FLOPs
            compute_frac_sample = flops_per_sample_est / (model.full_model_flops + 1e-9)
            eff_flops_sample = flops_full_cached * compute_frac_sample  # scale estimator to full model FLOPs

            # gate usage (per batch means)
            patch_keep.append(stats["mean_patch_keep"])
            head_keep.append(stats["mean_head_keep"])
            block_keep.append(stats["mean_block_keep"])

            for pred, target, comp_frac, eff_flop in zip(
                preds.cpu().numpy(),
                y_batch.cpu().numpy(),
                compute_frac_sample,
                eff_flops_sample,
            ):
                t = int(target)
                class_total[t] += 1
                if pred == target:
                    class_correct[t] += 1
                class_time[t].append(per_sample_time)
                class_compute_frac[t].append(float(comp_frac))

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
    avg_compute_frac = float(np.mean(compute_fracs)) if compute_fracs else 0.0

    class_0_compute = float(np.mean(class_compute_frac[0])) if class_compute_frac[0] else 0.0
    class_1_compute = float(np.mean(class_compute_frac[1])) if class_compute_frac[1] else 0.0

    eff_flops_overall = flops_full_cached * avg_compute_frac if flops_full_cached is not None else 0.0
    eff_flops_class_0 = flops_full_cached * class_0_compute if flops_full_cached is not None else 0.0
    eff_flops_class_1 = flops_full_cached * class_1_compute if flops_full_cached is not None else 0.0

    mean_patch_keep = float(np.mean(patch_keep)) if patch_keep else 0.0
    mean_head_keep = float(np.mean(head_keep)) if head_keep else 0.0
    mean_block_keep = float(np.mean(block_keep)) if block_keep else 0.0

    print("\nTest Results (Adaptive Selection):")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Avg compute fraction: {avg_compute_frac:.4f} (1.0 = full model)")
    print(f"  Eff. FLOPs (overall): {eff_flops_overall:.3e}")
    print("  Per-Class:")
    print(f"    Class 0 (Normal):   Acc={class_0_acc:.4f}, Compute={class_0_compute:.4f}, Time={class_0_time*1000:.3f} ms/sample")
    print(f"    Class 1 (Abnormal): Acc={class_1_acc:.4f}, Compute={class_1_compute:.4f}, Time={class_1_time*1000:.3f} ms/sample")
    if class_0_compute > 0:
        print(f"    Compute Ratio (Abn/Norm): {class_1_compute/class_0_compute:.3f}x")
    if class_0_time > 0:
        print(f"    Time Ratio (Abn/Norm): {class_1_time/class_0_time:.3f}x")
    print("  Gate utilization (means across layers/batches):")
    print(f"    Patch keep prob:  {mean_patch_keep:.3f}")
    print(f"    Head keep prob:   {mean_head_keep:.3f}")
    print(f"    Block keep prob:  {mean_block_keep:.3f}")
    print("  Compute baseline:")
    print(f"    Full-depth FLOPs: {flops_full_cached:.3e}")
    print(f"    Avg test inference time: {avg_infer_time*1000:.2f} ms/batch")

    # -----------------------
    # persist metrics
    # -----------------------
    metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "class_0_acc": class_0_acc,
        "class_1_acc": class_1_acc,
        "class_0_time_ms": class_0_time * 1000,
        "class_1_time_ms": class_1_time * 1000,
        "time_ratio_abn_norm": class_1_time / class_0_time if class_0_time > 0 else 0.0,
        "avg_test_infer_time_ms": avg_infer_time * 1000,
        "full_depth_flops": flops_full_cached,
        "avg_compute_fraction": avg_compute_frac,
        "class_0_compute_fraction": class_0_compute,
        "class_1_compute_fraction": class_1_compute,
        "eff_flops_overall": eff_flops_overall,
        "eff_flops_class_0": eff_flops_class_0,
        "eff_flops_class_1": eff_flops_class_1,
        "mean_patch_keep": mean_patch_keep,
        "mean_head_keep": mean_head_keep,
        "mean_block_keep": mean_block_keep,
        "num_layers": num_layers,
        "gumbel_tau": gumbel_tau,
        "alpha_p": alpha_p,
        "model_config": {
            "seq_len": seq_len,
            "patch_len": patch_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "num_layers": num_layers,
            "dim_ff": dim_ff,
            "dropout": dropout,
        },
    }
    torch.save(metrics, metrics_path)
    print(f"Saved test metrics to {metrics_path}")

    # -----------------------
    # comparative summary vs fixed + adaptive-halting
    # -----------------------
    fixed_metrics = load_metrics("checkpoints/fixed_afib_test_metrics.pt")
    halt_metrics = load_metrics("checkpoints/adaptive_afib_test_metrics.pt")

    print("\nComparison vs other models:")
    if fixed_metrics:
        f_flops = fixed_metrics.get("flops_per_forward", 0.0)
        f_c0_time = fixed_metrics.get("class_0_time_ms", 0.0)
        f_c1_time = fixed_metrics.get("class_1_time_ms", 0.0)
        f_acc = fixed_metrics.get("test_acc", 0.0)
        print(f"  Fixed: acc={f_acc:.4f}, FLOPs={f_flops:.3e}, time(ms) c0/c1={f_c0_time:.3f}/{f_c1_time:.3f}")
    else:
        print("  Fixed metrics not found. Run test_fixed_afib.py")

    if halt_metrics:
        h_flops = halt_metrics.get("eff_flops_overall", 0.0)
        h_c0_flops = halt_metrics.get("eff_flops_class_0", 0.0)
        h_c1_flops = halt_metrics.get("eff_flops_class_1", 0.0)
        h_acc = halt_metrics.get("test_acc", 0.0)
        print(f"  Adaptive-halting: acc={h_acc:.4f}, eff FLOPs overall={h_flops:.3e}, c0/c1={h_c0_flops:.3e}/{h_c1_flops:.3e}")
    else:
        print("  Adaptive-halting metrics not found. Run test_adaptive_afib.py")

    # quick deltas if available
    if fixed_metrics:
        compute_vs_fixed = (eff_flops_overall / fixed_metrics.get("flops_per_forward", 1.0)) if fixed_metrics.get("flops_per_forward", 0) else 0.0
        print(f"  Selection vs Fixed: overall compute {compute_vs_fixed:.3f}x of fixed")
    if halt_metrics:
        compute_vs_halt = (eff_flops_overall / halt_metrics.get("eff_flops_overall", 1.0)) if halt_metrics.get("eff_flops_overall", 0) else 0.0
        print(f"  Selection vs Adaptive-halting: overall compute {compute_vs_halt:.3f}x")


if __name__ == "__main__":
    main()
