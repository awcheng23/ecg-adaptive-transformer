import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASELINE_METRICS = Path("checkpoints/baseline_metrics.pt")
ADAPTIVE_METRICS = Path("checkpoints/adaptive_metrics.pt")
BASELINE_TRAIN_FLOPS = Path("checkpoints/flops_per_step.pt")
ADAPTIVE_TRAIN_FLOPS = Path("checkpoints/adaptive_flops_per_step.pt")



def load_metrics(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    # if saved as plain dict via torch.save
    if isinstance(data, dict):
        return {k: float(v) if isinstance(v, (torch.Tensor, np.generic)) else v for k, v in data.items()}
    return data


def summarize_flops(path: Path):
    if not path.exists():
        return None
    flops = torch.load(path, map_location="cpu")
    if not isinstance(flops, torch.Tensor):
        flops = torch.tensor(flops)
    flops = flops.double().flatten()
    return {
        "steps": flops.numel(),
        "mean": flops.mean().item(),
        "median": flops.median().item(),
        "min": flops.min().item(),
        "max": flops.max().item(),
        "total": flops.sum().item(),
    }


def print_section(title: str):
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def main():
    if not (BASELINE_METRICS.exists() and ADAPTIVE_METRICS.exists()):
        print("Missing metrics files. Expected baseline_metrics.pt and adaptive_metrics.pt in checkpoints/.")
        return

    base = load_metrics(BASELINE_METRICS)
    adap = load_metrics(ADAPTIVE_METRICS)

    adap_flops_per_sample = adap.get("avg_flops_per_sample_depth_scaled", adap["avg_flops_per_sample"])
    adap_total_flops = adap.get("total_est_flops_depth_scaled", adap.get("total_est_flops", None))

    base_train_flops = summarize_flops(BASELINE_TRAIN_FLOPS)
    adap_train_flops = summarize_flops(ADAPTIVE_TRAIN_FLOPS)

    # ---- Print summary ----
    print_section("Classification (Test)")
    print(f"Accuracy:   fixed={base['accuracy']:.4f} | adaptive={adap['accuracy']:.4f}")
    print(f"F1:         fixed={base['f1']:.4f} | adaptive={adap['f1']:.4f}")
    print(f"ROC-AUC:    fixed={base['roc_auc']:.4f} | adaptive={adap['roc_auc']:.4f}")
    print(f"Recall(TPR):fixed={base['sensitivity']:.4f} | adaptive={adap['sensitivity']:.4f}")
    print(f"Specificity:fixed={base['specificity']:.4f} | adaptive={adap['specificity']:.4f}")

    print_section("Compute (Test)")
    print(f"Time/sample (ms): fixed={base['avg_time_normal_ms']:.4f}* (same both classes), adaptive normal={adap['avg_time_normal_ms']:.4f}, adaptive abnormal={adap['avg_time_abnormal_ms']:.4f}")
    print(f"FLOPs/sample:     fixed={base['avg_flops_per_sample']:.3e}, adaptive (depth-scaled)={adap_flops_per_sample:.3e}")
    print(f"FLOPs/sample by class (adaptive): normal={adap['flops_per_sample_normal']:.3e}, abnormal={adap['flops_per_sample_abnormal']:.3e}, ratio={adap['flops_ratio_abnormal_to_normal']:.2f}x")

    print_section("Depth / Early Halting (Test)")
    print(f"Adaptive depth: mean={adap['avg_depth']:.3f}, median={adap['median_depth']:.3f}, normal={adap['avg_depth_normal']:.3f}, abnormal={adap['avg_depth_abnormal']:.3f}, delta={adap['depth_diff']:.3f}")
    print(f"Fixed depth implicit: 4 layers for all samples")

    print_section("Compute (Training)")
    if base_train_flops:
        print(f"Fixed train FLOPs/step mean={base_train_flops['mean']:.3e} (steps={base_train_flops['steps']})")
    else:
        print("Fixed train FLOPs missing")
    if adap_train_flops:
        print(f"Adaptive train FLOPs/step mean={adap_train_flops['mean']:.3e} (steps={adap_train_flops['steps']})")
    else:
        print("Adaptive train FLOPs missing")


if __name__ == "__main__":
    main()
