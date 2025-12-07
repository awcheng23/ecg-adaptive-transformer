import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

from data_dir.datasets import ECGDataset
from models.adaptive_transformer import AdaptiveCNNTransformer
from thop import profile


def forward_with_depth(model: AdaptiveCNNTransformer, x: torch.Tensor, eps: float = None):
    """Run adaptive model and return (logits, rho_per_sample) where rho≈depth.

    Mirrors the model's forward halting logic but keeps per-sample rho instead of mean.
    """
    if eps is None:
        eps = model.halt_epsilon

    B = x.size(0)
    device = x.device
    L = model.num_layers

    # CNN patch embedding
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.act1(x)

    x = model.conv2(x)
    x = model.bn2(x)
    x = model.act2(x)          # (B, d_model, num_patches)

    x = x.transpose(1, 2)     # (B, num_patches, d_model)
    x = model.pos_encoder(x)

    # Halting state
    c = torch.zeros(B, device=device)
    R = torch.ones(B, device=device)
    mask = torch.ones(B, device=device)
    rho = torch.zeros(B, device=device)

    output = torch.zeros(B, model.d_model, device=device)

    for l, layer in enumerate(model.layers):
        x = layer(x)
        z = x.mean(dim=1)  # (B, d_model)

        first_dim = z[:, 0]
        h_l = torch.sigmoid(model.halt_gamma * first_dim - model.halt_center)
        if l == L - 1:
            h_l = torch.ones_like(h_l)

        active = (mask > 0.0)
        h_eff = h_l * active.float()

        c = c + h_eff
        rho = rho + mask  # base cost per active layer

        reached = (c > 1.0 - eps) & active
        not_reached = (c < 1.0 - eps) & active

        reached_f = reached.float()
        not_reached_f = not_reached.float()

        delta1 = z * (R * reached_f).unsqueeze(-1)
        rho = rho + R * reached_f  # add remainder cost where halted

        R = R - (h_eff * not_reached_f)
        delta2 = z * (h_eff * not_reached_f).unsqueeze(-1)

        output = output + delta1 + delta2

        mask = (c < 1.0 - eps).float()

    logits = model.mlp_head(output)
    return logits, rho  # rho per sample


def evaluate_adaptive():
    """
    Evaluate adaptive transformer on test set.
    Tracks: depth distribution, per-class timing, FLOPs savings, early halting patterns.
    """
    
    # -----------------------
    # config
    # -----------------------
    test_path = "data/db_test_anomaly.npz"
    model_path = "checkpoints/adaptive_transformer.pth"
    batch_size = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # -----------------------
    # dataset & dataloader
    # -----------------------
    test_ds = ECGDataset(test_path)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    
    print(f"Test set size: {len(test_ds)}")
    
    # -----------------------
    # model loading
    # -----------------------
    model = AdaptiveCNNTransformer(
        seq_len=5000,
        patch_len=50,
        d_model=128,
        n_heads=2,
        num_layers=4,
        num_classes=2,
        dim_feedforward=256,
        dropout=0.1,
        halt_epsilon=0.05,
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                          if not ('total_ops' in k or 'total_params' in k)}
    model.load_state_dict(filtered_checkpoint)
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # -----------------------
    # inference on test set
    # -----------------------
    all_preds = []
    all_probs = []
    all_labels = []
    all_depths = []  # track halting depth per sample
    inference_times = []
    inference_times_per_sample = []
    flops_per_batch = []
    
    print("\nRunning inference on test set...")
    
    flops_cached = None
    num_layers = 4  # model.num_layers (constant here)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            # Estimate FLOPs once
            if flops_cached is None:
                flops_val, _ = profile(model, inputs=(x,), verbose=False)
                flops_cached = flops_val
            flops_per_batch.append(flops_cached)
            
            # Forward pass with timing and per-sample depth (rho)
            start_time = time.time()
            logits, rho = forward_with_depth(model, x)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Per-sample timing
            time_per_sample_batch = inference_time / x.size(0)
            inference_times_per_sample.extend([time_per_sample_batch] * x.size(0))

            # Per-sample depth (rho ≈ N + r)
            depths = rho.detach().cpu().numpy()
            all_depths.extend(depths)
            
            # Predictions
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(test_ds)} samples")
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_depths = np.array(all_depths)
    inference_times_per_sample = np.array(inference_times_per_sample)
    
    # -----------------------
    # compute metrics
    # -----------------------
    print("\n" + "="*60)
    print("ADAPTIVE MODEL EVALUATION METRICS")
    print("="*60)
    
    # 1. Classification Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\n1. CLASSIFICATION METRICS:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # 2. Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print(f"\n2. CONFUSION MATRIX:")
    print(f"   True Negatives (TN):  {tn}")
    print(f"   False Positives (FP): {fp}")
    print(f"   False Negatives (FN): {fn}")
    print(f"   True Positives (TP):  {tp}")
    
    # 3. Clinical Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n3. CLINICAL METRICS:")
    print(f"   Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"   Specificity (True Negative Rate): {specificity:.4f}")
    
    # 4. ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"\n4. ROC-AUC: {roc_auc:.4f}")
    
    # 5. Class Balance
    n_normal = np.sum(all_labels == 0)
    n_abnormal = np.sum(all_labels == 1)
    print(f"\n5. TEST SET COMPOSITION:")
    print(f"   Normal samples (class 0):    {n_normal} ({100*n_normal/len(all_labels):.1f}%)")
    print(f"   Abnormal samples (class 1):  {n_abnormal} ({100*n_abnormal/len(all_labels):.1f}%)")
    
    # 6. Computational Metrics
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    throughput = len(test_ds) / total_inference_time
    
    avg_flops_per_batch = np.mean(flops_per_batch)
    avg_flops_per_sample_full = avg_flops_per_batch / batch_size  # full-depth cost
    total_est_flops_full = avg_flops_per_batch * len(test_loader)
    
    print(f"\n6. COMPUTATIONAL EFFICIENCY (Adaptive):")
    print(f"   Avg inference time per batch ({batch_size} samples): {avg_inference_time*1000:.2f} ms")
    print(f"   Total inference time: {total_inference_time:.2f} seconds")
    print(f"   Throughput: {throughput:.1f} samples/second")
    print(f"   Avg FLOPs per batch (full depth): {avg_flops_per_batch:.2e}")
    print(f"   Avg FLOPs per sample (full depth): {avg_flops_per_sample_full:.2e}")
    print(f"   Total est. FLOPs over test set (full depth): {total_est_flops_full:.2e}")
    
    # 7. Per-class timing
    normal_mask = all_labels == 0
    normal_times = inference_times_per_sample[normal_mask]
    avg_time_normal = np.mean(normal_times) if len(normal_times) > 0 else 0
    median_time_normal = np.median(normal_times) if len(normal_times) > 0 else 0
    
    abnormal_mask = all_labels == 1
    abnormal_times = inference_times_per_sample[abnormal_mask]
    avg_time_abnormal = np.mean(abnormal_times) if len(abnormal_times) > 0 else 0
    median_time_abnormal = np.median(abnormal_times) if len(abnormal_times) > 0 else 0
    
    print(f"\n7. PER-CLASS TIMING ANALYSIS:")
    print(f"   Normal samples (class 0):")
    print(f"      Avg time per sample: {avg_time_normal*1000:.4f} ms")
    print(f"      Median time per sample: {median_time_normal*1000:.4f} ms")
    print(f"      Count: {len(normal_times)}")
    print(f"   Abnormal samples (class 1):")
    print(f"      Avg time per sample: {avg_time_abnormal*1000:.4f} ms")
    print(f"      Median time per sample: {median_time_abnormal*1000:.4f} ms")
    print(f"      Count: {len(abnormal_times)}")
    
    # 8. Depth Analysis (Early Halting)
    avg_depth = np.mean(all_depths)
    median_depth = np.median(all_depths)
    min_depth = np.min(all_depths)
    max_depth = np.max(all_depths)

    normal_depths = all_depths[normal_mask]
    abnormal_depths = all_depths[abnormal_mask]
    avg_depth_normal = np.mean(normal_depths) if len(normal_depths) > 0 else 0
    avg_depth_abnormal = np.mean(abnormal_depths) if len(abnormal_depths) > 0 else 0
    
    print(f"\n8. EARLY HALTING / DEPTH ANALYSIS:")
    print(f"   Overall depth (ponder_loss):")
    print(f"      Mean: {avg_depth:.4f} layers")
    print(f"      Median: {median_depth:.4f} layers")
    print(f"      Min/Max: {min_depth:.4f} / {max_depth:.4f} layers")
    print(f"      (Max possible depth = 4 layers)")
    print(f"   Per-class average depth:")
    print(f"      Normal: {avg_depth_normal:.4f} layers")
    print(f"      Abnormal: {avg_depth_abnormal:.4f} layers")
    print(f"      Difference: {avg_depth_abnormal - avg_depth_normal:.4f} layers")
    print(f"      (Positive = abnormal requires more computation)")
    
    # 9. Per-class FLOPs Analysis (depth-proportional)
    num_layers = 4  # fixed architecture
    flops_per_sample_normal = (avg_flops_per_sample_full * avg_depth_normal / num_layers) if avg_depth_normal > 0 else avg_flops_per_sample_full
    flops_per_sample_abnormal = (avg_flops_per_sample_full * avg_depth_abnormal / num_layers) if avg_depth_abnormal > 0 else avg_flops_per_sample_full
    total_flops_normal = flops_per_sample_normal * len(normal_times)
    total_flops_abnormal = flops_per_sample_abnormal * len(abnormal_times)
    
    print(f"\n9. PER-CLASS FLOPS ANALYSIS:")
    print(f"   Est. FLOPs per sample (depth-normalized):")
    print(f"      Normal: {flops_per_sample_normal:.2e}")
    print(f"      Abnormal: {flops_per_sample_abnormal:.2e}")
    print(f"      Ratio (Abnormal/Normal): {flops_per_sample_abnormal / flops_per_sample_normal if flops_per_sample_normal > 0 else 1.0:.2f}x")
    print(f"   Total est. FLOPs by class:")
    print(f"      Normal: {total_flops_normal:.2e}")
    print(f"      Abnormal: {total_flops_abnormal:.2e}")
    print(f"   (Early halting should reduce normal FLOPs significantly)")
    
    # 7. Per-class performance
    precision_0 = precision_score(all_labels, all_preds, labels=[0], zero_division=0)
    recall_0 = recall_score(all_labels, all_preds, labels=[0], zero_division=0)
    f1_0 = f1_score(all_labels, all_preds, labels=[0], zero_division=0)
    
    precision_1 = precision_score(all_labels, all_preds, labels=[1], zero_division=0)
    recall_1 = recall_score(all_labels, all_preds, labels=[1], zero_division=0)
    f1_1 = f1_score(all_labels, all_preds, labels=[1], zero_division=0)
    
    print(f"\n10. PER-CLASS PERFORMANCE:")
    print(f"   Class 0 (Normal):")
    print(f"      Precision: {precision_0:.4f}, Recall: {recall_0:.4f}, F1: {f1_0:.4f}")
    print(f"   Class 1 (Abnormal):")
    print(f"      Precision: {precision_1:.4f}, Recall: {recall_1:.4f}, F1: {f1_1:.4f}")
    
    print("\n" + "="*60)
    
    # -----------------------
    # save results
    # -----------------------
    results = {
        # Classification
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        # Timing
        "avg_inference_time_ms": avg_inference_time * 1000,
        "throughput": throughput,
        # FLOPs
        "avg_flops_per_batch": float(avg_flops_per_batch),
        "avg_flops_per_sample": float(avg_flops_per_sample_full),
        "total_est_flops": float(total_est_flops_full),
        # Per-class timing
        "avg_time_normal_ms": float(avg_time_normal * 1000),
        "median_time_normal_ms": float(median_time_normal * 1000),
        "count_normal": int(len(normal_times)),
        "avg_time_abnormal_ms": float(avg_time_abnormal * 1000),
        "median_time_abnormal_ms": float(median_time_abnormal * 1000),
        "count_abnormal": int(len(abnormal_times)),
        # Depth / Early Halting
        "avg_depth": float(avg_depth),
        "median_depth": float(median_depth),
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "avg_depth_normal": float(avg_depth_normal),
        "avg_depth_abnormal": float(avg_depth_abnormal),
        "depth_diff": float(avg_depth_abnormal - avg_depth_normal),
        # Per-class FLOPs
        "flops_per_sample_normal": float(flops_per_sample_normal),
        "flops_per_sample_abnormal": float(flops_per_sample_abnormal),
        "total_flops_normal": float(total_flops_normal),
        "total_flops_abnormal": float(total_flops_abnormal),
        "flops_ratio_abnormal_to_normal": float(flops_per_sample_abnormal / flops_per_sample_normal if flops_per_sample_normal > 0 else 1.0),
        # Per-class performance
        "precision_per_class": [float(precision_0), float(precision_1)],
        "recall_per_class": [float(recall_0), float(recall_1)],
        "f1_per_class": [float(f1_0), float(f1_1)],
    }
    
    torch.save(results, "checkpoints/adaptive_metrics.pt")
    print("Results saved to checkpoints/adaptive_metrics.pt")
    
    # -----------------------
    # visualizations
    # -----------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Prediction Distribution
    axes[0, 2].hist(all_probs[all_labels == 0], bins=30, alpha=0.6, label='Normal (GT)', color='blue')
    axes[0, 2].hist(all_probs[all_labels == 1], bins=30, alpha=0.6, label='Abnormal (GT)', color='red')
    axes[0, 2].set_xlabel('P(Abnormal)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Prediction Probability Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Depth Distribution by Class
    axes[1, 0].hist(normal_depths, bins=30, alpha=0.6, label=f'Normal (μ={avg_depth_normal:.2f})', color='blue')
    axes[1, 0].hist(abnormal_depths, bins=30, alpha=0.6, label=f'Abnormal (μ={avg_depth_abnormal:.2f})', color='red')
    axes[1, 0].set_xlabel('Halting Depth (layers)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Early Halting Depth Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Inference Time by Class
    axes[1, 1].hist(normal_times*1000, bins=30, alpha=0.6, label=f'Normal (μ={avg_time_normal*1000:.3f}ms)', color='blue')
    axes[1, 1].hist(abnormal_times*1000, bins=30, alpha=0.6, label=f'Abnormal (μ={avg_time_abnormal*1000:.3f}ms)', color='red')
    axes[1, 1].set_xlabel('Inference Time (ms)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Per-Sample Inference Time')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Metrics Summary Bar Plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Sensitivity', 'Specificity']
    metrics_values = [accuracy, precision, recall, f1, sensitivity, specificity]
    colors = ['#1f77b4' if v >= 0.8 else '#ff7f0e' if v >= 0.6 else '#d62728' for v in metrics_values]
    
    axes[1, 2].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Performance Metrics Summary')
    axes[1, 2].set_ylim([0, 1.05])
    axes[1, 2].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('checkpoints/adaptive_evaluation.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to checkpoints/adaptive_evaluation.png")
    
    return results


if __name__ == "__main__":
    evaluate_adaptive()
