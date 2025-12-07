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

from data_dir.datasets import ECGDataset
from models.fixed_transformer import CNNTransformer
from thop import profile


def evaluate_baseline():
    """
    Evaluate baseline transformer on test set.
    Loads saved model and FLOPs data.
    """
    
    # -----------------------
    # config
    # -----------------------
    test_path = "data/db_test_anomaly.npz"
    model_path = "checkpoints/fixed_transformer.pth"
    flops_path = "checkpoints/flops_per_step.pt"
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
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Filter out FLOPs profiling keys that thop adds
    # (thop's profile() adds total_ops and total_params which aren't part of actual model state)
    filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                          if not ('total_ops' in k or 'total_params' in k)}
    
    model.load_state_dict(filtered_checkpoint)
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # Load FLOPs data
    flops_per_step = torch.load(flops_path)
    if isinstance(flops_per_step, torch.Tensor):
        flops_per_step = flops_per_step.cpu().numpy()
    print(f"FLOPs data loaded from {flops_path}")
    print(f"Average FLOPs per batch: {np.mean(flops_per_step):.2e}")
    
    # -----------------------
    # inference on test set
    # -----------------------
    all_preds = []
    all_probs = []  # probability of class 1 (abnormal)
    all_labels = []
    inference_times = []  # time per batch
    flops_per_batch = []  # estimated FLOPs per test batch
    
    print("\nRunning inference on test set...")
    
    flops_cached = None  # reuse single FLOPs measurement for fixed shapes
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            # Estimate FLOPs once on a representative batch; constant for fixed model/input shapes.
            if flops_cached is None:
                flops_val, _ = profile(model, inputs=(x,), verbose=False)
                flops_cached = flops_val
            flops_per_batch.append(flops_cached)
            
            # Inference
            import time
            start_time = time.time()
            logits = model(x)  # (B, 2)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions & probabilities
            probs = torch.softmax(logits, dim=1)  # (B, 2)
            preds = logits.argmax(dim=1)  # (B,)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(class=1)
            all_labels.extend(y.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(test_ds)} samples")
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # -----------------------
    # compute metrics
    # -----------------------
    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION METRICS")
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
    
    # 3. Clinical Metrics (Sensitivity & Specificity)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall of abnormal class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall of normal class
    
    print(f"\n3. CLINICAL METRICS:")
    print(f"   Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"   Specificity (True Negative Rate): {specificity:.4f}")
    print(f"   (Important for ECG: high sensitivity to catch arrhythmias!)")
    
    # 4. ROC-AUC (diagnostic quality)
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"\n4. ROC-AUC: {roc_auc:.4f}")
    print(f"   (Measures model's ability to distinguish classes)")
    
    # 5. Class Balance
    n_normal = np.sum(all_labels == 0)
    n_abnormal = np.sum(all_labels == 1)
    print(f"\n5. TEST SET COMPOSITION:")
    print(f"   Normal samples (class 0):    {n_normal} ({100*n_normal/len(all_labels):.1f}%)")
    print(f"   Abnormal samples (class 1):  {n_abnormal} ({100*n_abnormal/len(all_labels):.1f}%)")
    
    # 6. Computational Metrics (Baseline for Adaptive Halting Comparison)
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    throughput = len(test_ds) / total_inference_time  # samples/sec

    avg_flops_per_batch = np.mean(flops_per_batch)
    avg_flops_per_sample = avg_flops_per_batch / batch_size
    total_est_flops = avg_flops_per_batch * len(test_loader)
    
    print(f"\n6. COMPUTATIONAL EFFICIENCY (Baseline):")
    print(f"   Avg inference time per batch ({batch_size} samples): {avg_inference_time*1000:.2f} ms")
    print(f"   Total inference time: {total_inference_time:.2f} seconds")
    print(f"   Throughput: {throughput:.1f} samples/second")
    print(f"   Avg FLOPs per batch: {avg_flops_per_batch:.2e}")
    print(f"   Avg FLOPs per sample: {avg_flops_per_sample:.2e}")
    print(f"   Total est. FLOPs over test set: {total_est_flops:.2e}")
    print(f"   (Adaptive halting will improve these metrics)")
    
    # 7. Per-class performance
    print(f"\n7. PER-CLASS PERFORMANCE:")
    precision_0 = precision_score(all_labels, all_preds, labels=[0], zero_division=0)
    recall_0 = recall_score(all_labels, all_preds, labels=[0], zero_division=0)
    f1_0 = f1_score(all_labels, all_preds, labels=[0], zero_division=0)
    
    precision_1 = precision_score(all_labels, all_preds, labels=[1], zero_division=0)
    recall_1 = recall_score(all_labels, all_preds, labels=[1], zero_division=0)
    f1_1 = f1_score(all_labels, all_preds, labels=[1], zero_division=0)
    
    print(f"   Class 0 (Normal):")
    print(f"      Precision: {precision_0:.4f}, Recall: {recall_0:.4f}, F1: {f1_0:.4f}")
    print(f"   Class 1 (Abnormal):")
    print(f"      Precision: {precision_1:.4f}, Recall: {recall_1:.4f}, F1: {f1_1:.4f}")
    
    print("\n" + "="*60)
    
    # -----------------------
    # save results
    # -----------------------
    results = {
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
        "avg_inference_time_ms": avg_inference_time * 1000,
        "throughput": throughput,
        "avg_flops_per_batch": float(avg_flops_per_batch),
        "avg_flops_per_sample": float(avg_flops_per_sample),
        "total_est_flops": float(total_est_flops),
        "precision_per_class": [float(precision_0), float(precision_1)],
        "recall_per_class": [float(recall_0), float(recall_1)],
        "f1_per_class": [float(f1_0), float(f1_1)],
    }
    
    torch.save(results, "checkpoints/baseline_metrics.pt")
    print("Results saved to checkpoints/baseline_metrics.pt")
    
    # -----------------------
    # visualizations
    # -----------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
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
    axes[1, 0].hist(all_probs[all_labels == 0], bins=30, alpha=0.6, label='Normal (GT)', color='blue')
    axes[1, 0].hist(all_probs[all_labels == 1], bins=30, alpha=0.6, label='Abnormal (GT)', color='red')
    axes[1, 0].set_xlabel('P(Abnormal)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Metrics Summary Bar Plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Sensitivity', 'Specificity']
    metrics_values = [accuracy, precision, recall, f1, sensitivity, specificity]
    colors = ['#1f77b4' if v >= 0.8 else '#ff7f0e' if v >= 0.6 else '#d62728' for v in metrics_values]
    
    axes[1, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics Summary')
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('checkpoints/baseline_evaluation.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to checkpoints/baseline_evaluation.png")
    
    return results


if __name__ == "__main__":
    evaluate_baseline()
