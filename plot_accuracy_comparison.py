"""
Plot training accuracy over iterations comparing Fixed, Adaptive Halting, and Adaptive Selection models.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def smooth_curve(values, window=50):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    return smoothed


def main():
    # Paths to accuracy data
    fixed_acc_path = "checkpoints/fixed_acc_per_step_afib.pt"
    adaptive_acc_path = "checkpoints/selective_acc_per_step_afib.pt"
    selection_acc_path = "checkpoints/adaptive_selection_acc_per_step_afib.pt"
    
    # Check which files exist
    paths = {
        "Fixed": fixed_acc_path,
        "Adaptive Halting": adaptive_acc_path,
        "Adaptive Selection": selection_acc_path,
    }
    
    available_models = {}
    for name, path in paths.items():
        if Path(path).exists():
            available_models[name] = path
        else:
            print(f"Warning: {name} accuracy file not found at {path}")
    
    if not available_models:
        print("Error: No accuracy data files found. Please train the models first.")
        return
    
    # Load accuracy data
    acc_data = {}
    for name, path in available_models.items():
        acc = torch.load(path).numpy()
        acc_data[name] = acc
        print(f"{name}: {len(acc)} training steps")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Colors for each model
    colors = {
        "Fixed": "#1f77b4",
        "Adaptive Halting": "#ff7f0e",
        "Adaptive Selection": "#2ca02c",
    }
    
    # Plot 1: Raw accuracy per batch
    ax1.set_title("Training Accuracy per Batch", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Training Step (Batch)", fontsize=12)
    ax1.set_ylabel("Batch Accuracy", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for name, acc in acc_data.items():
        steps = np.arange(len(acc))
        ax1.plot(steps, acc, label=name, color=colors[name], alpha=0.3, linewidth=0.5)
    
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Smoothed accuracy (moving average)
    ax2.set_title("Training Accuracy (Smoothed)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Training Step (Batch)", fontsize=12)
    ax2.set_ylabel("Accuracy (50-step Moving Average)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for name, acc in acc_data.items():
        smoothed = smooth_curve(acc, window=50)
        steps = np.arange(len(smoothed))
        ax2.plot(steps, smoothed, label=name, color=colors[name], linewidth=2)
    
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_path = "plots/accuracy_comparison.png"
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved accuracy comparison plot to {output_path}")
    
    # Print final statistics
    print("\n" + "="*60)
    print("Final Training Accuracy Statistics:")
    print("="*60)
    for name, acc in acc_data.items():
        # Last 100 steps average
        final_avg = acc[-100:].mean() if len(acc) >= 100 else acc.mean()
        overall_avg = acc.mean()
        max_acc = acc.max()
        print(f"{name}:")
        print(f"  Overall average: {overall_avg:.4f}")
        print(f"  Final 100 steps avg: {final_avg:.4f}")
        print(f"  Maximum: {max_acc:.4f}")
        print()


if __name__ == "__main__":
    main()
