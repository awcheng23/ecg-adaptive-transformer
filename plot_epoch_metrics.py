"""
Parse training logs and plot training accuracy and depth metrics over epochs.
Compares Fixed, Adaptive Halting, and Adaptive Selection models.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(log_path):
    """Parse a training log file to extract epoch-level metrics."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern to match epoch summary lines
    # Example: ==> Epoch 1/20 | Train Loss: 0.6119 Train Acc: 0.5848 | Val Loss: 0.3502 Val Acc: 0.8643
    epoch_pattern = r'==> Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) Train Acc: ([\d.]+) \| Val Loss: ([\d.]+) Val Acc: ([\d.]+)'
    
    # Pattern to match depth (for adaptive models)
    # Example: | Avg Val Depth: 3.894
    depth_pattern = r'\| Avg Val Depth: ([\d.]+)'
    
    # Pattern to match compute fraction (for selection model)
    # Example: | Val Compute: 0.1176
    compute_pattern = r'\| Val Compute: ([\d.]+)'
    
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    depths = []
    computes = []
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            train_loss = float(epoch_match.group(2))
            train_acc = float(epoch_match.group(3))
            val_loss = float(epoch_match.group(4))
            val_acc = float(epoch_match.group(5))
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Check if depth or compute info is on the same line
            depth_match = re.search(depth_pattern, line)
            compute_match = re.search(compute_pattern, line)
            
            if depth_match:
                depths.append(float(depth_match.group(1)))
            elif compute_match:
                computes.append(float(compute_match.group(1)))
            else:
                depths.append(None)
                computes.append(None)
    
    return {
        'epochs': np.array(epochs),
        'train_loss': np.array(train_losses),
        'train_acc': np.array(train_accs),
        'val_loss': np.array(val_losses),
        'val_acc': np.array(val_accs),
        'depths': [d for d in depths if d is not None],
        'computes': [c for c in computes if c is not None],
    }


def main():
    # Log file paths
    log_files = {
        'Fixed': 'checkpoints/ft_afib2.log',
        'Adaptive Halting': 'checkpoints/at_afib2.log',
        'Adaptive Selection': 'checkpoints/ast_afib.log',
    }
    
    # Parse all available logs
    data = {}
    for name, path in log_files.items():
        if Path(path).exists():
            print(f"Parsing {name} log from {path}...")
            data[name] = parse_log_file(path)
            print(f"  Found {len(data[name]['epochs'])} epochs")
        else:
            print(f"Warning: {name} log not found at {path}")
    
    if not data:
        print("Error: No log files found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        'Fixed': '#1f77b4',
        'Adaptive Halting': '#ff7f0e',
        'Adaptive Selection': '#2ca02c',
    }
    
    # Plot 1: Training Accuracy over Epochs
    ax1 = axes[0, 0]
    ax1.set_title('Training Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for name, metrics in data.items():
        ax1.plot(metrics['epochs'], metrics['train_acc'], 
                marker='o', label=name, color=colors[name], linewidth=2, markersize=4)
    
    ax1.legend(fontsize=10)
    ax1.set_ylim([0.5, 1.0])
    
    # Plot 2: Validation Accuracy over Epochs
    ax2 = axes[0, 1]
    ax2.set_title('Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for name, metrics in data.items():
        ax2.plot(metrics['epochs'], metrics['val_acc'], 
                marker='o', label=name, color=colors[name], linewidth=2, markersize=4)
    
    ax2.legend(fontsize=10)
    ax2.set_ylim([0.8, 1.0])
    
    # Plot 3: Average Depth / Compute over Epochs (for adaptive models)
    ax3 = axes[1, 0]
    ax3.set_title('Adaptive Model Compute Usage over Epochs', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Avg Depth / Compute Fraction', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    for name, metrics in data.items():
        if metrics['depths']:
            epochs_depth = metrics['epochs'][:len(metrics['depths'])]
            ax3.plot(epochs_depth, metrics['depths'], 
                    marker='s', label=f"{name} (Depth)", color=colors[name], 
                    linewidth=2, markersize=4)
        elif metrics['computes']:
            epochs_compute = metrics['epochs'][:len(metrics['computes'])]
            ax3.plot(epochs_compute, metrics['computes'], 
                    marker='^', label=f"{name} (Compute)", color=colors[name], 
                    linewidth=2, markersize=4, linestyle='--')
    
    # Add reference line for max depth
    if any(data[name]['depths'] for name in data if name in data):
        ax3.axhline(y=4.0, color='gray', linestyle=':', alpha=0.5, label='Max Depth (4 layers)')
    
    ax3.legend(fontsize=10)
    
    # Plot 4: Training Loss over Epochs
    ax4 = axes[1, 1]
    ax4.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Training Loss', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    for name, metrics in data.items():
        ax4.plot(metrics['epochs'], metrics['train_loss'], 
                marker='o', label=name, color=colors[name], linewidth=2, markersize=4)
    
    ax4.legend(fontsize=10)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "plots/epoch_metrics_comparison.png"
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved epoch metrics comparison to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("EPOCH-LEVEL TRAINING SUMMARY")
    print("="*80)
    
    for name, metrics in data.items():
        print(f"\n{name}:")
        print(f"  Epochs trained: {len(metrics['epochs'])}")
        print(f"  Final train acc: {metrics['train_acc'][-1]:.4f}")
        print(f"  Final val acc: {metrics['val_acc'][-1]:.4f}")
        print(f"  Best val acc: {metrics['val_acc'].max():.4f} (epoch {metrics['epochs'][metrics['val_acc'].argmax()]})")
        
        if metrics['depths']:
            print(f"  Final avg depth: {metrics['depths'][-1]:.3f}")
            print(f"  Mean depth across epochs: {np.mean(metrics['depths']):.3f}")
        elif metrics['computes']:
            print(f"  Final compute fraction: {metrics['computes'][-1]:.4f}")
            print(f"  Mean compute fraction: {np.mean(metrics['computes']):.4f}")


if __name__ == "__main__":
    main()
