"""
Plot 2: Per-class FLOPs/depth comparison over iterations.

Shows how each model allocates computation between Normal and Abnormal classes
during training iterations.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_depth_from_flops(flops_per_step, full_depth_flops, num_layers=4):
    """
    Estimate depth from FLOPs.
    depth ≈ (actual_flops / full_depth_flops) * num_layers
    """
    return (flops_per_step / full_depth_flops) * num_layers

def main():
    print("Loading per-step class-wise FLOPs data...")
    
    # Load training class-wise FLOPs per step
    classwise_adaptive = torch.load('checkpoints/adaptive_flops_per_step_afib_classwise.pt', map_location='cpu', weights_only=False)
    classwise_adaptive_ws = torch.load('checkpoints/adaptive_ws_flops_per_step_afib_classwise.pt', map_location='cpu', weights_only=False)
    
    # Load training metrics for full-depth FLOPs
    metrics_adaptive = torch.load('checkpoints/adaptive_afib_metrics.pt', map_location='cpu', weights_only=False)
    metrics_adaptive_ws = torch.load('checkpoints/adaptive_ws_afib_metrics.pt', map_location='cpu', weights_only=False)
    
    full_depth_flops_adaptive = metrics_adaptive['full_depth_flops']
    full_depth_flops_adaptive_ws = metrics_adaptive_ws['full_depth_flops']
    
    num_layers = 4
    
    # Extract class-wise data
    class0_flops_adaptive = classwise_adaptive['class0'].numpy()  # Normal
    class1_flops_adaptive = classwise_adaptive['class1'].numpy()  # Abnormal
    
    class0_flops_adaptive_ws = classwise_adaptive_ws['class0'].numpy()  # Normal
    class1_flops_adaptive_ws = classwise_adaptive_ws['class1'].numpy()  # Abnormal
    
    # Calculate depths
    depth_c0_adaptive = calculate_depth_from_flops(class0_flops_adaptive, full_depth_flops_adaptive, num_layers)
    depth_c1_adaptive = calculate_depth_from_flops(class1_flops_adaptive, full_depth_flops_adaptive, num_layers)
    
    depth_c0_adaptive_ws = calculate_depth_from_flops(class0_flops_adaptive_ws, full_depth_flops_adaptive_ws, num_layers)
    depth_c1_adaptive_ws = calculate_depth_from_flops(class1_flops_adaptive_ws, full_depth_flops_adaptive_ws, num_layers)
    
    # Smooth with moving average
    window = 100
    depth_c0_adaptive_smooth = pd.Series(depth_c0_adaptive).rolling(window=window, center=True).mean().values
    depth_c1_adaptive_smooth = pd.Series(depth_c1_adaptive).rolling(window=window, center=True).mean().values
    
    depth_c0_adaptive_ws_smooth = pd.Series(depth_c0_adaptive_ws).rolling(window=window, center=True).mean().values
    depth_c1_adaptive_ws_smooth = pd.Series(depth_c1_adaptive_ws).rolling(window=window, center=True).mean().values
    
    iterations = np.arange(len(depth_c0_adaptive))
    
    # Create 2x1 subplots
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), dpi=300)
    
    # ============ Plot 1: Adaptive (No WS) ============
    ax = axes[0]
    ax.plot(iterations, depth_c0_adaptive_smooth, label='Normal (Class 0)', linewidth=2, color='#2ca02c', alpha=0.8)
    ax.plot(iterations, depth_c1_adaptive_smooth, label='Abnormal (Class 1)', linewidth=2, color='#d62728', alpha=0.8)
    ax.axhline(y=num_layers, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Full Depth')
    
    ax.set_ylabel('Average Depth (layers)', fontsize=12, fontweight='bold')
    ax.set_title('Adaptive (No WS) - Per-Class Depth Over Iterations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, num_layers + 0.5])
    
    # ============ Plot 2: Adaptive (WS) ============
    ax = axes[1]
    ax.plot(iterations, depth_c0_adaptive_ws_smooth, label='Normal (Class 0)', linewidth=2, color='#2ca02c', alpha=0.8)
    ax.plot(iterations, depth_c1_adaptive_ws_smooth, label='Abnormal (Class 1)', linewidth=2, color='#d62728', alpha=0.8)
    ax.axhline(y=num_layers, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Full Depth')
    
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Depth (layers)', fontsize=12, fontweight='bold')
    ax.set_title('Adaptive (WS) - Per-Class Depth Over Iterations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, num_layers + 0.5])
    
    plt.tight_layout()
    plt.savefig('plots/perclass_depth_iterations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/perclass_depth_iterations.png")
    
    # Print summary statistics
    print(f"\nAdaptive (No WS) - Final Depths:")
    print(f"  Normal (Class 0): {depth_c0_adaptive_smooth[-1]:.3f}")
    print(f"  Abnormal (Class 1): {depth_c1_adaptive_smooth[-1]:.3f}")
    print(f"  Ratio (Abnormal/Normal): {depth_c1_adaptive_smooth[-1] / depth_c0_adaptive_smooth[-1]:.3f}")
    
    print(f"\nAdaptive (WS) - Final Depths:")
    print(f"  Normal (Class 0): {depth_c0_adaptive_ws_smooth[-1]:.3f}")
    print(f"  Abnormal (Class 1): {depth_c1_adaptive_ws_smooth[-1]:.3f}")
    print(f"  Ratio (Abnormal/Normal): {depth_c1_adaptive_ws_smooth[-1] / depth_c0_adaptive_ws_smooth[-1]:.3f}")


if __name__ == "__main__":
    main()
