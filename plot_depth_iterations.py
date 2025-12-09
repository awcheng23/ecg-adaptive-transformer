"""
Plot 1: Average depth of models over training iterations.

This tracks how the adaptive depth changes during training for both
weight-sharing and non-weight-sharing variants.
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
    print("Loading per-step FLOPs data...")
    
    # Load training FLOPs per step
    flops_adaptive = torch.load('checkpoints/adaptive_flops_per_step_afib.pt', map_location='cpu', weights_only=False)
    flops_adaptive_ws = torch.load('checkpoints/adaptive_ws_flops_per_step_afib.pt', map_location='cpu', weights_only=False)
    
    # Load training metrics for full-depth FLOPs
    metrics_adaptive = torch.load('checkpoints/adaptive_afib_metrics.pt', map_location='cpu', weights_only=False)
    metrics_adaptive_ws = torch.load('checkpoints/adaptive_ws_afib_metrics.pt', map_location='cpu', weights_only=False)
    
    full_depth_flops_adaptive = metrics_adaptive['full_depth_flops']
    full_depth_flops_adaptive_ws = metrics_adaptive_ws['full_depth_flops']
    
    num_layers = 4
    
    # Calculate depth from FLOPs
    depth_adaptive = calculate_depth_from_flops(flops_adaptive.numpy(), full_depth_flops_adaptive, num_layers)
    depth_adaptive_ws = calculate_depth_from_flops(flops_adaptive_ws.numpy(), full_depth_flops_adaptive_ws, num_layers)
    
    # Smooth with moving average
    window = 100
    depth_adaptive_smooth = pd.Series(depth_adaptive).rolling(window=window, center=True).mean().values
    depth_adaptive_ws_smooth = pd.Series(depth_adaptive_ws).rolling(window=window, center=True).mean().values
    
    iterations = np.arange(len(depth_adaptive))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    ax.plot(iterations, depth_adaptive_smooth, label='Adaptive (No WS)', linewidth=2, color='#1f77b4', alpha=0.8)
    ax.plot(iterations, depth_adaptive_ws_smooth, label='Adaptive (WS)', linewidth=2, color='#ff7f0e', alpha=0.8)
    
    ax.axhline(y=num_layers, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Full Depth')
    
    ax.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Depth (layers)', fontsize=12, fontweight='bold')
    ax.set_title('Model Average Depth Over Training Iterations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, num_layers + 0.5])
    
    plt.tight_layout()
    plt.savefig('plots/average_depth_iterations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/average_depth_iterations.png")
    
    # Print summary statistics
    print(f"\nAdaptive (No WS):")
    print(f"  Initial depth: {depth_adaptive_smooth[0]:.3f}")
    print(f"  Final depth: {depth_adaptive_smooth[-1]:.3f}")
    print(f"  Mean depth: {depth_adaptive_smooth[~np.isnan(depth_adaptive_smooth)].mean():.3f}")
    
    print(f"\nAdaptive (WS):")
    print(f"  Initial depth: {depth_adaptive_ws_smooth[0]:.3f}")
    print(f"  Final depth: {depth_adaptive_ws_smooth[-1]:.3f}")
    print(f"  Mean depth: {depth_adaptive_ws_smooth[~np.isnan(depth_adaptive_ws_smooth)].mean():.3f}")


if __name__ == "__main__":
    main()
