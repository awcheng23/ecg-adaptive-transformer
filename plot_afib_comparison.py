"""
Generate plots comparing fixed vs adaptive transformers on AFIB dataset.
Creates visualizations to illustrate hypothesis validation and performance differences.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def load_metrics(path):
    """Load metrics dict from saved torch file."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    return torch.load(path, map_location='cpu')


def setup_plot_style():
    """Configure matplotlib style for publication-quality plots."""
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


def plot_accuracy_comparison(fixed_metrics, adaptive_metrics, save_dir):
    """Plot overall and per-class accuracy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall accuracy
    models = ['Fixed', 'Adaptive']
    overall_acc = [
        fixed_metrics.get('test_acc', 0.0),
        adaptive_metrics.get('test_acc', 0.0)
    ]
    
    bars1 = ax1.bar(models, overall_acc, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Overall Test Accuracy')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Per-class accuracy
    classes = ['Normal\n(Class 0)', 'Abnormal\n(Class 1)']
    fixed_per_class = [
        fixed_metrics.get('class_0_acc', 0.0),
        fixed_metrics.get('class_1_acc', 0.0)
    ]
    adaptive_per_class = [
        adaptive_metrics.get('class_0_acc', 0.0),
        adaptive_metrics.get('class_1_acc', 0.0)
    ]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, fixed_per_class, width, label='Fixed', 
                    color='#3498db', alpha=0.7, edgecolor='black')
    bars3 = ax2.bar(x + width/2, adaptive_per_class, width, label='Adaptive',
                    color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylim([0, 1.0])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), bbox_inches='tight')
    print(f"  Saved: {save_dir}/accuracy_comparison.png")
    plt.close()


def plot_depth_allocation(adaptive_metrics, save_dir):
    """Plot adaptive depth allocation per class (Hypothesis 1 & 2)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    num_layers = adaptive_metrics.get('num_layers', 4)
    
    # 1. Average depth comparison
    classes = ['Normal', 'Abnormal']
    depths = [
        adaptive_metrics.get('class_0_depth', 0.0),
        adaptive_metrics.get('class_1_depth', 0.0)
    ]
    colors = ['#2ecc71', '#e67e22']
    
    bars = ax1.bar(classes, depths, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=num_layers, color='red', linestyle='--', linewidth=2, label=f'Max Depth ({num_layers})')
    ax1.set_ylabel('Average Depth (layers)')
    ax1.set_title('Average Depth per Class\n(H1 & H2: Abnormal should use more)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, depth in zip(bars, depths):
        ax1.text(bar.get_x() + bar.get_width()/2., depth,
                f'{depth:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Depth distribution (median + std as error bars)
    medians = [
        adaptive_metrics.get('depth_0_median', 0.0),
        adaptive_metrics.get('depth_1_median', 0.0)
    ]
    stds = [
        adaptive_metrics.get('depth_0_std', 0.0),
        adaptive_metrics.get('depth_1_std', 0.0)
    ]
    
    bars = ax2.bar(classes, medians, yerr=stds, color=colors, alpha=0.7, 
                   edgecolor='black', capsize=10, error_kw={'linewidth': 2})
    ax2.axhline(y=num_layers, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_ylabel('Depth (layers)')
    ax2.set_title('Depth Distribution (Median ± Std)')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, med, std) in enumerate(zip(bars, medians, stds)):
        ax2.text(bar.get_x() + bar.get_width()/2., med + std + 0.1,
                f'med={med:.2f}\nstd={std:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Full-depth fraction
    full_depth_fractions = [
        adaptive_metrics.get('full_depth_ratio_0', 0.0) * 100,
        adaptive_metrics.get('full_depth_ratio_1', 0.0) * 100
    ]
    
    bars = ax3.bar(classes, full_depth_fractions, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title(f'Samples Reaching Full Depth ({num_layers} layers)')
    ax3.set_ylim([0, 100])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, frac in zip(bars, full_depth_fractions):
        ax3.text(bar.get_x() + bar.get_width()/2., frac,
                f'{frac:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Depth ratio visualization
    depth_ratio = depths[1] / depths[0] if depths[0] > 0 else 0
    
    ax4.text(0.5, 0.6, f'Depth Ratio\n(Abnormal / Normal)', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.4, f'{depth_ratio:.3f}×', 
             ha='center', va='center', fontsize=32, color='#e74c3c', fontweight='bold')
    
    if depth_ratio > 1.0:
        ax4.text(0.5, 0.15, '✓ Abnormal uses MORE depth\n(Hypothesis validated)', 
                 ha='center', va='center', fontsize=11, color='green', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    else:
        ax4.text(0.5, 0.15, '✗ Hypothesis not validated', 
                 ha='center', va='center', fontsize=11, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'depth_allocation.png'), bbox_inches='tight')
    print(f"  Saved: {save_dir}/depth_allocation.png")
    plt.close()


def plot_computational_efficiency(fixed_metrics, adaptive_metrics, save_dir):
    """Plot FLOPs comparison (Hypothesis 3)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Overall FLOPs comparison
    fixed_flops = fixed_metrics.get('flops_per_forward', 0.0)
    adaptive_full_flops = adaptive_metrics.get('full_depth_flops', 0.0)
    adaptive_eff_flops = adaptive_metrics.get('eff_flops_overall', 0.0)
    
    models = ['Fixed\n(Full Depth)', 'Adaptive\n(Full Depth)', 'Adaptive\n(Effective)']
    flops = [fixed_flops, adaptive_full_flops, adaptive_eff_flops]
    colors_flops = ['#3498db', '#95a5a6', '#e74c3c']
    
    bars = ax1.bar(models, flops, color=colors_flops, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('FLOPs per Forward Pass')
    ax1.set_title('FLOPs Comparison')
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, flop in zip(bars, flops):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{flop:.2e}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Add savings annotation
    if adaptive_full_flops > 0:
        savings = (1 - adaptive_eff_flops / adaptive_full_flops) * 100
        ax1.annotate('', xy=(2, adaptive_eff_flops), xytext=(2, adaptive_full_flops),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax1.text(2.3, (adaptive_eff_flops + adaptive_full_flops) / 2,
                f'{savings:.1f}%\nsavings', fontsize=10, color='green', fontweight='bold')
    
    # 2. Per-class effective FLOPs
    classes = ['Normal', 'Abnormal']
    adaptive_class_flops = [
        adaptive_metrics.get('eff_flops_class_0', 0.0),
        adaptive_metrics.get('eff_flops_class_1', 0.0)
    ]
    colors_class = ['#2ecc71', '#e67e22']
    
    bars = ax2.bar(classes, adaptive_class_flops, color=colors_class, alpha=0.7, edgecolor='black')
    ax2.axhline(y=adaptive_full_flops, color='red', linestyle='--', linewidth=2, 
                label='Full-depth FLOPs', alpha=0.7)
    ax2.set_ylabel('Effective FLOPs')
    ax2.set_title('Adaptive FLOPs per Class\n(Lower = Early Halting)')
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, flop in zip(bars, adaptive_class_flops):
        height = bar.get_height()
        if adaptive_full_flops > 0:
            reduction = (1 - flop / adaptive_full_flops) * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{flop:.2e}\n({reduction:.1f}% ↓)', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'computational_efficiency.png'), bbox_inches='tight')
    print(f"  Saved: {save_dir}/computational_efficiency.png")
    plt.close()


def plot_inference_time(fixed_metrics, adaptive_metrics, save_dir):
    """Plot inference time comparison (Hypothesis 4)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Overall batch inference time
    models = ['Fixed', 'Adaptive']
    batch_times = [
        fixed_metrics.get('avg_test_infer_time_ms', 0.0),
        adaptive_metrics.get('avg_test_infer_time_ms', 0.0)
    ]
    
    bars = ax1.bar(models, batch_times, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Average Batch Inference Time')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, time in zip(bars, batch_times):
        ax1.text(bar.get_x() + bar.get_width()/2., time,
                f'{time:.2f} ms', ha='center', va='bottom', fontweight='bold')
    
    # 2. Per-class inference time
    classes = ['Normal', 'Abnormal']
    fixed_times = [
        fixed_metrics.get('class_0_time_ms', 0.0),
        fixed_metrics.get('class_1_time_ms', 0.0)
    ]
    adaptive_times = [
        adaptive_metrics.get('class_0_time_ms', 0.0),
        adaptive_metrics.get('class_1_time_ms', 0.0)
    ]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, fixed_times, width, label='Fixed',
                    color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, adaptive_times, width, label='Adaptive',
                    color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Time per Sample (ms)')
    ax2.set_title('Per-Class Inference Time\n(H4: Adaptive should be faster on Normal)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add speedup annotations
    for i, (fixed_t, adaptive_t) in enumerate(zip(fixed_times, adaptive_times)):
        if adaptive_t > 0:
            speedup = fixed_t / adaptive_t
            color = 'green' if speedup > 1 else 'red'
            ax2.text(i, max(fixed_t, adaptive_t) * 1.05,
                    f'{speedup:.2f}×', ha='center', va='bottom',
                    fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_time.png'), bbox_inches='tight')
    print(f"  Saved: {save_dir}/inference_time.png")
    plt.close()


def plot_hypothesis_summary(fixed_metrics, adaptive_metrics, save_dir):
    """Create summary visualization of all hypothesis validation."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Extract key metrics
    adaptive_c0_depth = adaptive_metrics.get('class_0_depth', 0.0)
    adaptive_c1_depth = adaptive_metrics.get('class_1_depth', 0.0)
    fixed_acc = fixed_metrics.get('test_acc', 0.0)
    adaptive_acc = adaptive_metrics.get('test_acc', 0.0)
    fixed_c0_time = fixed_metrics.get('class_0_time_ms', 0.0)
    adaptive_c0_time = adaptive_metrics.get('class_0_time_ms', 0.0)
    adaptive_full_flops = adaptive_metrics.get('full_depth_flops', 0.0)
    adaptive_eff_flops = adaptive_metrics.get('eff_flops_overall', 0.0)
    
    # H1: Normal uses less depth
    ax1 = fig.add_subplot(gs[0, 0])
    h1_pass = adaptive_c0_depth < adaptive_c1_depth
    depths = [adaptive_c0_depth, adaptive_c1_depth]
    bars = ax1.barh(['Normal', 'Abnormal'], depths, color=['#2ecc71', '#e67e22'], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Average Depth')
    ax1.set_title(f'H1: Normal Uses LESS Depth\n{"✓ PASS" if h1_pass else "✗ FAIL"}',
                 color='green' if h1_pass else 'red', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for bar, depth in zip(bars, depths):
        ax1.text(depth, bar.get_y() + bar.get_height()/2., f'{depth:.2f}',
                va='center', ha='left', fontweight='bold')
    
    # H2: Abnormal uses more depth (same data, different perspective)
    ax2 = fig.add_subplot(gs[0, 1])
    h2_pass = adaptive_c1_depth > adaptive_c0_depth
    depth_ratio = adaptive_c1_depth / adaptive_c0_depth if adaptive_c0_depth > 0 else 0
    ax2.text(0.5, 0.5, f'{depth_ratio:.2f}×',
            ha='center', va='center', fontsize=48, color='#e74c3c', fontweight='bold')
    ax2.text(0.5, 0.2, 'Abnormal / Normal\nDepth Ratio',
            ha='center', va='center', fontsize=12)
    ax2.set_title(f'H2: Abnormal Uses MORE Depth\n{"✓ PASS" if h2_pass else "✗ FAIL"}',
                 color='green' if h2_pass else 'red', fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.axis('off')
    
    # H3: Competitive accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    acc_drop = fixed_acc - adaptive_acc
    h3_pass = abs(acc_drop) <= 0.02
    accs = [fixed_acc, adaptive_acc]
    bars = ax3.bar(['Fixed', 'Adaptive'], accs, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim([max(0, min(accs) - 0.05), 1.0])
    ax3.set_title(f'H3: Competitive Accuracy (Δ={acc_drop:+.4f})\n{"✓ PASS" if h3_pass else "✗ FAIL"}',
                 color='green' if h3_pass else 'red', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax3.text(bar.get_x() + bar.get_width()/2., acc, f'{acc:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # H4: Computational savings on normal
    ax4 = fig.add_subplot(gs[1, 1])
    h4_pass = adaptive_c0_time <= fixed_c0_time
    speedup = fixed_c0_time / adaptive_c0_time if adaptive_c0_time > 0 else 0
    times = [fixed_c0_time, adaptive_c0_time]
    bars = ax4.bar(['Fixed', 'Adaptive'], times, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Time per Normal Sample (ms)')
    ax4.set_title(f'H4: Faster on Normal (Speedup={speedup:.2f}×)\n{"✓ PASS" if h4_pass else "✗ FAIL"}',
                 color='green' if h4_pass else 'red', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, time in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2., time, f'{time:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Overall summary
    ax5 = fig.add_subplot(gs[2, :])
    hypotheses = ['H1: Normal\nLess Depth', 'H2: Abnormal\nMore Depth', 
                  'H3: Competitive\nAccuracy', 'H4: Normal\nFaster']
    results = [h1_pass, h2_pass, h3_pass, h4_pass]
    colors_results = ['green' if r else 'red' for r in results]
    
    bars = ax5.barh(hypotheses, [1]*4, color=colors_results, alpha=0.6, edgecolor='black')
    ax5.set_xlim([0, 1.2])
    ax5.set_title('Hypothesis Validation Summary', fontweight='bold', fontsize=14)
    ax5.set_xticks([])
    
    for i, (bar, result) in enumerate(zip(bars, results)):
        symbol = '✓ PASS' if result else '✗ FAIL'
        ax5.text(0.5, bar.get_y() + bar.get_height()/2., symbol,
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    # Add overall score
    hypotheses_pass = sum(results)
    ax5.text(1.1, 1.5, f'{hypotheses_pass}/4\nValidated',
            ha='center', va='center', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Adaptive Halting Transformer: Hypothesis Validation', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(save_dir, 'hypothesis_summary.png'), bbox_inches='tight')
    print(f"  Saved: {save_dir}/hypothesis_summary.png")
    plt.close()


def plot_flops_reduction_breakdown(adaptive_metrics, save_dir):
    """Detailed FLOPs reduction breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    num_layers = adaptive_metrics.get('num_layers', 4)
    adaptive_c0_depth = adaptive_metrics.get('class_0_depth', 0.0)
    adaptive_c1_depth = adaptive_metrics.get('class_1_depth', 0.0)
    adaptive_full_flops = adaptive_metrics.get('full_depth_flops', 0.0)
    adaptive_eff_flops_c0 = adaptive_metrics.get('eff_flops_class_0', 0.0)
    adaptive_eff_flops_c1 = adaptive_metrics.get('eff_flops_class_1', 0.0)
    
    # 1. FLOPs reduction by class
    classes = ['Normal', 'Abnormal']
    full_flops_vals = [adaptive_full_flops, adaptive_full_flops]
    eff_flops_vals = [adaptive_eff_flops_c0, adaptive_eff_flops_c1]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, full_flops_vals, width, label='Full Depth',
                    color='#95a5a6', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, eff_flops_vals, width, label='Effective (Actual)',
                    color=['#2ecc71', '#e67e22'], alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('FLOPs')
    ax1.set_title('FLOPs: Full vs Effective per Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add reduction percentages
    for i, (full, eff) in enumerate(zip(full_flops_vals, eff_flops_vals)):
        if full > 0:
            reduction = (1 - eff / full) * 100
            ax1.text(i, max(full, eff) * 1.05, f'{reduction:.1f}% ↓',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='green')
    
    # 2. Layer utilization (depth as % of max)
    depth_utilization = [
        (adaptive_c0_depth / num_layers) * 100,
        (adaptive_c1_depth / num_layers) * 100
    ]
    
    bars = ax2.bar(classes, depth_utilization, color=['#2ecc71', '#e67e22'], alpha=0.7, edgecolor='black')
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Full Depth (100%)')
    ax2.set_ylabel('Layer Utilization (%)')
    ax2.set_title('Average Layer Utilization per Class')
    ax2.set_ylim([0, 110])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, util in zip(bars, depth_utilization):
        ax2.text(bar.get_x() + bar.get_width()/2., util,
                f'{util:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'flops_breakdown.png'), bbox_inches='tight')
    print(f"  Saved: {save_dir}/flops_breakdown.png")
    plt.close()


def main():
    print("="*70)
    print("  Generating Plots for AFIB Model Comparison")
    print("="*70)
    
    # Create output directory
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load metrics
    fixed_metrics = load_metrics("checkpoints/fixed_afib_test_metrics.pt")
    adaptive_metrics = load_metrics("checkpoints/adaptive_afib_test_metrics.pt")
    
    if fixed_metrics is None or adaptive_metrics is None:
        print("\nERROR: Could not load both test metrics. Run:")
        print("  python test_fixed_afib.py")
        print("  python test_adaptive_afib.py")
        return
    
    # Setup plotting style
    setup_plot_style()
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("\n[1/6] Accuracy comparison...")
    plot_accuracy_comparison(fixed_metrics, adaptive_metrics, save_dir)
    
    print("[2/6] Depth allocation analysis...")
    plot_depth_allocation(adaptive_metrics, save_dir)
    
    print("[3/6] Computational efficiency...")
    plot_computational_efficiency(fixed_metrics, adaptive_metrics, save_dir)
    
    print("[4/6] Inference time comparison...")
    plot_inference_time(fixed_metrics, adaptive_metrics, save_dir)
    
    print("[5/6] Hypothesis summary...")
    plot_hypothesis_summary(fixed_metrics, adaptive_metrics, save_dir)
    
    print("[6/6] FLOPs breakdown...")
    plot_flops_reduction_breakdown(adaptive_metrics, save_dir)
    
    print("\n" + "="*70)
    print(f"  All plots saved to '{save_dir}/' directory")
    print("="*70)
    print("\nGenerated plots:")
    print("  1. accuracy_comparison.png - Overall and per-class accuracy")
    print("  2. depth_allocation.png - Adaptive depth usage per class (H1 & H2)")
    print("  3. computational_efficiency.png - FLOPs comparison")
    print("  4. inference_time.png - Timing comparison (H4)")
    print("  5. hypothesis_summary.png - All hypothesis validation results")
    print("  6. flops_breakdown.png - Detailed computational savings")
    print()


if __name__ == "__main__":
    main()
