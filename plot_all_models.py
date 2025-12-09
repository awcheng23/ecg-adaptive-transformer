"""
Comprehensive comparison plotting for 4 ECG adaptive transformer models:
1. Fixed baseline
2. Adaptive Halting (with weight sharing)
3. Adaptive Halting (without weight sharing)
4. Adaptive Selection
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Color palette for 4 models
COLORS = {
    'Fixed': '#1f77b4',           # Blue
    'Halting (WS)': '#ff7f0e',    # Orange - Weight Sharing
    'Halting (No WS)': '#2ca02c', # Green - No Weight Sharing
    'Selection': '#d62728',        # Red
}

MODEL_NAMES = ['Fixed', 'Halting (WS)', 'Halting (No WS)', 'Selection']


def load_checkpoint_data(model_name):
    """Load metrics and FLOPs data for a given model."""
    if model_name == 'Fixed':
        metrics_path = 'checkpoints/fixed_afib_metrics[97].pt'
        flops_path = 'checkpoints/flops_per_step_afib[97].pt'
        acc_path = 'checkpoints/fixed_acc_per_step_afib.pt'
    elif model_name == 'Halting (WS)':
        metrics_path = 'checkpoints/adaptive_ws_afib_metrics.pt'
        flops_path = 'checkpoints/adaptive_ws_flops_per_step_afib.pt'
        acc_path = 'checkpoints/adaptive_ws_acc_per_step_afib.pt'
    elif model_name == 'Halting (No WS)':
        metrics_path = 'checkpoints/adaptive_afib_metrics.pt'
        flops_path = 'checkpoints/adaptive_flops_per_step_afib.pt'
        acc_path = 'checkpoints/adaptive_acc_per_step_afib.pt'
    elif model_name == 'Selection':
        metrics_path = 'checkpoints/adaptive_selection_afib_metrics.pt'
        flops_path = 'checkpoints/adaptive_selection_flops_per_step_afib.pt'
        acc_path = 'checkpoints/adaptive_selection_acc_per_step_afib.pt'
    else:
        return None
    
    data = {}
    
    # Load metrics
    if Path(metrics_path).exists():
        data['metrics'] = torch.load(metrics_path, map_location='cpu', weights_only=False)
    else:
        print(f"WARNING: Metrics not found for {model_name}: {metrics_path}")
        data['metrics'] = {}
    
    # Load FLOPs per step
    if Path(flops_path).exists():
        data['flops'] = torch.load(flops_path, map_location='cpu', weights_only=False).numpy()
    else:
        print(f"WARNING: FLOPs not found for {model_name}: {flops_path}")
        data['flops'] = None
    
    # Load accuracy per step
    if Path(acc_path).exists():
        data['acc'] = torch.load(acc_path, map_location='cpu', weights_only=False).numpy()
    else:
        print(f"WARNING: Accuracy not found for {model_name}: {acc_path}")
        data['acc'] = None
    
    return data


def plot_epoch_metrics_comparison():
    """Plot training/validation metrics across all 4 models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Load data from notebooks or saved checkpoint metrics
    # This assumes you have already run training for all models
    print("Creating epoch metrics comparison...")
    
    # Placeholder: In practice, this would be populated from training logs
    # For now, create a template that shows the structure
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(MODEL_NAMES)
    
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0.5, 1.0])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(MODEL_NAMES)
    
    axes[1, 0].set_title('Average Depth/Compute (Adaptive Models)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Depth (layers) / Compute Fraction')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(MODEL_NAMES)
    
    plt.tight_layout()
    plt.savefig('plots/epoch_metrics_all_models.png', dpi=300, bbox_inches='tight')
    print("  Saved: plots/epoch_metrics_all_models.png")
    plt.close()


def plot_accuracy_per_batch():
    """Plot per-batch training accuracy with moving average."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    window = 50  # Moving average window
    
    for model_name in MODEL_NAMES:
        data = load_checkpoint_data(model_name)
        if data and data['acc'] is not None:
            acc = data['acc']
            
            # Plot raw accuracy with low alpha
            ax.plot(acc, alpha=0.2, color=COLORS[model_name], linewidth=1)
            
            # Plot moving average
            if len(acc) > window:
                acc_smooth = pd.Series(acc).rolling(window=window, center=True).mean()
                ax.plot(acc_smooth, label=model_name, color=COLORS[model_name], linewidth=2)
            else:
                ax.plot(acc, label=model_name, color=COLORS[model_name], linewidth=2)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Batch Accuracy')
    ax.set_title('Training Accuracy per Batch (with 50-step Moving Average)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('plots/accuracy_per_batch_all_models.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/accuracy_per_batch_all_models.png")
    plt.close()


def plot_flops_efficiency():
    """Plot FLOPs per step comparison - efficiency gains across models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: FLOPs over training steps
    for model_name in MODEL_NAMES:
        data = load_checkpoint_data(model_name)
        if data and data['flops'] is not None:
            flops = data['flops']
            ax_main = axes[0]
            
            # Smooth with moving average for clarity
            flops_smooth = pd.Series(flops).rolling(window=10, center=True).mean()
            ax_main.plot(flops_smooth, label=model_name, color=COLORS[model_name], linewidth=2, marker='o', markersize=3, markevery=100)
    
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('FLOPs per Forward Pass')
    axes[0].set_title('FLOPs Evolution During Training')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # Plot 2: Efficiency comparison (normalized to fixed baseline)
    final_flops = {}
    for model_name in MODEL_NAMES:
        data = load_checkpoint_data(model_name)
        if data and data['flops'] is not None:
            flops = data['flops']
            final_flops[model_name] = flops[-1]  # Last step FLOPs
    
    if final_flops:
        fixed_flops = final_flops.get('Fixed', 1.0)
        efficiency = {model: (1 - val/fixed_flops) * 100 for model, val in final_flops.items() if model != 'Fixed'}
        
        models = list(efficiency.keys())
        reductions = list(efficiency.values())
        
        bars = axes[1].bar(models, reductions, color=[COLORS[m] for m in models], 
                          alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('FLOPs Reduction (%)')
        axes[1].set_title('Computational Efficiency vs Fixed Baseline')
        axes[1].set_ylim([0, max(reductions) * 1.2 if reductions else 100])
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, reductions):
            ax_1 = axes[1]
            ax_1.text(bar.get_x() + bar.get_width()/2, val,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/flops_efficiency_all_models.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/flops_efficiency_all_models.png")
    plt.close()


def plot_per_class_flops_time_series():
    """Plot class-wise FLOPs allocation over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for model_name in MODEL_NAMES:
        data = load_checkpoint_data(model_name)
        if data and data['flops'] is not None:
            # For non-fixed models, try to load classwise data
            classwise_path = None
            if model_name == 'Halting (WS)':
                classwise_path = 'checkpoints/adaptive_ws_flops_per_step_afib_classwise.pt'
            elif model_name == 'Halting (No WS)':
                classwise_path = 'checkpoints/adaptive_flops_per_step_afib_classwise.pt'
            elif model_name == 'Selection':
                classwise_path = 'checkpoints/adaptive_selection_flops_per_step_afib_classwise.pt'
            elif model_name == 'Fixed':
                classwise_path = 'checkpoints/flops_per_step_afib_classwise.pt'
            
            if classwise_path and Path(classwise_path).exists():
                classwise = torch.load(classwise_path, map_location='cpu', weights_only=False)
                
                if 'class0' in classwise and 'class1' in classwise:
                    class0_flops = np.array(classwise['class0'])
                    class1_flops = np.array(classwise['class1'])
                    
                    # Smooth with larger window for cleaner lines
                    c0_smooth = pd.Series(class0_flops).rolling(window=50, center=True).mean()
                    c1_smooth = pd.Series(class1_flops).rolling(window=50, center=True).mean()
                    
                    ax.plot(c0_smooth, label=f'{model_name} (Normal)', 
                           color=COLORS[model_name], linestyle='--', linewidth=2.5, alpha=0.8)
                    ax.plot(c1_smooth, label=f'{model_name} (Abnormal)', 
                           color=COLORS[model_name], linestyle='-', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('FLOPs per Class', fontsize=12)
    ax.set_title('Class-wise FLOPs Allocation During Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    plt.tight_layout()
    plt.savefig('plots/per_class_flops_time_series.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/per_class_flops_time_series.png")
    plt.close()


def plot_per_class_flops_comparison():
    """Plot final class-wise FLOPs comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    final_class_flops = {model: {'normal': None, 'abnormal': None} for model in MODEL_NAMES}
    
    for model_name in MODEL_NAMES:
        classwise_path = None
        if model_name == 'Halting (WS)':
            classwise_path = 'checkpoints/adaptive_ws_flops_per_step_afib_classwise.pt'
        elif model_name == 'Halting (No WS)':
            classwise_path = 'checkpoints/adaptive_flops_per_step_afib_classwise.pt'
        elif model_name == 'Selection':
            classwise_path = 'checkpoints/adaptive_selection_flops_per_step_afib_classwise.pt'
        elif model_name == 'Fixed':
            classwise_path = 'checkpoints/flops_per_step_afib_classwise.pt'
        
        if classwise_path and Path(classwise_path).exists():
            classwise = torch.load(classwise_path, map_location='cpu', weights_only=False)
            if 'class0' in classwise and 'class1' in classwise:
                class0_arr = np.array(classwise['class0'])
                class1_arr = np.array(classwise['class1'])
                final_class_flops[model_name]['normal'] = np.nanmean(class0_arr[-100:]) if len(class0_arr) > 100 else np.nanmean(class0_arr)
                final_class_flops[model_name]['abnormal'] = np.nanmean(class1_arr[-100:]) if len(class1_arr) > 100 else np.nanmean(class1_arr)
    
    # Prepare data for bar plot
    models_with_data = [m for m in MODEL_NAMES if final_class_flops[m]['normal'] is not None]
    normals = [final_class_flops[m]['normal'] for m in models_with_data]
    abnormals = [final_class_flops[m]['abnormal'] for m in models_with_data]
    
    x = np.arange(len(models_with_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, normals, width, label='Normal', 
                   color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, abnormals, width, label='Abnormal', 
                   color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Average FLOPs', fontsize=12)
    ax.set_title('Final Class-wise FLOPs Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_with_data, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/per_class_flops_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/per_class_flops_comparison.png")
    plt.close()
    print("Saved: plots/per_class_flops_all_models.png")
    plt.close()


def plot_model_comparison_summary():
    """Summary comparison bar chart of all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    summary_data = {model: {} for model in MODEL_NAMES}
    
    for model_name in MODEL_NAMES:
        data = load_checkpoint_data(model_name)
        if data and data['metrics']:
            metrics = data['metrics']
            summary_data[model_name]['accuracy'] = metrics.get('last_val_acc', 0.0)
            summary_data[model_name]['params'] = metrics.get('total_params', 0)
            summary_data[model_name]['inference_time'] = metrics.get('avg_train_infer_time_ms', 0)
            
            # Get final FLOPs
            if data['flops'] is not None:
                summary_data[model_name]['final_flops'] = data['flops'][-1] if len(data['flops']) > 0 else metrics.get('full_depth_flops', 0)
            else:
                summary_data[model_name]['final_flops'] = metrics.get('full_depth_flops', 0)
        else:
            # Default values if metrics not found
            summary_data[model_name]['accuracy'] = 0.0
            summary_data[model_name]['params'] = 0
            summary_data[model_name]['inference_time'] = 0
            summary_data[model_name]['final_flops'] = 0
    
    # Plot 1: Validation Accuracy
    models_with_data = [m for m in MODEL_NAMES if summary_data[m]['accuracy'] > 0]
    if not models_with_data:
        models_with_data = MODEL_NAMES
    accs = [summary_data[m]['accuracy'] for m in models_with_data]
    bars = axes[0, 0].bar(models_with_data, accs, color=[COLORS[m] for m in models_with_data], alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].set_title('Final Validation Accuracy')
    axes[0, 0].set_ylim([0.8, 1.0])
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Final FLOPs (normalized to Fixed)
    fixed_final_flops = summary_data['Fixed']['final_flops']
    if fixed_final_flops > 0:
        flops_ratios = [summary_data[m]['final_flops'] / fixed_final_flops for m in models_with_data]
    else:
        flops_ratios = [1.0] * len(models_with_data)
    bars = axes[0, 1].bar(models_with_data, flops_ratios, color=[COLORS[m] for m in models_with_data], alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Fixed Baseline')
    axes[0, 1].set_ylabel('FLOPs Ratio (relative to Fixed)')
    axes[0, 1].set_title('Computational Efficiency')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=15)
    axes[0, 1].legend()
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.2f}x', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Inference Time
    inf_times = [summary_data[m]['inference_time'] for m in models_with_data]
    bars = axes[1, 0].bar(models_with_data, inf_times, color=[COLORS[m] for m in models_with_data], alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Inference Time (ms)')
    axes[1, 0].set_title('Average Inference Time')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Parameter Count
    params = [summary_data[m]['params'] / 1e6 for m in models_with_data]  # Convert to millions
    bars = axes[1, 1].bar(models_with_data, params, color=[COLORS[m] for m in models_with_data], alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Parameters (Millions)')
    axes[1, 1].set_title('Model Size')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=15)
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.1f}M', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/model_comparison_summary.png")
    plt.close()


def main():
    """Generate all comparison plots."""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating comparison plots for all 4 models")
    print("="*60)
    
    print("\n1. Creating accuracy per batch plot...")
    plot_accuracy_per_batch()
    
    print("\n2. Creating FLOPs efficiency plot...")
    plot_flops_efficiency()
    
    print("\n3. Creating per-class FLOPs time series plot...")
    plot_per_class_flops_time_series()
    
    print("\n4. Creating per-class FLOPs comparison plot...")
    plot_per_class_flops_comparison()
    
    print("\n5. Creating model comparison summary...")
    plot_model_comparison_summary()
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
