#!/usr/bin/env python3
"""
Quick reference for the 4 ECG adaptive transformer models.
Run this to see model comparison and checkpoint status.
"""
import os
import torch
from pathlib import Path


def check_checkpoint(path, name):
    """Check if a checkpoint exists and report its size."""
    if Path(path).exists():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return f"✓ {size_mb:.1f} MB"
    else:
        return "✗ Not found"


def main():
    print("\n" + "="*70)
    print("ECG ADAPTIVE TRANSFORMER - 4-MODEL COMPARISON")
    print("="*70)
    
    models = {
        'Fixed Baseline': {
            'description': 'Standard 4-layer transformer, constant computation',
            'architecture': 'FixedCNNTransformer',
            'model_file': 'models/fixed_transformer.py',
            'key_feature': 'No adaptation - constant FLOPs',
            'parameters': '~3.5M',
            'checkpoint': 'checkpoints/fixed_transformer_afib.pth',
            'metrics': 'checkpoints/fixed_afib_metrics.pt',
            'flops': 'checkpoints/flops_per_step_afib.pt',
        },
        'Adaptive Halting (WS)': {
            'description': 'Adaptive-depth with weight sharing across layers',
            'architecture': 'AdaptiveCNNTransformer (halting)',
            'model_file': 'models/adaptive_transformer.py',
            'key_feature': 'ACT-style halting, shared layer weights',
            'parameters': '~3.5M',
            'checkpoint': 'checkpoints/adaptive_transformer_afib.pth',
            'metrics': 'checkpoints/adaptive_afib_metrics.pt',
            'flops': 'checkpoints/adaptive_flops_per_step_afib.pt',
        },
        'Adaptive Halting (No WS)': {
            'description': 'Adaptive-depth WITHOUT weight sharing (independent layers)',
            'architecture': 'AdaptiveCNNTransformer (no weight sharing)',
            'model_file': 'models/adaptive_transformer_ws.py',
            'key_feature': 'ACT halting + independent layer weights',
            'parameters': '~4.5M',
            'checkpoint': 'checkpoints/adaptive_transformer_ws_afib.pth',
            'metrics': 'checkpoints/adaptive_ws_afib_metrics.pt',
            'flops': 'checkpoints/adaptive_ws_flops_per_step_afib.pt',
        },
        'Adaptive Selection': {
            'description': 'Adaptive computation via learned gating (patches/heads/blocks)',
            'architecture': 'SelectiveTransformer',
            'model_file': 'models/adaptive_selection_transformer.py',
            'key_feature': 'Gating mechanism, skip computation',
            'parameters': '~4.0M',
            'checkpoint': 'checkpoints/adaptive_selection_transformer_afib.pth',
            'metrics': 'checkpoints/adaptive_selection_afib_metrics.pt',
            'flops': 'checkpoints/adaptive_selection_flops_per_step_afib.pt',
        },
    }
    
    for i, (model_name, info) in enumerate(models.items(), 1):
        print(f"\n{i}. {model_name.upper()}")
        print("-" * 70)
        print(f"   Description:  {info['description']}")
        print(f"   Architecture: {info['architecture']}")
        print(f"   Key Feature:  {info['key_feature']}")
        print(f"   Parameters:   {info['parameters']}")
        print(f"\n   Checkpoints:")
        print(f"     Weights:    {check_checkpoint(info['checkpoint'], model_name)}")
        print(f"     Metrics:    {check_checkpoint(info['metrics'], model_name)}")
        print(f"     FLOPs:      {check_checkpoint(info['flops'], model_name)}")
    
    print("\n" + "="*70)
    print("TRAINING SCRIPTS")
    print("="*70)
    
    scripts = [
        ('train_fixed_afib.py', 'Fixed Baseline', '~10 min'),
        ('train_adaptive_afib.py', 'Adaptive Halting (WS)', '~10 min'),
        ('train_adaptive_ws_afib.py', 'Adaptive Halting (No WS)', '~15 min'),
        ('train_selection_afib.py', 'Adaptive Selection', '~12 min'),
    ]
    
    for script, model, time in scripts:
        status = "✓" if Path(script).exists() else "✗"
        print(f"{status} {script:30s} → {model:30s} ({time})")
    
    print("\n" + "="*70)
    print("TESTING & PLOTTING")
    print("="*70)
    
    print("\nTest scripts (generates test-set metrics):")
    test_scripts = [
        'test_fixed_afib.py',
        'test_adaptive_afib.py',
        'test_adaptive_ws_afib.py',
        'test_selection_afib.py',
    ]
    for script in test_scripts:
        status = "✓" if Path(script).exists() else "✗"
        print(f"  {status} {script}")
    
    print("\nPlotting script (creates 4-model comparison plots):")
    status = "✓" if Path('plot_all_models.py').exists() else "✗"
    print(f"  {status} plot_all_models.py")
    
    print("\n" + "="*70)
    print("WORKFLOW")
    print("="*70)
    print("""
Step 1: Train all 4 models (if needed)
  $ python train_fixed_afib.py
  $ python train_adaptive_afib.py
  $ python train_adaptive_ws_afib.py
  $ python train_selection_afib.py

Step 2: Test all 4 models
  $ python test_fixed_afib.py
  $ python test_adaptive_afib.py
  $ python test_adaptive_ws_afib.py
  $ python test_selection_afib.py

Step 3: Generate comparison plots
  $ python plot_all_models.py

Output plots:
  plots/accuracy_per_batch_all_models.png      - Training convergence
  plots/flops_efficiency_all_models.png        - Computational efficiency
  plots/per_class_flops_all_models.png         - Normal vs Abnormal allocation
  plots/model_comparison_summary.png           - Summary dashboard
    """)
    
    print("="*70)
    print("For detailed setup and hypothesis guide, see: 4_MODEL_SETUP.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
