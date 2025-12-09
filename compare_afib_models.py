"""
Compare four AFIB models by loading training metrics from checkpoints.

NOTE: These are training/validation metrics, not test metrics.
For complete evaluation on test data, use: python test_*_afib.py scripts.

Model naming convention:
- 'ws' in filename = WITH weight sharing (shared layer weights, fewer parameters ~3.5M)
- no 'ws' in filename = WITHOUT weight sharing (independent layer weights, more parameters ~4.5M)
"""
import torch
import os


def load_metrics(path):
    """Load metrics dict from saved torch file."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except:
        return None


def load_model_metrics(model_name):
    """Load training metrics for a model from checkpoint.
    
    Naming convention:
    - 'ws' in filename = WITH weight sharing (shared layer weights)
    - no 'ws' = WITHOUT weight sharing (independent layer weights)
    """
    paths = {
        'Fixed': 'checkpoints/fixed_afib_metrics.pt',
        'Halting (WS)': 'checkpoints/adaptive_ws_afib_metrics.pt',  # WITH weight sharing
        'Halting (No WS)': 'checkpoints/adaptive_afib_metrics.pt',  # WITHOUT weight sharing
        'Selection': 'checkpoints/adaptive_selection_afib_metrics.pt',
    }
    
    path = paths.get(model_name)
    if not path:
        return None
    
    return load_metrics(path)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print("=" * 70)
    print("  AFIB: FIXED vs ADAPTIVE(WS) vs ADAPTIVE(NO WS) vs SELECTION")
    print("=" * 70)

    # Load training metrics for all 4 models
    fixed_metrics = load_model_metrics('Fixed')
    halt_ws_metrics = load_model_metrics('Halting (WS)')
    halt_nws_metrics = load_model_metrics('Halting (No WS)')
    sel_metrics = load_model_metrics('Selection')

    missing = []
    if fixed_metrics is None:
        missing.append("checkpoints/fixed_afib_metrics.pt")
    if halt_ws_metrics is None:
        missing.append("checkpoints/adaptive_afib_metrics.pt")
    if halt_nws_metrics is None:
        missing.append("checkpoints/adaptive_ws_afib_metrics.pt")
    if sel_metrics is None:
        missing.append("checkpoints/adaptive_selection_afib_metrics.pt")

    if missing:
        print("\nERROR: Missing metric files:")
        for path in missing:
            print(f"  {path}")
        print("\nAvailable metrics files:")
        import glob
        for f in sorted(glob.glob("checkpoints/*afib_metrics.pt")):
            print(f"  {f}")
        return
    
    # ========== ACCURACY COMPARISON ==========
    print_section("1. TRAINING METRICS")

    fixed_acc = fixed_metrics.get('last_val_acc', 0.0)
    halt_ws_acc = halt_ws_metrics.get('last_val_acc', 0.0)
    halt_nws_acc = halt_nws_metrics.get('last_val_acc', 0.0)
    sel_acc = sel_metrics.get('last_val_acc', 0.0)

    print(f"  Final Validation Accuracy:")
    print(f"  Fixed-depth:          {fixed_acc:.4f}")
    print(f"  Halting (WS):         {halt_ws_acc:.4f} (Δ vs fixed {halt_ws_acc - fixed_acc:+.4f})")
    print(f"  Halting (No WS):      {halt_nws_acc:.4f} (Δ vs fixed {halt_nws_acc - fixed_acc:+.4f})")
    print(f"  Selection:            {sel_acc:.4f} (Δ vs fixed {sel_acc - fixed_acc:+.4f})")

    fixed_loss = fixed_metrics.get('last_val_loss', 0.0)
    halt_ws_loss = halt_ws_metrics.get('last_val_loss', 0.0)
    halt_nws_loss = halt_nws_metrics.get('last_val_loss', 0.0)
    sel_loss = sel_metrics.get('last_val_loss', 0.0)
    print(f"\n  Final Validation Loss:")
    print(f"    Fixed:     {fixed_loss:.4f} | Halting(WS): {halt_ws_loss:.4f} | Halting(NoWS): {halt_nws_loss:.4f} | Selection: {sel_loss:.4f}")

    print(f"\n  Model Parameters:")
    for model_name in ['Fixed', 'Adaptive Halting (WS)', 'Adaptive Halting (No WS)', 'Adaptive Selection']:
        if model_name == 'Fixed':
            m = fixed_metrics
        elif model_name == 'Adaptive Halting (WS)':
            m = halt_ws_metrics
        elif model_name == 'Adaptive Halting (No WS)':
            m = halt_nws_metrics
        else:
            m = sel_metrics
        params = m.get('total_params', 0)
        print(f"    {model_name:26} {params:,} params")
    
    # ========== MODEL ARCHITECTURE ==========
    print_section("2. MODEL ARCHITECTURE & FLOPS")

    print(f"  Full-depth FLOPs per forward pass:")
    fixed_flops = fixed_metrics.get('flops_per_forward', 0.0)
    halt_ws_flops = halt_ws_metrics.get('flops_per_forward', 0.0)
    halt_nws_flops = halt_nws_metrics.get('flops_per_forward', 0.0)
    sel_flops = sel_metrics.get('flops_per_forward', 0.0)
    
    print(f"    Fixed (always full):         {fixed_flops:.3e}")
    print(f"    Adaptive Halting (WS):       {halt_ws_flops:.3e}")
    print(f"    Adaptive Halting (No WS):    {halt_nws_flops:.3e}")
    print(f"    Adaptive Selection:          {sel_flops:.3e}")

    print(f"\n  Model Configuration:")
    print(f"    Fixed:           {fixed_metrics.get('num_layers')} layers, {fixed_metrics.get('total_params'):,} params")
    print(f"    Adaptive (WS):   {halt_ws_metrics.get('num_layers')} layers, {halt_ws_metrics.get('total_params'):,} params")
    print(f"    Adaptive (No WS):{halt_nws_metrics.get('num_layers')} layers, {halt_nws_metrics.get('total_params'):,} params")
    print(f"    Selection:       {sel_metrics.get('num_layers')} layers, {sel_metrics.get('total_params'):,} params")
    
    # ========== EFFECTIVE FLOPs ==========
    print_section("3. INFERENCE TIME & EFFICIENCY")
    print("   (Requires running: python test_*_afib.py for full metrics)")

    print(f"\n  Average training inference time (ms):")
    fixed_time = fixed_metrics.get('avg_train_infer_time_ms', 0.0)
    halt_ws_time = halt_ws_metrics.get('avg_train_infer_time_ms', 0.0)
    halt_nws_time = halt_nws_metrics.get('avg_train_infer_time_ms', 0.0)
    sel_time = sel_metrics.get('avg_train_infer_time_ms', 0.0)
    
    print(f"    Fixed:                 {fixed_time:.2f}")
    print(f"    Adaptive Halting (WS): {halt_ws_time:.2f} (Δ vs fixed {halt_ws_time - fixed_time:+.2f} ms)")
    print(f"    Adaptive Halting (No WS): {halt_nws_time:.2f} (Δ vs fixed {halt_nws_time - fixed_time:+.2f} ms)")
    print(f"    Adaptive Selection:    {sel_time:.2f} (Δ vs fixed {sel_time - fixed_time:+.2f} ms)")
    
    # ========== INFERENCE TIME ==========
    print_section("4. NEXT STEPS")

    print(f"\n  To compute full test metrics including:")
    print(f"    - Per-class test accuracy")
    print(f"    - Compute allocation (depth distribution)")
    print(f"    - Per-sample inference time")
    print(f"    - Full per-class FLOPs analysis")
    print(f"\n  Run the test scripts:")
    print(f"    python test_fixed_afib.py")
    print(f"    python test_adaptive_afib.py")
    print(f"    python test_adaptive_ws_afib.py")
    print(f"    python test_selection_afib.py")
    print(f"\n  Then regenerate this comparison for complete analysis.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
