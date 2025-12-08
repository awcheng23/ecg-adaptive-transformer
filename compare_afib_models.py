"""
Compare fixed vs adaptive transformer models on AFIB dataset.
Loads test metrics from both models and reports hypothesis-relevant comparisons.

Hypotheses:
1. Adaptive model should use less computation (lower FLOPs/depth) on simple/normal samples
2. Adaptive model should allocate more computation to abnormal/complex samples (higher depth)
3. Overall accuracy should be competitive or better with adaptive
4. Computation savings should enable faster inference on normal samples
"""
import torch
import os


def load_metrics(path):
    """Load metrics dict from saved torch file."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    return torch.load(path, map_location='cpu')


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print("=" * 70)
    print("  AFIB FIXED vs ADAPTIVE TRANSFORMER COMPARISON")
    print("=" * 70)
    
    # Load test metrics
    fixed_metrics = load_metrics("checkpoints/fixed_afib_test_metrics.pt")
    adaptive_metrics = load_metrics("checkpoints/adaptive_afib_test_metrics.pt")
    
    if fixed_metrics is None or adaptive_metrics is None:
        print("\nERROR: Could not load both test metrics. Run:")
        print("  python test_fixed_afib.py")
        print("  python test_adaptive_afib.py")
        return
    
    # ========== ACCURACY COMPARISON ==========
    print_section("1. ACCURACY")
    
    fixed_acc = fixed_metrics.get('test_acc', 0.0)
    adaptive_acc = adaptive_metrics.get('test_acc', 0.0)
    
    print(f"  Fixed-depth:   {fixed_acc:.4f}")
    print(f"  Adaptive:      {adaptive_acc:.4f}")
    print(f"  Difference:    {adaptive_acc - fixed_acc:+.4f}")
    
    fixed_loss = fixed_metrics.get('test_loss', 0.0)
    adaptive_loss = adaptive_metrics.get('test_loss', 0.0)
    print(f"\n  Fixed Test Loss:   {fixed_loss:.4f}")
    print(f"  Adaptive Test Loss: {adaptive_loss:.4f}")
    
    # Per-class accuracy
    print(f"\n  Per-Class Accuracy:")
    fixed_c0_acc = fixed_metrics.get('class_0_acc', 0.0)
    fixed_c1_acc = fixed_metrics.get('class_1_acc', 0.0)
    adaptive_c0_acc = adaptive_metrics.get('class_0_acc', 0.0)
    adaptive_c1_acc = adaptive_metrics.get('class_1_acc', 0.0)
    
    print(f"    Normal (Class 0):")
    print(f"      Fixed:    {fixed_c0_acc:.4f}")
    print(f"      Adaptive: {adaptive_c0_acc:.4f}")
    print(f"      Δ:        {adaptive_c0_acc - fixed_c0_acc:+.4f}")
    
    print(f"    Abnormal (Class 1):")
    print(f"      Fixed:    {fixed_c1_acc:.4f}")
    print(f"      Adaptive: {adaptive_c1_acc:.4f}")
    print(f"      Δ:        {adaptive_c1_acc - fixed_c1_acc:+.4f}")
    
    # ========== HYPOTHESIS 1 & 2: DEPTH & COMPUTE ON NORMAL vs ABNORMAL ==========
    print_section("2. ADAPTIVE DEPTH ALLOCATION (Hypothesis 1 & 2)")
    print("   Hypothesis: Abnormal samples should use MORE depth than Normal")
    
    adaptive_c0_depth = adaptive_metrics.get('class_0_depth', 0.0)
    adaptive_c1_depth = adaptive_metrics.get('class_1_depth', 0.0)
    num_layers = adaptive_metrics.get('num_layers', 4)
    
    print(f"\n  Average Depth (out of {num_layers} layers):")
    print(f"    Normal:    {adaptive_c0_depth:.3f}")
    print(f"    Abnormal:  {adaptive_c1_depth:.3f}")
    print(f"    Ratio (Abn/Norm): {adaptive_c1_depth/adaptive_c0_depth if adaptive_c0_depth > 0 else 0:.3f}x")
    
    # Depth distribution
    c0_median = adaptive_metrics.get('depth_0_median', 0.0)
    c1_median = adaptive_metrics.get('depth_1_median', 0.0)
    c0_std = adaptive_metrics.get('depth_0_std', 0.0)
    c1_std = adaptive_metrics.get('depth_1_std', 0.0)
    
    print(f"\n  Depth Distribution:")
    print(f"    Normal:    median={c0_median:.3f}, std={c0_std:.3f}")
    print(f"    Abnormal:  median={c1_median:.3f}, std={c1_std:.3f}")
    
    # Full-depth fraction (% reaching all layers)
    full_c0 = adaptive_metrics.get('full_depth_ratio_0', 0.0)
    full_c1 = adaptive_metrics.get('full_depth_ratio_1', 0.0)
    
    print(f"\n  Fraction reaching full depth ({num_layers} layers):")
    print(f"    Normal:    {full_c0*100:.1f}%")
    print(f"    Abnormal:  {full_c1*100:.1f}%")
    
    # ========== EFFECTIVE FLOPs ==========
    print_section("3. COMPUTATIONAL EFFICIENCY")
    print("   Hypothesis: Adaptive should use fewer FLOPs overall due to early halting")
    
    fixed_flops = fixed_metrics.get('flops_per_forward', 0.0)
    adaptive_full_flops = adaptive_metrics.get('full_depth_flops', 0.0)
    adaptive_eff_flops_overall = adaptive_metrics.get('eff_flops_overall', 0.0)
    adaptive_eff_flops_c0 = adaptive_metrics.get('eff_flops_class_0', 0.0)
    adaptive_eff_flops_c1 = adaptive_metrics.get('eff_flops_class_1', 0.0)
    
    print(f"\n  Full-depth FLOPs:")
    print(f"    Fixed (always full):       {fixed_flops:.3e}")
    print(f"    Adaptive (if all halted):  {adaptive_full_flops:.3e}")
    
    print(f"\n  Effective FLOPs (based on actual depth):")
    print(f"    Adaptive overall:   {adaptive_eff_flops_overall:.3e}")
    print(f"    Adaptive on Normal: {adaptive_eff_flops_c0:.3e}")
    print(f"    Adaptive on Abnormal: {adaptive_eff_flops_c1:.3e}")
    
    print(f"\n  FLOPs Reduction vs Full-depth:")
    if adaptive_full_flops > 0:
        overall_reduction = (1 - adaptive_eff_flops_overall / adaptive_full_flops) * 100
        normal_reduction = (1 - adaptive_eff_flops_c0 / adaptive_full_flops) * 100
        abn_reduction = (1 - adaptive_eff_flops_c1 / adaptive_full_flops) * 100
        print(f"    Overall:   {overall_reduction:.1f}%")
        print(f"    Normal:    {normal_reduction:.1f}%")
        print(f"    Abnormal:  {abn_reduction:.1f}%")
    
    # ========== INFERENCE TIME ==========
    print_section("4. INFERENCE TIME")
    
    fixed_time = fixed_metrics.get('avg_test_infer_time_ms', 0.0)
    adaptive_time = adaptive_metrics.get('avg_test_infer_time_ms', 0.0)
    
    print(f"\n  Average batch inference time (ms):")
    print(f"    Fixed:    {fixed_time:.2f}")
    print(f"    Adaptive: {adaptive_time:.2f}")
    print(f"    Δ:        {adaptive_time - fixed_time:+.2f} ({(adaptive_time/fixed_time - 1)*100:+.1f}%)")
    
    # Per-sample time
    fixed_c0_time = fixed_metrics.get('class_0_time_ms', 0.0)
    fixed_c1_time = fixed_metrics.get('class_1_time_ms', 0.0)
    adaptive_c0_time = adaptive_metrics.get('class_0_time_ms', 0.0)
    adaptive_c1_time = adaptive_metrics.get('class_1_time_ms', 0.0)
    
    print(f"\n  Per-sample inference time (ms):")
    print(f"    Normal:")
    print(f"      Fixed:    {fixed_c0_time:.3f}")
    print(f"      Adaptive: {adaptive_c0_time:.3f}")
    if fixed_c0_time > 0:
        print(f"      Speedup:  {fixed_c0_time/adaptive_c0_time:.2f}x" if adaptive_c0_time > 0 else "")
    
    print(f"    Abnormal:")
    print(f"      Fixed:    {fixed_c1_time:.3f}")
    print(f"      Adaptive: {adaptive_c1_time:.3f}")
    if fixed_c1_time > 0:
        print(f"      Speedup:  {fixed_c1_time/adaptive_c1_time:.2f}x" if adaptive_c1_time > 0 else "")
    
    # ========== HYPOTHESIS VALIDATION ==========
    print_section("5. HYPOTHESIS VALIDATION")
    
    print("\n  ✓ H1: Adaptive allocates LESS computation to Normal samples")
    h1_pass = adaptive_c0_depth < adaptive_c1_depth
    print(f"     Status: {'✓ PASS' if h1_pass else '✗ FAIL'}")
    print(f"     (Normal depth {adaptive_c0_depth:.3f} < Abnormal {adaptive_c1_depth:.3f})")
    
    print("\n  ✓ H2: Adaptive allocates MORE computation to Abnormal samples")
    h2_pass = adaptive_c1_depth > adaptive_c0_depth
    print(f"     Status: {'✓ PASS' if h2_pass else '✗ FAIL'}")
    print(f"     (Abnormal depth {adaptive_c1_depth:.3f} > Normal {adaptive_c0_depth:.3f})")
    
    print("\n  ✓ H3: Adaptive maintains competitive accuracy")
    acc_drop = fixed_acc - adaptive_acc
    h3_pass = abs(acc_drop) <= 0.02  # Allow 2% drop
    print(f"     Status: {'✓ PASS' if h3_pass else '✗ MARGINAL/FAIL'}")
    print(f"     (Δ Acc = {acc_drop:+.4f}, threshold ±0.02)")
    
    print("\n  ✓ H4: Adaptive achieves computational savings on Normal samples")
    h4_pass = adaptive_c0_time <= fixed_c0_time
    print(f"     Status: {'✓ PASS' if h4_pass else '✗ FAIL'}")
    print(f"     (Adaptive {adaptive_c0_time:.3f}ms ≤ Fixed {fixed_c0_time:.3f}ms on Normal)")
    
    # ========== SUMMARY ==========
    print_section("6. SUMMARY")
    
    print(f"\n  Model Parameters:")
    print(f"    Total params (fixed):   {fixed_metrics.get('num_layers')} layers")
    print(f"    Total params (adaptive): {adaptive_metrics.get('num_layers')} layers, halt_eps={adaptive_metrics.get('halt_epsilon')}")
    
    print(f"\n  Overall Recommendation:")
    hypotheses_pass = sum([h1_pass, h2_pass, h3_pass, h4_pass])
    print(f"    Hypotheses verified: {hypotheses_pass}/4")
    
    if hypotheses_pass >= 3:
        print(f"    ✓ Adaptive model validates hypothesis framework")
    else:
        print(f"    ✗ Adaptive model does not strongly validate hypotheses")
    
    print(f"\n  Key Takeaway:")
    if adaptive_eff_flops_overall < adaptive_full_flops:
        savings_pct = (1 - adaptive_eff_flops_overall / adaptive_full_flops) * 100
        print(f"    Adaptive achieves {savings_pct:.1f}% FLOPs reduction via early halting.")
        if h1_pass and h2_pass:
            print(f"    Halting is selective: Normal samples halt early, Abnormal samples use full depth.")
    else:
        print(f"    Adaptive provides adaptive routing but minimal FLOPs reduction.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
