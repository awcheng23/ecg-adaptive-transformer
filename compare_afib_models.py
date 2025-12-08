"""
Compare three AFIB models: Fixed-depth, Adaptive-Halting, Adaptive-Selection.
Loads test metrics from all models and reports hypothesis-relevant comparisons.

Hypotheses (apply to both adaptive models):
1. Adaptive models should use less computation (lower FLOPs/depth/compute-fraction) on simple/normal samples
2. Adaptive models should allocate more computation to abnormal/complex samples
3. Overall accuracy should be competitive with fixed
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
    print("  AFIB FIXED vs ADAPTIVE-HALTING vs ADAPTIVE-SELECTION")
    print("=" * 70)

    # Load test metrics
    fixed_metrics = load_metrics("checkpoints/fixed_afib_test_metrics.pt")
    halt_metrics = load_metrics("checkpoints/adaptive_afib_test_metrics.pt")
    sel_metrics = load_metrics("checkpoints/adaptive_selection_afib_test_metrics.pt")

    missing = []
    if fixed_metrics is None:
        missing.append("python test_fixed_afib.py")
    if halt_metrics is None:
        missing.append("python test_adaptive_afib.py")
    if sel_metrics is None:
        missing.append("python test_selection_afib.py")

    if missing:
        print("\nERROR: Missing metrics. Run:")
        for cmd in missing:
            print(f"  {cmd}")
        return
    
    # ========== ACCURACY COMPARISON ==========
    print_section("1. ACCURACY")

    fixed_acc = fixed_metrics.get('test_acc', 0.0)
    halt_acc = halt_metrics.get('test_acc', 0.0)
    sel_acc = sel_metrics.get('test_acc', 0.0)

    print(f"  Fixed-depth:        {fixed_acc:.4f}")
    print(f"  Adaptive-Halting:   {halt_acc:.4f} (Δ vs fixed {halt_acc - fixed_acc:+.4f})")
    print(f"  Adaptive-Selection: {sel_acc:.4f} (Δ vs fixed {sel_acc - fixed_acc:+.4f})")

    fixed_loss = fixed_metrics.get('test_loss', 0.0)
    halt_loss = halt_metrics.get('test_loss', 0.0)
    sel_loss = sel_metrics.get('test_loss', 0.0)
    print(f"\n  Test Loss: fixed={fixed_loss:.4f} | halting={halt_loss:.4f} | selection={sel_loss:.4f}")

    # Per-class accuracy
    print(f"\n  Per-Class Accuracy (Normal / Abnormal):")
    fixed_c0_acc = fixed_metrics.get('class_0_acc', 0.0)
    fixed_c1_acc = fixed_metrics.get('class_1_acc', 0.0)
    halt_c0_acc = halt_metrics.get('class_0_acc', 0.0)
    halt_c1_acc = halt_metrics.get('class_1_acc', 0.0)
    sel_c0_acc = sel_metrics.get('class_0_acc', 0.0)
    sel_c1_acc = sel_metrics.get('class_1_acc', 0.0)

    print(f"    Fixed:              {fixed_c0_acc:.4f} / {fixed_c1_acc:.4f}")
    print(f"    Adaptive-Halting:   {halt_c0_acc:.4f} / {halt_c1_acc:.4f}")
    print(f"    Adaptive-Selection: {sel_c0_acc:.4f} / {sel_c1_acc:.4f}")
    
    # ========== HYPOTHESIS 1 & 2: DEPTH & COMPUTE ON NORMAL vs ABNORMAL ==========
    print_section("2. COMPUTE ALLOCATION (Hypothesis 1 & 2)")
    print("   Hypothesis: Abnormal samples should use MORE compute than Normal")

    # Halting model: depth-based compute
    halt_c0_depth = halt_metrics.get('class_0_depth', 0.0)
    halt_c1_depth = halt_metrics.get('class_1_depth', 0.0)
    halt_layers = halt_metrics.get('num_layers', 4)

    print(f"\n  Adaptive-Halting depth (out of {halt_layers}):")
    print(f"    Normal:   {halt_c0_depth:.3f}")
    print(f"    Abnormal: {halt_c1_depth:.3f}  | Ratio (Abn/Norm): {halt_c1_depth/halt_c0_depth if halt_c0_depth > 0 else 0:.3f}x")

    # Selection model: compute fractions (FLOPs-normalized)
    sel_c0_comp = sel_metrics.get('class_0_compute_fraction', 0.0)
    sel_c1_comp = sel_metrics.get('class_1_compute_fraction', 0.0)
    print(f"\n  Adaptive-Selection compute fraction (1.0 = full model):")
    print(f"    Normal:   {sel_c0_comp:.3f}")
    print(f"    Abnormal: {sel_c1_comp:.3f}  | Ratio (Abn/Norm): {sel_c1_comp/sel_c0_comp if sel_c0_comp > 0 else 0:.3f}x")

    # Depth distribution (halting only)
    c0_median = halt_metrics.get('depth_0_median', 0.0)
    c1_median = halt_metrics.get('depth_1_median', 0.0)
    c0_std = halt_metrics.get('depth_0_std', 0.0)
    c1_std = halt_metrics.get('depth_1_std', 0.0)
    full_c0 = halt_metrics.get('full_depth_ratio_0', 0.0)
    full_c1 = halt_metrics.get('full_depth_ratio_1', 0.0)

    print(f"\n  Halting depth distribution:")
    print(f"    Normal:   median={c0_median:.3f}, std={c0_std:.3f}, full-depth={full_c0*100:.1f}%")
    print(f"    Abnormal: median={c1_median:.3f}, std={c1_std:.3f}, full-depth={full_c1*100:.1f}%")
    
    # ========== EFFECTIVE FLOPs ==========
    print_section("3. COMPUTATIONAL EFFICIENCY")
    print("   Hypothesis: Adaptive should use fewer FLOPs overall")

    fixed_flops = fixed_metrics.get('flops_per_forward', 0.0)

    # Halting FLOPs
    halt_full_flops = halt_metrics.get('full_depth_flops', 0.0)
    halt_eff_overall = halt_metrics.get('eff_flops_overall', 0.0)
    halt_eff_c0 = halt_metrics.get('eff_flops_class_0', 0.0)
    halt_eff_c1 = halt_metrics.get('eff_flops_class_1', 0.0)

    # Selection FLOPs (compute fraction * profiled full)
    sel_full_flops = sel_metrics.get('full_depth_flops', 0.0)
    sel_eff_overall = sel_metrics.get('eff_flops_overall', 0.0)
    sel_eff_c0 = sel_metrics.get('eff_flops_class_0', 0.0)
    sel_eff_c1 = sel_metrics.get('eff_flops_class_1', 0.0)

    print(f"\n  Full-depth FLOPs:")
    print(f"    Fixed (always full):        {fixed_flops:.3e}")
    print(f"    Adaptive-Halting (full):    {halt_full_flops:.3e}")
    print(f"    Adaptive-Selection (full):  {sel_full_flops:.3e}")

    print(f"\n  Effective FLOPs (based on actual compute):")
    print(f"    Halting overall:   {halt_eff_overall:.3e} | Normal {halt_eff_c0:.3e} | Abnormal {halt_eff_c1:.3e}")
    print(f"    Selection overall: {sel_eff_overall:.3e} | Normal {sel_eff_c0:.3e} | Abnormal {sel_eff_c1:.3e}")

    def reduction(eff, full):
        return (1 - eff / full) * 100 if full else 0.0

    print(f"\n  FLOPs Reduction vs full-depth within each adaptive model:")
    print(f"    Halting:   overall {reduction(halt_eff_overall, halt_full_flops):.1f}% | Normal {reduction(halt_eff_c0, halt_full_flops):.1f}% | Abnormal {reduction(halt_eff_c1, halt_full_flops):.1f}%")
    print(f"    Selection: overall {reduction(sel_eff_overall, sel_full_flops):.1f}% | Normal {reduction(sel_eff_c0, sel_full_flops):.1f}% | Abnormal {reduction(sel_eff_c1, sel_full_flops):.1f}%")

    if fixed_flops:
        print(f"\n  Compute vs Fixed (lower is better):")
        print(f"    Halting overall:   {halt_eff_overall/fixed_flops:.3f}x of fixed")
        print(f"    Selection overall: {sel_eff_overall/fixed_flops:.3f}x of fixed")
    
    # ========== INFERENCE TIME ==========
    print_section("4. INFERENCE TIME")

    fixed_time = fixed_metrics.get('avg_test_infer_time_ms', 0.0)
    halt_time = halt_metrics.get('avg_test_infer_time_ms', 0.0)
    sel_time = sel_metrics.get('avg_test_infer_time_ms', 0.0)

    print(f"\n  Average batch inference time (ms):")
    print(f"    Fixed:    {fixed_time:.2f}")
    print(f"    Halting:  {halt_time:.2f} (Δ vs fixed {halt_time - fixed_time:+.2f} ms)")
    print(f"    Selection:{sel_time:.2f} (Δ vs fixed {sel_time - fixed_time:+.2f} ms)")

    # Per-sample time
    fixed_c0_time = fixed_metrics.get('class_0_time_ms', 0.0)
    fixed_c1_time = fixed_metrics.get('class_1_time_ms', 0.0)
    halt_c0_time = halt_metrics.get('class_0_time_ms', 0.0)
    halt_c1_time = halt_metrics.get('class_1_time_ms', 0.0)
    sel_c0_time = sel_metrics.get('class_0_time_ms', 0.0)
    sel_c1_time = sel_metrics.get('class_1_time_ms', 0.0)

    print(f"\n  Per-sample inference time (ms):")
    print(f"    Normal:   fixed={fixed_c0_time:.3f} | halting={halt_c0_time:.3f} | selection={sel_c0_time:.3f}")
    if fixed_c0_time > 0:
        if halt_c0_time > 0:
            print(f"      Speedup halting vs fixed:   {fixed_c0_time/halt_c0_time:.2f}x")
        if sel_c0_time > 0:
            print(f"      Speedup selection vs fixed: {fixed_c0_time/sel_c0_time:.2f}x")

    print(f"    Abnormal: fixed={fixed_c1_time:.3f} | halting={halt_c1_time:.3f} | selection={sel_c1_time:.3f}")
    if fixed_c1_time > 0:
        if halt_c1_time > 0:
            print(f"      Speedup halting vs fixed:   {fixed_c1_time/halt_c1_time:.2f}x")
        if sel_c1_time > 0:
            print(f"      Speedup selection vs fixed: {fixed_c1_time/sel_c1_time:.2f}x")
    
    # ========== HYPOTHESIS VALIDATION ==========
    print_section("5. HYPOTHESIS VALIDATION")

    def pass_h1(model_norm, model_abn):
        return model_norm < model_abn

    def pass_h3(acc):
        return abs(fixed_acc - acc) <= 0.02

    def pass_h4(time_norm):
        return time_norm <= fixed_c0_time if fixed_c0_time > 0 else False

    h1_halting = pass_h1(halt_c0_depth, halt_c1_depth)
    h1_selection = pass_h1(sel_c0_comp, sel_c1_comp)

    h3_halting = pass_h3(halt_acc)
    h3_selection = pass_h3(sel_acc)

    h4_halting = pass_h4(halt_c0_time)
    h4_selection = pass_h4(sel_c0_time)

    print("\n  H1 (Less compute on Normal):")
    print(f"    Halting:   {'PASS' if h1_halting else 'FAIL'} (depth norm={halt_c0_depth:.3f}, abn={halt_c1_depth:.3f})")
    print(f"    Selection: {'PASS' if h1_selection else 'FAIL'} (compute norm={sel_c0_comp:.3f}, abn={sel_c1_comp:.3f})")

    print("\n  H2 (More compute on Abnormal):")
    print(f"    Halting:   {'PASS' if h1_halting else 'FAIL'}")
    print(f"    Selection: {'PASS' if h1_selection else 'FAIL'}")

    print("\n  H3 (Competitive accuracy, ±2% of fixed):")
    print(f"    Halting:   {'PASS' if h3_halting else 'MARGINAL/FAIL'} (Δ={halt_acc - fixed_acc:+.4f})")
    print(f"    Selection: {'PASS' if h3_selection else 'MARGINAL/FAIL'} (Δ={sel_acc - fixed_acc:+.4f})")

    print("\n  H4 (Faster on Normal):")
    print(f"    Halting:   {'PASS' if h4_halting else 'FAIL'} (normal time {halt_c0_time:.3f} ms vs fixed {fixed_c0_time:.3f} ms)")
    print(f"    Selection: {'PASS' if h4_selection else 'FAIL'} (normal time {sel_c0_time:.3f} ms vs fixed {fixed_c0_time:.3f} ms)")
    
    # ========== SUMMARY ==========
    print_section("6. SUMMARY")

    print(f"\n  Model Params (layers shown; params similar across models):")
    print(f"    Fixed:    {fixed_metrics.get('num_layers')} layers (always full)")
    print(f"    Halting:  {halt_metrics.get('num_layers')} layers, halt_eps={halt_metrics.get('halt_epsilon')}")
    print(f"    Selection:{sel_metrics.get('num_layers')} layers, gumbel_tau={sel_metrics.get('gumbel_tau')}")

    def count_passes(flags):
        return sum(1 for f in flags if f)

    print(f"\n  Hypotheses verified (out of 4):")
    print(f"    Halting:   {count_passes([h1_halting, h1_halting, h3_halting, h4_halting])}/4")
    print(f"    Selection: {count_passes([h1_selection, h1_selection, h3_selection, h4_selection])}/4")

    print(f"\n  Key Takeaways:")
    if halt_full_flops:
        savings_pct = (1 - halt_eff_overall / halt_full_flops) * 100
        print(f"    Halting:    {savings_pct:.1f}% FLOPs reduction overall; selective depth by class.")
    if sel_full_flops:
        savings_pct = (1 - sel_eff_overall / sel_full_flops) * 100
        print(f"    Selection:  {savings_pct:.1f}% FLOPs reduction overall; learns patch/head/block sparsity.")
    if fixed_flops and sel_eff_overall and halt_eff_overall:
        best = min([(halt_eff_overall, 'Halting'), (sel_eff_overall, 'Selection')], key=lambda x: x[0])
        print(f"    Most efficient overall compute: {best[1]} (vs fixed baseline).")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
