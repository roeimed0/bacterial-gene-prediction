# EXPERIMENT: GC-adaptive Hybrid threshold -- offline sweep of alpha parameter
# STATUS: active
# RESULT: pending
"""
Uses the pre-computed hybrid_threshold_sweep.csv to find the best alpha for:

    effective_threshold(gc) = base_t - alpha * max(0, gc - gc_floor)

where base_t=0.471 (current production), gc_floor=0.55.

For each genome, the effective threshold is interpolated from the pre-computed
sweep (nearest threshold). Reports mean F1, regressions, and improvements vs
the corrected baseline (F1=74.50%) for each alpha.

No pipeline re-run needed -- pure analysis of existing sweep data.

Run from repo root:
    python scripts/experiments/analyze_hybrid_gc_adaptive.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
CSV_PATH = OUT_DIR / "hybrid_threshold_sweep.csv"

df = pd.read_csv(CSV_PATH)
print(
    f"Loaded: {len(df):,} rows  ({df['acc'].nunique()} genomes x {df['threshold'].nunique()} thresholds)"
)

BASE_T = 0.471
GC_FLOOR = 0.55
REGRESSION_THRESH = 0.05  # pp drop counts as regression

# Build lookup: acc -> {threshold -> {f1, sensitivity, precision}}
lookup = {}
for acc, grp in df.groupby("acc"):
    lookup[acc] = {row["threshold"]: row for _, row in grp.iterrows()}

genomes = sorted(df["acc"].unique())
thresholds = sorted(df["threshold"].unique())

# Alpha values to sweep
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

SEP = "=" * 100
print(f"\n{SEP}")
print(f"GC-ADAPTIVE THRESHOLD SWEEP: threshold = {BASE_T} - alpha * max(0, gc - {GC_FLOOR})")
print(f"  alpha=0 => current production threshold ({BASE_T}) for all genomes")
print(SEP)

print(f"\n  {'alpha':<7}  {'Mean F1':>8}  {'Regress':>8}  {'Improve':>8}  {'dF1':>7}  Notes")
print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  -----")

best_alpha = None
best_f1 = -1.0
best_regress = 999

results_by_alpha = {}

for alpha in ALPHAS:
    f1s = []
    regressions = 0
    improvements = 0
    per_genome = {}

    for acc in genomes:
        gc = df[df["acc"] == acc]["gc_pct"].iloc[0] / 100.0
        eff_t = BASE_T - alpha * max(0.0, gc - GC_FLOOR)
        eff_t = float(np.clip(eff_t, thresholds[0], thresholds[-1]))

        # Find nearest pre-computed threshold
        nearest_t = min(thresholds, key=lambda t: abs(t - eff_t))
        row = lookup[acc][nearest_t]

        f1 = row["f1"]
        base = row["baseline_f1"]
        df_v = f1 - base

        f1s.append(f1)
        per_genome[acc] = {
            "f1": f1,
            "eff_t": eff_t,
            "nearest_t": nearest_t,
            "baseline": base,
            "df": df_v,
            "gc": gc * 100,
        }
        if df_v < -REGRESSION_THRESH:
            regressions += 1
        if df_v > REGRESSION_THRESH:
            improvements += 1

    mean_f1 = np.mean(f1s)
    mean_baseline = np.mean(
        [row["baseline_f1"] for row in lookup[genomes[0]].values() if True][:1]
    )  # placeholder — compute properly
    mean_baseline = df.groupby("acc")["baseline_f1"].first().mean()
    d_f1 = mean_f1 - mean_baseline

    marker = " ✓ ZERO REGRESSIONS" if regressions == 0 else ""
    print(
        f"  {alpha:<7.1f}  {mean_f1:>8.2f}  {regressions:>8}  {improvements:>8}  {d_f1:>+7.2f}{marker}"
    )

    results_by_alpha[alpha] = {
        "mean_f1": mean_f1,
        "regressions": regressions,
        "improvements": improvements,
        "per_genome": per_genome,
    }
    if regressions < best_regress or (regressions == best_regress and mean_f1 > best_f1):
        best_regress = regressions
        best_f1 = mean_f1
        best_alpha = alpha

# ── Detail for best alpha ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"BEST ALPHA: {best_alpha}  (regressions={best_regress}  mean_F1={best_f1:.2f})")
print(SEP)

pg = results_by_alpha[best_alpha]["per_genome"]
print(
    f"\n  {'Genome':<15} {'GC%':>5} {'eff_t':>6} {'nearest':>8} {'Baseline':>9} {'F1':>7} {'dF1':>7}"
)
print("  " + "-" * 65)
for acc in genomes:
    r = pg[acc]
    marker = (
        " +++" if r["df"] > REGRESSION_THRESH else (" <<<" if r["df"] < -REGRESSION_THRESH else "")
    )
    print(
        f"  {acc:<15} {r['gc']:>5.1f} {r['eff_t']:>6.3f} {r['nearest_t']:>8.2f} "
        f"{r['baseline']:>9.2f} {r['f1']:>7.2f} {r['df']:>+7.2f}{marker}"
    )

# ── Sensitivity table for best alpha vs current ────────────────────────────────
print(f"\n{SEP}")
print(f"SENSITIVITY GAIN at alpha={best_alpha}")
print(SEP)
pg0 = results_by_alpha[0.0]["per_genome"]  # current (alpha=0)
pg_b = results_by_alpha[best_alpha]["per_genome"]  # best alpha

for acc in genomes:
    r0 = pg0[acc]
    rb = pg_b[acc]
    r_row = df[(df["acc"] == acc) & (np.abs(df["threshold"] - rb["nearest_t"]) < 0.001)]
    b_row = df[(df["acc"] == acc) & (np.abs(df["threshold"] - BASE_T + 0.001) < 0.03)]
    if len(r_row) and len(b_row):
        ds = float(r_row["sensitivity"].values[0]) - float(b_row["sensitivity"].values[0])
        print(
            f"  {acc:<15} gc={rb['gc']:>5.1f}%  eff_t={rb['eff_t']:.3f}"
            f"  sens_gain={ds:>+7.2f}pp  f1_gain={rb['df']-r0['df']:>+7.2f}pp"
        )

# ── Formula summary ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"RECOMMENDED FORMULA: threshold = {BASE_T} - {best_alpha} * max(0, genome_gc - {GC_FLOOR})")
print(f"  Example thresholds:")
for gc_ex in [0.35, 0.45, 0.55, 0.60, 0.65, 0.70]:
    t = BASE_T - best_alpha * max(0, gc_ex - GC_FLOOR)
    print(f"    gc={gc_ex*100:.0f}%  threshold={t:.3f}")
print(SEP)
