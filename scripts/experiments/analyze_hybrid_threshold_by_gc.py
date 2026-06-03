# EXPERIMENT: Hybrid filter threshold calibration by genome GC
# STATUS: active
# RESULT: pending
"""
ML-NEW2: Diagnose how the Hybrid filter threshold interacts with genome GC.

The stage analysis showed the Hybrid filter loses 6-9pp sensitivity
disproportionately for problem genomes (M.leprae, Streptomyces, Bordetella)
vs only 1pp for clean genomes. The current global threshold is 0.471.

This script:
  1. Runs the full pipeline up to second filter on 20 holdout genomes
  2. For each genome: sweeps Hybrid threshold from 0.30 to 0.60
  3. Records F1/Sens/Prec at each threshold
  4. Reports the optimal threshold per genome and the GC-F1 correlation
  5. Identifies whether a single lower threshold or a GC-adaptive threshold
     would improve problem genomes without hurting clean ones

Run from repo root:
    python scripts/experiments/analyze_hybrid_threshold_by_gc.py
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.comparative_analysis import compare_orfs_to_reference
from src.config import (
    FIRST_FILTER_THRESHOLD,
    SECOND_FILTER_THRESHOLD,
    START_SELECTION_WEIGHTS,
    TEST_GENOMES,
)
from src.data_management import get_data_dir, load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier, StartSelectionClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    organize_nested_orfs,
    score_all_orfs,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

THRESHOLDS = np.round(np.arange(0.30, 0.62, 0.02), 2).tolist()

# Corrected baseline per-genome F1/Sens (entry #23)
BASELINE_F1 = {
    "NC_002947.4": 72.13,
    "NC_002929.2": 64.68,
    "NC_003143.1": 82.99,
    "NC_003116.1": 76.48,
    "NC_004757.1": 73.27,
    "NC_008497.1": 86.97,
    "NC_004350.2": 88.75,
    "NC_006270.3": 80.90,
    "NC_006274.1": 85.92,
    "NC_003030.1": 88.63,
    "NC_003155.5": 68.95,
    "NC_003450.3": 71.69,
    "NC_002677.1": 49.33,
    "NC_008268.1": 67.29,
    "NC_006958.1": 70.71,
    "NC_008818.1": 61.37,
    "NC_015948.1": 76.75,
    "NC_014408.1": 75.53,
    "NC_019977.1": 75.81,
    "NC_007644.1": 71.90,
}

# Load models once
lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))

hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))

ss = StartSelectionClassifier()
ss.load(str(MODELS_DIR / "start_selector.pkl"))

with open(MODELS_DIR / "thresholds.json") as f:
    thr = json.load(f)
LGB_T = thr["orf_classifier_lgb"]["threshold"]
HF_T = thr["hybrid_best_model"]["threshold"]  # current = 0.471

genomes = [a for a in TEST_GENOMES if (Path(DATA_DIR) / f"{a}.fasta").exists()]
SEP = "=" * 95

print(f"\n{SEP}")
print(f"HYBRID THRESHOLD SWEEP BY GENOME GC")
print(f"  Current threshold: {HF_T}  |  Sweep: {THRESHOLDS[0]:.2f} – {THRESHOLDS[-1]:.2f}")
print(SEP)

# results[acc][threshold] = {"f1", "sensitivity", "precision"}
results = {acc: {} for acc in genomes}
gc_values = {}

for idx, acc in enumerate(genomes, 1):
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    gc_values[acc] = gc
    print(f"  [{idx:>2}/{len(genomes)}] {acc}  gc={gc*100:.1f}%", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        interg = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, interg)
        scored = score_all_orfs(orfs, models)
        filt1 = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups0 = organize_nested_orfs(filt1)
        groups = lgb.filter_groups(
            groups=groups0,
            genome_id=acc,
            weights=START_SELECTION_WEIGHTS,
            threshold=LGB_T,
            genome_gc=gc,
        )
        top = ss.select_best_starts(groups, seq, models, START_SELECTION_WEIGHTS)
        filt2 = filter_candidates(top, **SECOND_FILTER_THRESHOLD)

    # Get hybrid scores for all candidates once (predict returns preds, probs, ids)
    filt2_list = filt2.to_dict("records") if hasattr(filt2, "to_dict") else list(filt2)
    _, hf_probs, _ = hf.predict(filt2_list, genome_id=acc, batch_size=32)
    hf_probs = np.array(hf_probs).ravel()

    # Sweep thresholds without re-running the pipeline
    for t in THRESHOLDS:
        final = [c for c, p in zip(filt2_list, hf_probs) if p >= t]
        metrics = compare_orfs_to_reference(final, acc)
        results[acc][t] = {
            "f1": metrics["f1_pct"],
            "sensitivity": metrics["sensitivity_pct"],
            "precision": metrics["precision_pct"],
        }

    # Quick summary: optimal threshold for this genome
    best_t = max(THRESHOLDS, key=lambda t: results[acc][t]["f1"])
    nearest_t = min(THRESHOLDS, key=lambda t: abs(t - HF_T))
    curr = results[acc][nearest_t]
    print(
        f"    current(t={HF_T}): F1={curr['f1']:.2f} Sens={curr['sensitivity']:.2f}  "
        f"best(t={best_t:.2f}): F1={results[acc][best_t]['f1']:.2f} "
        f"Sens={results[acc][best_t]['sensitivity']:.2f}  "
        f"baseline={BASELINE_F1.get(acc,0):.2f}",
        flush=True,
    )

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY: F1 at each threshold (vs baseline)")
print(f"  {'t':<6}", end="")
for acc in genomes:
    print(f"  {acc[3:10]:>8}", end="")
print(f"  {'MEAN':>8}  {'Regress':>8}")
print("  " + "-" * (6 + len(genomes) * 10 + 20))

best_global_t = None
best_global_f1 = -1.0
best_regressions = 999

for t in THRESHOLDS:
    f1s = [results[acc][t]["f1"] for acc in genomes]
    mean_f1 = np.mean(f1s)
    regressions = sum(
        1 for acc in genomes if results[acc][t]["f1"] - BASELINE_F1.get(acc, 0) < -0.05
    )
    marker = " ✓" if regressions == 0 else ""
    if regressions < best_regressions or (
        regressions == best_regressions and mean_f1 > best_global_f1
    ):
        best_regressions = regressions
        best_global_f1 = mean_f1
        best_global_t = t
    print(f"  {t:<6.2f}", end="")
    for acc, f1 in zip(genomes, f1s):
        df = f1 - BASELINE_F1.get(acc, 0)
        sym = "+" if df > 0.05 else ("-" if df < -0.05 else "=")
        print(f"  {f1:>6.2f}{sym}", end="")
    print(f"  {mean_f1:>8.2f}  {regressions:>8}{marker}")

print(
    f"\n  Best threshold: t={best_global_t:.2f}  "
    f"mean_F1={best_global_f1:.2f}  regressions={best_regressions}"
)

# GC correlation
print(f"\n{SEP}")
print("GC CORRELATION: optimal threshold per genome")
print(
    f"  {'Genome':<15} {'GC%':>5} {'Opt_t':>6} {'F1@opt':>8} {'Sens@opt':>9} {'F1@curr':>8} {'Gain':>7}"
)
print("  " + "-" * 65)
opt_thresholds = []
for acc in genomes:
    gc_pct = gc_values[acc] * 100
    best_t = max(THRESHOLDS, key=lambda t: results[acc][t]["f1"])
    f1_opt = results[acc][best_t]["f1"]
    sens_opt = results[acc][best_t]["sensitivity"]
    f1_curr = results[acc][min(THRESHOLDS, key=lambda t: abs(t - HF_T))]
    gain = f1_opt - f1_curr["f1"]
    opt_thresholds.append((gc_pct, best_t))
    print(
        f"  {acc:<15} {gc_pct:>5.1f} {best_t:>6.2f} {f1_opt:>8.2f} {sens_opt:>9.2f} "
        f"{f1_curr['f1']:>8.2f} {gain:>+7.2f}"
    )

# Correlation between GC and optimal threshold
gcs = [x[0] for x in opt_thresholds]
ts = [x[1] for x in opt_thresholds]
corr = np.corrcoef(gcs, ts)[0, 1]
print(f"\n  Correlation(GC, optimal_threshold): r = {corr:.3f}")
print(
    f"  Interpretation: {'negative = lower threshold needed for high-GC' if corr < -0.2 else 'no clear GC-threshold relationship'}"
)

# Save results
rows = []
for acc in genomes:
    for t in THRESHOLDS:
        r = results[acc][t]
        rows.append(
            {
                "acc": acc,
                "gc_pct": gc_values[acc] * 100,
                "threshold": t,
                **r,
                "baseline_f1": BASELINE_F1.get(acc, 0),
                "df1": r["f1"] - BASELINE_F1.get(acc, 0),
            }
        )
pd.DataFrame(rows).to_csv(OUT_DIR / "hybrid_threshold_sweep.csv", index=False)
print(f"\nSaved: {OUT_DIR}/hybrid_threshold_sweep.csv")
print(SEP)
