# EXPERIMENT: Sweep flip_t threshold for start classifier v4b candidate
# STATUS: active
# RESULT: pending
"""
Finds the optimal flip_t for start_selector_v4b_candidate.pkl on the 20-genome holdout.

Runs the full pipeline ONCE per genome, then evaluates all flip_t values in the same
pass — no redundant genome processing. Reports F1/Sens/Prec per threshold and marks
which thresholds have zero regressions vs the corrected baseline (entry #23, F1=74.50).

Run from repo root:
    python scripts/experiments/sweep_flip_threshold.py
"""

import contextlib
import io
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

FLIP_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
CANDIDATE_MODEL = str(MODELS_DIR / "start_selector_v4b_candidate.pkl")

# Corrected baseline per-genome F1 (entry #23, F1=74.50%)
BASELINE = {
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
ss.load(CANDIDATE_MODEL)
print(f"Candidate model: {Path(CANDIDATE_MODEL).name}")
print(f"  n_features={len(ss.features)}  contest_t={ss.contest_t}  calibrated={ss.calibrated}")
print(f"  Sweeping flip_t: {FLIP_THRESHOLDS}")

import json

with open(MODELS_DIR / "thresholds.json") as f:
    thr = json.load(f)
LGB_T = thr.get("orf_classifier_lgb", {}).get("threshold", 0.07)
HF_T = thr.get("hybrid_best_model", {}).get("threshold", 0.471)

genomes = [a for a in TEST_GENOMES if (Path(DATA_DIR) / f"{a}.fasta").exists()]
SEP = "=" * 90

# results[acc][flip_t] = {"f1": ..., "sensitivity": ..., "precision": ...}
results = {acc: {} for acc in genomes}

print(f"\n{SEP}")
print(f"Running pipeline on {len(genomes)} holdout genomes...")
print(SEP)

for idx, acc in enumerate(genomes, 1):
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    print(f"  [{idx:>2}/{len(genomes)}] {acc}  gc={gc*100:.1f}%", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        interg = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, interg)
        scored = score_all_orfs(orfs, models)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups0 = organize_nested_orfs(filtered)
        groups = lgb.filter_groups(
            groups=groups0,
            genome_id=acc,
            weights=START_SELECTION_WEIGHTS,
            threshold=LGB_T,
            genome_gc=gc,
        )

    # Build per-genome classifier context once
    rbs_pwm = ss._build_rbs_pwm(groups, seq)
    ctx_pwm = ss._build_ctx_pwm(groups, seq)
    len_mean, len_std = ss._build_len_prior(groups)

    # Sweep all flip_t values in one pass per genome
    for flip_t in FLIP_THRESHOLDS:
        ss.flip_t = flip_t
        top = ss.select_best_starts(groups, seq, models, START_SELECTION_WEIGHTS)

        with contextlib.redirect_stdout(io.StringIO()):
            filt2 = filter_candidates(top, **SECOND_FILTER_THRESHOLD)
            final = hf.filter_candidates(
                candidates=filt2, genome_id=acc, threshold=HF_T, batch_size=32
            )

        final_list = final.to_dict("records") if hasattr(final, "to_dict") else list(final)
        metrics = compare_orfs_to_reference(final_list, acc)
        results[acc][flip_t] = {
            "f1": metrics["f1_pct"],
            "sensitivity": metrics["sensitivity_pct"],
            "precision": metrics["precision_pct"],
        }

    # Print quick summary for this genome
    row = "  ".join(f"t={t:.2f}:{results[acc][t]['f1']:.1f}" for t in FLIP_THRESHOLDS)
    print(f"    {row}", flush=True)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("THRESHOLD SWEEP RESULTS")
print(
    f"  {'flip_t':<8}  {'F1':>7}  {'Sens':>7}  {'Prec':>7}  {'Regress':>8}  {'Improve':>8}  {'dF1':>7}"
)
print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*7}")

best_t = None
best_f1 = -1.0

for flip_t in FLIP_THRESHOLDS:
    f1s = [results[acc][flip_t]["f1"] for acc in genomes]
    senss = [results[acc][flip_t]["sensitivity"] for acc in genomes]
    precs = [results[acc][flip_t]["precision"] for acc in genomes]

    mean_f1 = np.mean(f1s)
    mean_sens = np.mean(senss)
    mean_prec = np.mean(precs)

    regressions = sum(
        1 for acc in genomes if results[acc][flip_t]["f1"] - BASELINE.get(acc, 0) < -0.05
    )
    improvements = sum(
        1 for acc in genomes if results[acc][flip_t]["f1"] - BASELINE.get(acc, 0) > 0.05
    )
    df1 = mean_f1 - np.mean(list(BASELINE.values()))

    marker = " ✓ NO REGRESSION" if regressions == 0 else ""
    print(
        f"  {flip_t:<8.2f}  {mean_f1:>7.2f}  {mean_sens:>7.2f}  {mean_prec:>7.2f}  "
        f"{regressions:>8}  {improvements:>8}  {df1:>+7.2f}{marker}"
    )

    if regressions == 0 and mean_f1 > best_f1:
        best_f1 = mean_f1
        best_t = flip_t

print(
    f"\n{'Best flip_t with 0 regressions: ' + str(best_t) if best_t else 'No threshold achieved 0 regressions'}"
)

# Per-genome detail for best threshold
if best_t:
    print(f"\n{SEP}")
    print(f"PER-GENOME DETAIL at flip_t={best_t}")
    print(f"  {'Genome':<15}  {'Baseline':>9}  {'v4b':>7}  {'dF1':>7}")
    print("  " + "-" * 45)
    for acc in genomes:
        b = BASELINE.get(acc, 0)
        v = results[acc][best_t]["f1"]
        df = v - b
        marker = " +++" if df > 0.05 else (" <<<" if df < -0.05 else "")
        print(f"  {acc:<15}  {b:>9.2f}  {v:>7.2f}  {df:>+7.2f}{marker}")

print(SEP)
