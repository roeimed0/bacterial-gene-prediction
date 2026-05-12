"""
Calibrate START_SELECTION_WEIGHTS against the current pipeline.

For every positive group (≥1 ORF matches reference GFF) with multiple
candidate starts, we want select_best_starts() to pick the true reference
start.  The weights are optimised via softmax cross-entropy (L-BFGS-B).

Robustness measures:
  - Multiple random starting points + current weights as starting point
  - Genome-level cross-validation (80% train / 20% val)
  - Bootstrap confidence intervals on accuracy delta
  - Full provenance logged to calibration_report.json

Run from repo root:
    python scripts/calibrate_start_weights.py [--genomes N] [--seed S]

Options:
    --genomes N   Limit to N genomes from GENOME_CATALOG (0 = all available)
    --seed S      Random seed for reproducible train/val split and bootstrap
"""

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    FIRST_FILTER_THRESHOLD,
    GENOME_CATALOG,
    START_SELECTION_WEIGHTS,
)
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import OrfGroupClassifier
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
REPORT_PATH = MODELS_DIR / "calibration_report.json"
SEP = "=" * 80

FEAT_ORDER = ["codon", "imm", "rbs", "length", "start"]
NORM_COLS = [f"{f}_score_norm" for f in FEAT_ORDER]

LGB_THRESHOLD = 0.07  # threshold used to filter groups
N_STARTS = 8  # number of random starting points for optimiser
BOOTSTRAP_N = 500  # bootstrap resamples for CI
TRAIN_FRAC = 0.8  # genome-level train fraction

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--genomes", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

# ── Load LGB model ────────────────────────────────────────────────────────────

lgb_path = MODELS_DIR / "orf_classifier_lgb.pkl"
if not lgb_path.exists():
    print(f"ERROR: {lgb_path} not found.")
    sys.exit(1)
clf = OrfGroupClassifier()
clf.load(str(lgb_path))

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_ref_set(accession: str) -> set:
    gff = get_gff_path(accession)
    if not os.path.exists(gff):
        return set()
    ref = pd.read_csv(gff, sep="\t", comment="#", header=None)
    cds = ref[ref[2] == "CDS"][[3, 4]]
    cds.columns = ["start", "end"]
    return set(zip(cds["start"], cds["end"]))


def collect_genome(accession: str):
    """Return list of (k×5 score matrix, true_start_idx) for multi-start groups."""
    genome = load_genome_sequence(os.path.join(DATA_DIR, f"{accession}.fasta"))
    if not genome:
        return None
    seq = genome["sequence"]
    ref_set = _load_ref_set(accession)
    if not ref_set:
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups = organize_nested_orfs(filtered)
        groups = clf.filter_groups(
            groups=groups,
            genome_id=accession,
            weights=START_SELECTION_WEIGHTS,
            threshold=LGB_THRESHOLD,
        )

    result = []
    for _key, group_df in groups.items():
        if len(group_df) < 2:
            continue  # single-start — no ambiguity
        if "genome_start" in group_df.columns:
            match = group_df.apply(
                lambda r: (int(r["genome_start"]), int(r["genome_end"])) in ref_set, axis=1
            )
        else:
            match = group_df.apply(lambda r: (int(r["start"]), int(r["end"])) in ref_set, axis=1)
        true_idxs = np.where(match.values)[0]
        if len(true_idxs) == 0:
            continue
        scores = group_df[NORM_COLS].values.astype(np.float64)
        result.append((scores, int(true_idxs[0])))
    return result


def _make_buckets(data: list) -> list:
    """
    Group (scores, true_idx) pairs by number of candidates k.
    Returns list of (S, T) where S is (n×k×5) and T is (n,) int array.
    Bucketing by k lets us vectorise the loss with a single matmul per bucket
    instead of a Python loop over every group — ~100× faster at 240K groups.
    """
    by_k: dict = {}
    for scores, true_idx in data:
        k = scores.shape[0]
        if k not in by_k:
            by_k[k] = ([], [])
        by_k[k][0].append(scores)
        by_k[k][1].append(true_idx)
    return [
        (np.array(s_list), np.array(t_list, dtype=np.int64)) for s_list, t_list in by_k.values()
    ]


def selection_accuracy(w: np.ndarray, buckets: list) -> float:
    if not buckets:
        return 0.0
    correct = total = 0
    for S, T in buckets:
        preds = (S @ w).argmax(axis=1)
        correct += int((preds == T).sum())
        total += len(T)
    return correct / total if total else 0.0


def softmax_ce_loss(w: np.ndarray, buckets: list) -> float:
    """Vectorised softmax cross-entropy: -mean log P(true start | scores, w)."""
    total = n_total = 0
    for S, T in buckets:
        logits = S @ w  # (n × k)
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        log_z = np.log(np.exp(logits).sum(axis=1))  # (n,)
        true_logits = logits[np.arange(len(T)), T]  # (n,)
        total += (log_z - true_logits).sum()
        n_total += len(T)
    return total / n_total if n_total else 0.0


def optimise(buckets: list, seed=None) -> np.ndarray:
    """Run L-BFGS-B from N_STARTS starting points; return best weights."""
    bounds = [(1e-4, None)] * 5
    best_loss = np.inf
    best_w = None
    local_rng = np.random.default_rng(seed)

    starts = [np.array([START_SELECTION_WEIGHTS[f] for f in FEAT_ORDER])]
    for _ in range(N_STARTS - 1):
        starts.append(local_rng.uniform(0.1, 10.0, 5))

    for x0 in starts:
        res = minimize(
            softmax_ce_loss,
            x0,
            args=(buckets,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = res.x.copy()

    current_max = max(START_SELECTION_WEIGHTS.values())
    best_w = best_w * (current_max / best_w.max())
    return best_w


def bootstrap_ci(old_w, new_w, buckets: list, n: int, seed) -> tuple:
    """95% CI on Δaccuracy(new − old) via bootstrap resampling of groups."""
    local_rng = np.random.default_rng(seed)
    # Flatten to per-group correct/incorrect for both models
    new_correct = np.concatenate(
        [((S @ new_w).argmax(axis=1) == T).astype(float) for S, T in buckets]
    )
    old_correct = np.concatenate(
        [((S @ old_w).argmax(axis=1) == T).astype(float) for S, T in buckets]
    )
    delta_per_group = new_correct - old_correct
    deltas = [
        delta_per_group[local_rng.integers(0, len(delta_per_group), len(delta_per_group))].mean()
        for _ in range(n)
    ]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ── Collect data ──────────────────────────────────────────────────────────────

available = [
    e
    for e in GENOME_CATALOG
    if os.path.exists(os.path.join(DATA_DIR, f"{e['accession']}.fasta"))
    and os.path.exists(get_gff_path(e["accession"]))
]
if args.genomes:
    idx = rng.choice(len(available), min(args.genomes, len(available)), replace=False)
    available = [available[i] for i in sorted(idx)]

# Genome-level train/val split
rng.shuffle(available)
n_train = int(len(available) * TRAIN_FRAC)
train_genomes = available[:n_train]
val_genomes = available[n_train:]

print(f"\n{SEP}")
print(
    f"COLLECTING DATA — {len(available)} genomes  ({len(train_genomes)} train / {len(val_genomes)} val)"
)
print(SEP)

train_data, val_data = [], []

for split_name, genome_list, bucket in [
    ("train", train_genomes, train_data),
    ("val", val_genomes, val_data),
]:
    for entry in genome_list:
        acc = entry["accession"]
        print(f"  [{split_name}] {acc}...", end=" ", flush=True)
        groups = collect_genome(acc)
        if groups is None:
            print("SKIP")
            continue
        print(f"{len(groups)} multi-start groups")
        bucket.extend(groups)

print(f"\n  Train: {len(train_data):,} multi-start groups")
print(f"  Val:   {len(val_data):,} multi-start groups")

if len(train_data) < 200:
    print("ERROR: too few training groups. Try --genomes with a larger number.")
    sys.exit(1)

# Pre-build vectorised buckets (groups same-k stacked into numpy arrays)
print("  Pre-computing buckets...", end=" ", flush=True)
train_buckets = _make_buckets(train_data)
val_buckets = _make_buckets(val_data)
del train_data, val_data  # free raw list memory
n_buckets = sum(len(T) for _, T in train_buckets)
print(f"done  ({len(train_buckets)} size buckets, {n_buckets:,} groups)")

# ── Baseline ──────────────────────────────────────────────────────────────────

current_w = np.array([START_SELECTION_WEIGHTS[f] for f in FEAT_ORDER])
train_baseline = selection_accuracy(current_w, train_buckets)
val_baseline = selection_accuracy(current_w, val_buckets)

print(f"\n  Baseline accuracy — train: {train_baseline:.4f}  val: {val_baseline:.4f}")

# ── Optimise ──────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print(f"OPTIMISING ({N_STARTS} starting points, L-BFGS-B, vectorised)")
print(SEP)

new_w = optimise(train_buckets, seed=int(rng.integers(0, 2**31)))

train_new = selection_accuracy(new_w, train_buckets)
val_new = selection_accuracy(new_w, val_buckets)

# ── Bootstrap CI on val delta ─────────────────────────────────────────────────

ci_lo, ci_hi = bootstrap_ci(
    current_w, new_w, val_buckets, BOOTSTRAP_N, seed=int(rng.integers(0, 2**31))
)

# ── Results ───────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("RESULTS")
print(SEP)

print(f"\n  {'Feature':<10} {'Current':>10} {'Optimised':>11}  {'Change':>8}")
print(f"  {'-------':<10} {'-------':>10} {'---------':>11}  {'------':>8}")
for feat, cw, nw in zip(FEAT_ORDER, current_w, new_w):
    pct = (nw - cw) / cw * 100
    sym = "+" if pct >= 0 else ""
    print(f"  {feat:<10} {cw:>10.4f} {nw:>11.4f}  {sym}{pct:.1f}%")

print(
    f"\n  Selection accuracy (train):  {train_baseline:.4f}  ->  {train_new:.4f}  ({train_new-train_baseline:+.4f})"
)
print(
    f"  Selection accuracy (val):    {val_baseline:.4f}  ->  {val_new:.4f}  ({val_new-val_baseline:+.4f})"
)
print(f"  Bootstrap 95% CI on val delta:  [{ci_lo:+.4f}, {ci_hi:+.4f}]")

if ci_lo > 0:
    verdict = "SIGNIFICANT IMPROVEMENT — safe to adopt"
elif ci_hi < 0:
    verdict = "REGRESSION — do not adopt"
else:
    verdict = "NO SIGNIFICANT IMPROVEMENT — CI crosses zero, keep current weights"

print(f"\n  Verdict: {verdict}")

# ── Coordinate sensitivity analysis ──────────────────────────────────────────
# Perturb each weight independently to confirm whether current values
# are truly at a local optimum and how sensitive accuracy is to each weight.

print(f"\n{SEP}")
print("COORDINATE SENSITIVITY (val accuracy — how much does each weight matter?)")
print(SEP)
print("\n  Factors tested: 0.25x  0.5x  0.75x  0.9x  [1.0x current]  1.1x  1.25x  1.5x  2.0x  4.0x")

PERTURB_FACTORS = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 4.0]
coord_best = dict(zip(FEAT_ORDER, current_w))

for i, feat in enumerate(FEAT_ORDER):
    print(f"\n  {feat} (current={current_w[i]:.4f}):")
    best_acc = -1.0
    best_factor = 1.0
    for f in PERTURB_FACTORS:
        w_test = current_w.copy()
        w_test[i] = current_w[i] * f
        acc = selection_accuracy(w_test, val_buckets)
        marker = " <-- current" if f == 1.0 else ""
        star = "*" if acc > val_baseline + 0.0005 else " "
        print(f"    {star} x{f:<5.2f}  w={w_test[i]:>7.4f}  acc={acc:.4f}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_factor = f
    coord_best[feat] = current_w[i] * best_factor
    if best_factor != 1.0:
        print(
            f"    --> best at x{best_factor} ({coord_best[feat]:.4f}), acc={best_acc:.4f} vs baseline {val_baseline:.4f}"
        )

# Report coordinate-search suggested weights
coord_w = np.array([coord_best[f] for f in FEAT_ORDER])
coord_acc = selection_accuracy(coord_w, val_buckets)
print(f"\n  Coordinate-search suggestion (val acc={coord_acc:.4f} vs baseline {val_baseline:.4f}):")
for feat, cw, sw in zip(FEAT_ORDER, current_w, coord_w):
    marker = f"  ({sw/cw:+.0%})" if abs(sw - cw) > 0.001 else "  (unchanged)"
    print(f"    {feat:<10} {cw:.4f}  ->  {sw:.4f}{marker}")

# ── Save report ───────────────────────────────────────────────────────────────

report = {
    "date": datetime.now().isoformat(),
    "lgb_model": str(lgb_path),
    "lgb_threshold": LGB_THRESHOLD,
    "genomes_total": len(available),
    "genomes_train": len(train_genomes),
    "genomes_val": len(val_genomes),
    "train_groups": sum(len(T) for _, T in train_buckets),
    "val_groups": sum(len(T) for _, T in val_buckets),
    "n_starts": N_STARTS,
    "bootstrap_n": BOOTSTRAP_N,
    "current_weights": {f: float(w) for f, w in zip(FEAT_ORDER, current_w)},
    "optimised_weights": {f: float(w) for f, w in zip(FEAT_ORDER, new_w)},
    "train_accuracy_baseline": float(train_baseline),
    "train_accuracy_new": float(train_new),
    "val_accuracy_baseline": float(val_baseline),
    "val_accuracy_new": float(val_new),
    "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
    "verdict": verdict,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Report saved -> {REPORT_PATH}")

print(f"\n{SEP}")
print("NEXT STEPS")
print(SEP)
if ci_lo > 0:
    print("\nUpdate src/config.py START_SELECTION_WEIGHTS:")
    print("\nSTART_SELECTION_WEIGHTS = {")
    for feat, nw in zip(FEAT_ORDER, new_w):
        print(f'    "{feat}": {nw:.4f},')
    print("}")
    print("\nThen retrain HybridGeneFilter on new pipeline output (issue #112).")
else:
    print("\nCurrent weights are already well-calibrated for this pipeline.")
    print("Proceed directly to HybridGeneFilter retraining (issue #112).")
print(SEP)
