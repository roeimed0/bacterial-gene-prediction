"""
Train HybridGeneFilter (CNN+Dense) on output of the current LGB model.

Data pipeline:
  For each genome: run full pipeline through LGB filter -> select_best_starts
  -> second filter_candidates.  Label each resulting candidate 1/0 via GFF.

Split: 68 train / 16 val / 16 test  (stratified by taxonomy, genome-level).
  val  -- early stopping + threshold calibration
  test -- held out, only used for final old-vs-new comparison

Saves to models/hybrid_best_model_v2.pkl -- never overwrites production model.

Run from repo root:
    python scripts/train_hybrid.py [--epochs N] [--focal-loss] [--seed S] [--limit N]

Options:
    --epochs N     Max training epochs (default: 50)
    --focal-loss   Use focal loss instead of BCEWithLogitsLoss(pos_weight)
    --seed S       Random seed for train/val/test split
    --limit N      Use at most N genomes per taxonomy group (0=all)
"""

import argparse
import contextlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    FIRST_FILTER_THRESHOLD,
    GENOME_CATALOG,
    SECOND_FILTER_THRESHOLD,
    START_SELECTION_WEIGHTS,
)
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    organize_nested_orfs,
    score_all_orfs,
    select_best_starts,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = get_data_dir("full_dataset")
PROD_MODEL = MODELS_DIR / "hybrid_best_model.pkl"
NEW_MODEL = MODELS_DIR / "hybrid_best_model_v2.pkl"
BACKUP_MODEL = MODELS_DIR / "hybrid_best_model_v1_backup.pkl"

VAL_PER_GROUP = 4
TEST_PER_GROUP = 4
SEP = "=" * 85

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--focal-loss", action="store_true")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

# ── Load LGB model ────────────────────────────────────────────────────────────

lgb_clf = OrfGroupClassifier()
lgb_clf.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_ref_set(accession: str) -> set:
    gff = get_gff_path(accession)
    if not os.path.exists(gff):
        return set()
    ref = pd.read_csv(gff, sep="\t", comment="#", header=None)
    cds = ref[ref[2] == "CDS"][[3, 4]]
    cds.columns = ["start", "end"]
    return set(zip(cds["start"], cds["end"]))


def run_pipeline(accession: str):
    """
    Run pipeline through LGB -> select_best_starts -> second filter.
    Returns list of candidate dicts (the exact input HybridGeneFilter sees).
    """
    genome = load_genome_sequence(os.path.join(DATA_DIR, f"{accession}.fasta"))
    if not genome:
        return None
    seq = genome["sequence"]

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups = organize_nested_orfs(filtered)
        groups = lgb_clf.filter_groups(
            groups=groups,
            genome_id=accession,
            weights=START_SELECTION_WEIGHTS,
            threshold=0.07,
        )
        top = select_best_starts(groups, START_SELECTION_WEIGHTS)
        candidates = filter_candidates(top, **SECOND_FILTER_THRESHOLD)

    if hasattr(candidates, "to_dict"):
        candidates = candidates.to_dict("records")
    return candidates


def label_candidates(candidates: list, ref_set: set) -> np.ndarray:
    labels = []
    for c in candidates:
        gs = int(c.get("genome_start", c.get("start", 0)))
        ge = int(c.get("genome_end", c.get("end", 0)))
        labels.append(1 if (gs, ge) in ref_set else 0)
    return np.array(labels, dtype=np.int32)


def build_splits(available_by_group: dict, val_per_group: int, test_per_group: int, seed):
    local_rng = np.random.default_rng(seed)
    train_accs, val_accs, test_accs = [], [], []
    for group, accs in sorted(available_by_group.items()):
        accs = list(accs)
        local_rng.shuffle(accs)
        n_held = min(val_per_group + test_per_group, len(accs) // 3)
        n_val = n_held // 2
        n_test = n_held - n_val
        val_accs.extend(accs[:n_val])
        test_accs.extend(accs[n_val : n_val + n_test])
        train_accs.extend(accs[n_val + n_test :])
    return train_accs, val_accs, test_accs


def collect_candidates(accessions: list, split: str):
    all_cands, all_labels = [], []
    for i, acc in enumerate(accessions, 1):
        print(f"  [{split}] [{i:>3}/{len(accessions)}] {acc}...", end=" ", flush=True)
        if not (
            os.path.exists(os.path.join(DATA_DIR, f"{acc}.fasta"))
            and os.path.exists(get_gff_path(acc))
        ):
            print("SKIP (missing)")
            continue
        candidates = run_pipeline(acc)
        if candidates is None:
            print("SKIP (pipeline failed)")
            continue
        ref_set = _load_ref_set(acc)
        labels = label_candidates(candidates, ref_set)
        n_pos = int(labels.sum())
        print(f"n={len(labels):,}  pos={n_pos}  neg={len(labels)-n_pos}")
        all_cands.extend(candidates)
        all_labels.append(labels)
    if not all_cands:
        return None, None
    return all_cands, np.concatenate(all_labels)


def evaluate(
    hf: HybridGeneFilter, candidates: list, labels: np.ndarray, label: str, threshold: float
):
    _, probs, _ = hf.predict(candidates, batch_size=64)
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    print(f"  {label:<35}  F1={f1:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  (t={threshold:.3f})")
    return f1


# ── Gather available genomes ──────────────────────────────────────────────────

available_by_group = defaultdict(list)
for entry in GENOME_CATALOG:
    acc = entry["accession"]
    if os.path.exists(os.path.join(DATA_DIR, f"{acc}.fasta")) and os.path.exists(get_gff_path(acc)):
        available_by_group[entry["group"]].append(acc)

if args.limit:
    available_by_group = {
        g: list(rng.choice(accs, min(args.limit, len(accs)), replace=False))
        for g, accs in available_by_group.items()
    }

total = sum(len(v) for v in available_by_group.values())
print(f"\n{SEP}\nTRAINING DATA: {total} genomes\n{SEP}")
for g, accs in sorted(available_by_group.items()):
    print(f"  {g:<22} {len(accs)} genomes")

train_accs, val_accs, test_accs = build_splits(
    available_by_group, VAL_PER_GROUP, TEST_PER_GROUP, args.seed
)
print(f"\nSplit: {len(train_accs)} train / {len(val_accs)} val / {len(test_accs)} test")

# ── Collect candidates ────────────────────────────────────────────────────────

print(f"\n{SEP}\nCOLLECTING TRAIN CANDIDATES ({len(train_accs)} genomes)\n{SEP}")
train_cands, train_labels = collect_candidates(train_accs, "train")
if train_cands is None:
    print("ERROR: no training data.")
    sys.exit(1)
n_pos = int(train_labels.sum())
n_neg = int((train_labels == 0).sum())
print(
    f"\nTrain: {len(train_labels):,} candidates  pos={n_pos:,}  neg={n_neg:,}  ratio={n_neg/max(n_pos,1):.2f}"
)

print(f"\n{SEP}\nCOLLECTING VAL CANDIDATES ({len(val_accs)} genomes)\n{SEP}")
val_cands, val_labels = collect_candidates(val_accs, "val")
if val_cands is None:
    print("WARNING: no val data — training without early stopping.")

print(f"\n{SEP}\nCOLLECTING TEST CANDIDATES ({len(test_accs)} genomes)\n{SEP}")
test_cands, test_labels = collect_candidates(test_accs, "test")
if test_cands is None:
    print("WARNING: no test data — comparison will be skipped.")

# ── Train ─────────────────────────────────────────────────────────────────────

print(
    f"\n{SEP}\nTRAINING  (epochs={args.epochs}, {'focal loss' if args.focal_loss else 'BCEWithLogitsLoss+pos_weight'})\n{SEP}"
)

hf = HybridGeneFilter()
hf.train(
    candidates=train_cands,
    labels=train_labels,
    val_candidates=val_cands,
    val_labels=val_labels,
    epochs=args.epochs,
    batch_size=64,
    focal_loss=args.focal_loss,
)

# ── Calibrate threshold ───────────────────────────────────────────────────────

print(f"\n{SEP}\nTHRESHOLD CALIBRATION\n{SEP}")
if val_cands is not None and val_labels is not None:
    best_t = hf.calibrate_threshold(val_cands, val_labels)
    hf.threshold = best_t
else:
    best_t = 0.5
    print("  No val set — using default 0.5")

# ── Compare on test set ───────────────────────────────────────────────────────

if test_cands is not None and test_labels is not None and PROD_MODEL.exists():
    print(f"\n{SEP}\nTEST-SET COMPARISON (held out — not seen during training)\n{SEP}")

    old_hf = HybridGeneFilter()
    with contextlib.redirect_stdout(io.StringIO()):
        old_hf.load(str(PROD_MODEL))

    evaluate(old_hf, test_cands, test_labels, "current model (t=0.12)", threshold=0.12)
    evaluate(hf, test_cands, test_labels, "new model (calibrated threshold)", threshold=best_t)
    evaluate(hf, test_cands, test_labels, "new model (t=0.12 for fair compare)", threshold=0.12)

    # Threshold sweep
    import numpy as _np
    from sklearn.metrics import recall_score as _rs

    old_hf_thresh = 0.12
    _, old_probs, _ = old_hf.predict(test_cands, batch_size=64)
    _, new_probs, _ = hf.predict(test_cands, batch_size=64)
    old_recall = _rs(test_labels, (old_probs >= old_hf_thresh).astype(int), zero_division=0)

    print(f"\n  Threshold sweep (old recall target at t=0.12: {old_recall:.4f}):")
    print(
        f"  {'t':>5}  {'OLD F1':>7}  {'OLD Rec':>8}  |  {'NEW F1':>7}  {'NEW Rec':>8}  {'dF1':>7}"
    )
    for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, best_t]:
        op = (old_probs >= t).astype(int)
        np_ = (new_probs >= t).astype(int)
        of1 = f1_score(test_labels, op, zero_division=0)
        ore = recall_score(test_labels, op, zero_division=0)
        nf1 = f1_score(test_labels, np_, zero_division=0)
        nre = recall_score(test_labels, np_, zero_division=0)
        note = "  <- same recall" if abs(nre - old_recall) < 0.004 else ""
        print(
            f"  {t:>5.3f}  {of1:>7.4f}  {ore:>8.4f}  |  {nf1:>7.4f}  {nre:>8.4f}  {nf1-of1:>+7.4f}{note}"
        )

# ── Save ──────────────────────────────────────────────────────────────────────

print(f"\n{SEP}\nSAVING -> {NEW_MODEL}\n{SEP}")
hf.save(str(NEW_MODEL))
print(f"\n  Calibrated threshold: {best_t:.3f}")
print(f"  To promote: python scripts/train_hybrid.py promote")
print(f"  Only promote after confirming end-to-end F1 improvement.")
print(SEP)
