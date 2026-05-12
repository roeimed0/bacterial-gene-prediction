"""
Train OrfGroupClassifier (LightGBM) with scale_pos_weight to fix class imbalance.

Generates labelled training data from GENOME_CATALOG genomes by:
  1. Running the scoring pipeline up to organize_nested_orfs()
  2. Labelling each group as positive (≥1 ORF matches reference GFF) or negative
  3. Extracting group features via extract_group_features()

Train/val split: 20 genomes per taxonomy group → train (16) / val (4), fixed seed.

Saves to models/orf_classifier_lgb_v2.pkl — does NOT overwrite the existing model.
Compare old vs new on the val set before deciding to promote.

Run from repo root:
    python scripts/train_lgb.py [--limit N] [--no-val-compare]

Options:
    --limit N          Use only N genomes per group (default: all available)
    --no-val-compare   Skip side-by-side comparison against the current model
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, GENOME_CATALOG, START_SELECTION_WEIGHTS
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

VAL_PER_GROUP = 4  # used for early stopping + threshold calibration
TEST_PER_GROUP = 4  # held out — only used for final old-vs-new comparison

SEP = "=" * 90


# ── Label generation ──────────────────────────────────────────────────────────


def _load_ref_set(accession: str):
    """Return set of (start, end) CDS tuples from the reference GFF."""
    gff_path = get_gff_path(accession)
    if not os.path.exists(gff_path):
        return set()
    ref = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
    cds = ref[ref[2] == "CDS"][[3, 4]]
    cds.columns = ["start", "end"]
    return set(zip(cds["start"], cds["end"]))


def label_groups(groups: dict, ref_set: set) -> np.ndarray:
    """
    Return binary label array aligned to groups.items() order.
    A group is positive (1) if any ORF's (genome_start, genome_end) is in ref_set.
    """
    labels = []
    for _key, group_df in groups.items():
        if "genome_start" in group_df.columns and "genome_end" in group_df.columns:
            matched = any(
                (int(row.genome_start), int(row.genome_end)) in ref_set
                for row in group_df.itertuples(index=False)
            )
        else:
            matched = any(
                (int(row.start), int(row.end)) in ref_set
                for row in group_df.itertuples(index=False)
            )
        labels.append(int(matched))
    return np.array(labels, dtype=np.int32)


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run_pipeline(accession: str):
    """
    Run pipeline up to organize_nested_orfs().
    Returns (groups, classifier_instance_for_features) or None on failure.
    """
    fasta = os.path.join(DATA_DIR, f"{accession}.fasta")
    if not os.path.exists(fasta):
        return None

    genome = load_genome_sequence(fasta)
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

    return groups


# ── Train/val split ───────────────────────────────────────────────────────────


def build_splits(available_by_group: dict, val_per_group: int, test_per_group: int, seed: int):
    """
    Stratified three-way split: train / val / test.

    val  — used for early stopping and threshold calibration (seen during training)
    test — held out; only used for the final old-vs-new comparison
    """
    rng = np.random.default_rng(seed)
    train_accs, val_accs, test_accs = [], [], []
    for group, accs in sorted(available_by_group.items()):
        accs = list(accs)
        rng.shuffle(accs)
        n_held = min(val_per_group + test_per_group, len(accs) // 3)
        n_val = n_held // 2
        n_test = n_held - n_val
        val_accs.extend(accs[:n_val])
        test_accs.extend(accs[n_val : n_val + n_test])
        train_accs.extend(accs[n_val + n_test :])
    return train_accs, val_accs, test_accs


# ── Feature collection ────────────────────────────────────────────────────────


def collect_features(accessions: list, clf: OrfGroupClassifier, desc: str):
    """Run pipeline on each genome, extract group features + labels."""
    all_X, all_y = [], []
    for i, acc in enumerate(accessions, 1):
        print(f"  [{i:>3}/{len(accessions)}] {acc}...", end=" ", flush=True)
        groups = run_pipeline(acc)
        if groups is None:
            print("SKIP (missing data)")
            continue

        ref_set = _load_ref_set(acc)
        if not ref_set:
            print("SKIP (no reference)")
            continue

        y = label_groups(groups, ref_set)
        feat_df = clf.extract_group_features(groups, acc, weights=START_SELECTION_WEIGHTS)
        feat_df = feat_df.drop(columns=["group_id"], errors="ignore")

        n_pos = int(y.sum())
        print(f"groups={len(y):,}  pos={n_pos}  neg={len(y)-n_pos}")
        all_X.append(feat_df)
        all_y.append(y)

    if not all_X:
        return None, None
    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    return X, y


# ── Val-set evaluation ────────────────────────────────────────────────────────


def evaluate(clf: OrfGroupClassifier, X: pd.DataFrame, y: np.ndarray, label: str, threshold: float):
    from sklearn.metrics import f1_score, precision_score, recall_score

    probs = np.asarray(clf.model.predict_proba(X.values, num_threads=1))[:, 1]
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    print(f"  {label:<30}  F1={f1:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  (t={threshold:.3f})")
    return f1


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train OrfGroupClassifier (LightGBM) with scale_pos_weight."
    )
    parser.add_argument("--limit", type=int, default=0, help="Max genomes per group (0=all)")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: system entropy)"
    )
    parser.add_argument("--no-val-compare", action="store_true")
    args = parser.parse_args()
    seed = args.seed

    # Group available genomes by taxonomy
    available_by_group: dict = defaultdict(list)
    for entry in GENOME_CATALOG:
        acc = entry["accession"]
        fasta = os.path.join(DATA_DIR, f"{acc}.fasta")
        gff = get_gff_path(acc)
        if os.path.exists(fasta) and os.path.exists(gff):
            available_by_group[entry["group"]].append(acc)

    if args.limit:
        rng = np.random.default_rng(seed)
        available_by_group = {
            g: list(rng.choice(accs, min(args.limit, len(accs)), replace=False))
            for g, accs in available_by_group.items()
        }

    total = sum(len(v) for v in available_by_group.values())
    print(f"\n{SEP}")
    print(f"TRAINING DATA: {total} genomes available")
    for g, accs in sorted(available_by_group.items()):
        print(f"  {g:<20} {len(accs)} genomes")
    print(SEP)

    train_accs, val_accs, test_accs = build_splits(
        available_by_group, VAL_PER_GROUP, TEST_PER_GROUP, seed
    )
    print(f"\nSplit: {len(train_accs)} train / {len(val_accs)} val / {len(test_accs)} test")

    clf = OrfGroupClassifier()

    # ── Collect training features ─────────────────────────────────────────────
    print(f"\n{SEP}\nCOLLECTING TRAINING FEATURES ({len(train_accs)} genomes)\n{SEP}")
    X_train, y_train = collect_features(train_accs, clf, "train")
    if X_train is None:
        print("ERROR: no training data collected. Aborting.")
        sys.exit(1)
    print(
        f"\nTraining matrix: {X_train.shape}"
        f"  pos={int(y_train.sum())}  neg={int((y_train == 0).sum())}"
    )

    # ── Collect val features (early stopping + threshold calibration) ─────────
    print(f"\n{SEP}\nCOLLECTING VAL FEATURES ({len(val_accs)} genomes)\n{SEP}")
    X_val, y_val = collect_features(val_accs, clf, "val")
    if X_val is None:
        print("WARNING: no val data — training without early stopping")

    # ── Collect test features (held out — final comparison only) ─────────────
    print(f"\n{SEP}\nCOLLECTING TEST FEATURES ({len(test_accs)} genomes)\n{SEP}")
    X_test, y_test = collect_features(test_accs, clf, "test")
    if X_test is None:
        print("WARNING: no test data — final comparison will be skipped")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}\nTRAINING\n{SEP}")
    clf.train(X_train, y_train, X_val, y_val)

    # ── Calibrate threshold ───────────────────────────────────────────────────
    print(f"\n{SEP}\nTHRESHOLD CALIBRATION\n{SEP}")
    if X_val is not None and y_val is not None:
        best_threshold = clf.calibrate_threshold(X_val, y_val)
    else:
        best_threshold = 0.5
        print("  No val set — using default threshold 0.5")

    # ── Compare against existing model on held-out TEST set ──────────────────
    if not args.no_val_compare and X_test is not None and y_test is not None:
        old_model_path = MODELS_DIR / "orf_classifier_lgb.pkl"
        if old_model_path.exists():
            print(f"\n{SEP}\nTEST-SET COMPARISON (held-out — not seen during training)\n{SEP}")
            old_clf = OrfGroupClassifier()
            old_clf.load(str(old_model_path))

            old_feats = old_clf.model.feature_name_
            new_feats = clf.feature_names or list(X_test.columns)
            missing_old = [f for f in old_feats if f not in X_test.columns]
            if missing_old:
                print(f"  WARNING: old model expects features not in test set: {missing_old}")
            else:
                X_test_old = X_test[old_feats]
                evaluate(old_clf, X_test_old, y_test, "current model (t=0.10)", threshold=0.10)

            X_test_new = X_test[new_feats] if new_feats else X_test

            # Threshold sweep: find the point where new model matches old recall
            from sklearn.metrics import f1_score, precision_score, recall_score

            old_probs = np.asarray(
                old_clf.model.predict_proba(X_test[old_feats].values, num_threads=1)
            )[:, 1]
            old_recall_at_010 = recall_score(y_test, old_probs >= 0.10, zero_division=0)

            new_probs = np.asarray(clf.model.predict_proba(X_test_new.values, num_threads=1))[:, 1]
            print(
                f"\n  Threshold sweep on new model (old recall target = {old_recall_at_010:.4f}):"
            )
            print(f"  {'t':>6}  {'F1':>7}  {'Prec':>7}  {'Recall':>8}")
            print(f"  {'---':>6}  {'---':>7}  {'---':>7}  {'------':>8}")
            for t in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, best_threshold]:
                preds = (new_probs >= t).astype(int)
                f1 = f1_score(y_test, preds, zero_division=0)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                marker = " <-- matches old recall" if abs(rec - old_recall_at_010) < 0.005 else ""
                print(f"  {t:>6.3f}  {f1:>7.4f}  {prec:>7.4f}  {rec:>8.4f}{marker}")
        else:
            print("  (No existing model found — skipping comparison)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = MODELS_DIR / "orf_classifier_lgb_v2.pkl"
    print(f"\n{SEP}\nSAVING\n{SEP}")
    clf.save(str(out_path))
    print(f"\n  Calibrated threshold: {best_threshold:.3f}")
    print(f"  To promote: rename {out_path.name} -> orf_classifier_lgb.pkl")
    print("  Only promote after confirming F1 improvement on test set above.")
    print(SEP)


if __name__ == "__main__":
    main()
