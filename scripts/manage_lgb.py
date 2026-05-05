"""
Manage OrfGroupClassifier (LightGBM) lifecycle: train, compare, promote.

Subcommands:

  train    Retrain on GENOME_CATALOG genomes with scale_pos_weight.
           Saves to models/orf_classifier_lgb_v2.pkl — never overwrites production.

  compare  Evaluate old (production) vs new (v2) on 20 held-out genomes that are
           outside GENOME_CATALOG. Prints a threshold sweep table.

  promote  Copy v2 → production after you have verified the comparison.
           Backs up the old model to orf_classifier_lgb_v1_backup.pkl.

Run from repo root:
    python scripts/manage_lgb.py train   [--seed N] [--limit N]
    python scripts/manage_lgb.py compare [--download]
    python scripts/manage_lgb.py promote [--threshold T]
"""

import argparse
import contextlib
import io
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import FIRST_FILTER_THRESHOLD, GENOME_CATALOG, START_SELECTION_WEIGHTS
from src.data_management import (
    download_genome_and_reference,
    get_data_dir,
    get_gff_path,
    load_genome_sequence,
)
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

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = get_data_dir("full_dataset")
PROD_MODEL = MODELS_DIR / "orf_classifier_lgb.pkl"
NEW_MODEL = MODELS_DIR / "orf_classifier_lgb_v2.pkl"
BACKUP_MODEL = MODELS_DIR / "orf_classifier_lgb_v1_backup.pkl"

VAL_PER_GROUP = 4
TEST_PER_GROUP = 4
SEP = "=" * 85

# 20 held-out genomes — verified not in GENOME_CATALOG
HOLDOUT_GENOMES = [
    {"accession": "NC_004347.2", "name": "Shewanella oneidensis MR-1", "group": "Proteobacteria"},
    {"accession": "NC_009512.1", "name": "Pseudomonas putida F1", "group": "Proteobacteria"},
    {"accession": "NC_000915.1", "name": "Helicobacter pylori 26695", "group": "Proteobacteria"},
    {
        "accession": "NC_007493.2",
        "name": "Rhodobacter sphaeroides 2.4.1",
        "group": "Proteobacteria",
    },
    {
        "accession": "NC_002655.2",
        "name": "Escherichia coli O157:H7 EDL933",
        "group": "Proteobacteria",
    },
    {"accession": "NC_009009.1", "name": "Streptococcus sanguinis SK36", "group": "Firmicutes"},
    {"accession": "NC_007622.1", "name": "Staphylococcus aureus RF122", "group": "Firmicutes"},
    {"accession": "NC_012563.1", "name": "Bacillus anthracis str. A0248", "group": "Firmicutes"},
    {"accession": "NC_011375.1", "name": "Streptococcus pyogenes NZ131", "group": "Firmicutes"},
    {"accession": "NC_015975.1", "name": "Lactobacillus ruminis ATCC 27782", "group": "Firmicutes"},
    {
        "accession": "NC_003155.5",
        "name": "Streptomyces avermitilis MA-4680",
        "group": "Actinobacteria",
    },
    {
        "accession": "NC_008596.1",
        "name": "Mycobacterium smegmatis mc2155",
        "group": "Actinobacteria",
    },
    {
        "accession": "NC_003903.1",
        "name": "Streptomyces coelicolor A3(2)",
        "group": "Actinobacteria",
    },
    {
        "accession": "NC_006958.1",
        "name": "Corynebacterium glutamicum ATCC 13032",
        "group": "Actinobacteria",
    },
    {"accession": "NC_012490.1", "name": "Rhodococcus erythropolis PR4", "group": "Actinobacteria"},
    {"accession": "NC_000961.1", "name": "Pyrococcus horikoshii OT3", "group": "Archaea"},
    {"accession": "NC_002689.2", "name": "Thermoplasma volcanium GSS1", "group": "Archaea"},
    {"accession": "NC_008818.1", "name": "Hyperthermus butylicus DSM 5456", "group": "Archaea"},
    {"accession": "NC_015562.1", "name": "Methanosaeta thermophila PT", "group": "Archaea"},
    {"accession": "NC_002578.1", "name": "Thermoplasma acidophilum DSM 1728", "group": "Archaea"},
]

catalog_accs = {g["accession"] for g in GENOME_CATALOG}
_bad = [g["accession"] for g in HOLDOUT_GENOMES if g["accession"] in catalog_accs]
assert not _bad, f"Holdout genomes found in GENOME_CATALOG: {_bad}"


# ── Shared helpers ────────────────────────────────────────────────────────────


def _load_ref_set(accession: str) -> set:
    gff_path = get_gff_path(accession)
    if not os.path.exists(gff_path):
        return set()
    ref = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
    cds = ref[ref[2] == "CDS"][[3, 4]]
    cds.columns = ["start", "end"]
    return set(zip(cds["start"], cds["end"]))


def label_groups(groups: dict, ref_set: set) -> np.ndarray:
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


def run_pipeline(accession: str):
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


def collect_features(accessions: list, clf: OrfGroupClassifier):
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
        print(f"groups={len(y):,}  pos={n_pos}  neg={len(y) - n_pos}")
        all_X.append(feat_df)
        all_y.append(y)
    if not all_X:
        return None, None
    return pd.concat(all_X, ignore_index=True), np.concatenate(all_y)


def build_splits(available_by_group: dict, val_per_group: int, test_per_group: int, seed):
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


def print_sweep(old_clf, new_clf, X, y, old_feats, new_feats):
    old_probs = np.asarray(old_clf.model.predict_proba(X[old_feats].values, num_threads=1))[:, 1]
    new_probs = np.asarray(new_clf.model.predict_proba(X[new_feats].values, num_threads=1))[:, 1]
    old_recall_at_010 = recall_score(y, old_probs >= 0.10, zero_division=0)

    print(
        f"\n  {'t':>5}  {'OLD F1':>7}  {'OLD Prec':>9}  {'OLD Rec':>8}  |  {'NEW F1':>7}  {'NEW Prec':>9}  {'NEW Rec':>8}  {'dF1':>7}"
    )
    print(
        f"  {'---':>5}  {'------':>7}  {'--------':>9}  {'-------':>8}  |  {'------':>7}  {'--------':>9}  {'-------':>8}  {'---':>7}"
    )
    for t in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50]:
        op = (old_probs >= t).astype(int)
        np_ = (new_probs >= t).astype(int)
        of1 = f1_score(y, op, zero_division=0)
        opr = precision_score(y, op, zero_division=0)
        ore = recall_score(y, op, zero_division=0)
        nf1 = f1_score(y, np_, zero_division=0)
        npr = precision_score(y, np_, zero_division=0)
        nre = recall_score(y, np_, zero_division=0)
        note = (
            "  <- same recall as old@0.10"
            if abs(nre - old_recall_at_010) < 0.004
            else ("  <- apples-to-apples" if t == 0.10 else "")
        )
        print(
            f"  {t:>5.3f}  {of1:>7.4f}  {opr:>9.4f}  {ore:>8.4f}  |"
            f"  {nf1:>7.4f}  {npr:>9.4f}  {nre:>8.4f}  {nf1-of1:>+7.4f}{note}"
        )
    print(f"\n  Old model recall at t=0.10: {old_recall_at_010:.4f}")


# ── Subcommands ───────────────────────────────────────────────────────────────


def cmd_train(args):
    available_by_group: dict = defaultdict(list)
    for entry in GENOME_CATALOG:
        acc = entry["accession"]
        if os.path.exists(os.path.join(DATA_DIR, f"{acc}.fasta")) and os.path.exists(
            get_gff_path(acc)
        ):
            available_by_group[entry["group"]].append(acc)

    if args.limit:
        rng = np.random.default_rng(args.seed)
        available_by_group = {
            g: list(rng.choice(accs, min(args.limit, len(accs)), replace=False))
            for g, accs in available_by_group.items()
        }

    total = sum(len(v) for v in available_by_group.values())
    print(f"\n{SEP}\nTRAINING DATA: {total} genomes\n{SEP}")
    for g, accs in sorted(available_by_group.items()):
        print(f"  {g:<20} {len(accs)} genomes")

    train_accs, val_accs, test_accs = build_splits(
        available_by_group, VAL_PER_GROUP, TEST_PER_GROUP, args.seed
    )
    print(f"\nSplit: {len(train_accs)} train / {len(val_accs)} val / {len(test_accs)} test")

    clf = OrfGroupClassifier()

    print(f"\n{SEP}\nCOLLECTING TRAINING FEATURES ({len(train_accs)} genomes)\n{SEP}")
    X_train, y_train = collect_features(train_accs, clf)
    if X_train is None:
        print("ERROR: no training data.")
        sys.exit(1)
    print(
        f"\nTraining matrix: {X_train.shape}  pos={int(y_train.sum())}  neg={int((y_train==0).sum())}"
    )

    print(f"\n{SEP}\nCOLLECTING VAL FEATURES ({len(val_accs)} genomes)\n{SEP}")
    X_val, y_val = collect_features(val_accs, clf)

    print(f"\n{SEP}\nCOLLECTING TEST FEATURES ({len(test_accs)} genomes)\n{SEP}")
    X_test, y_test = collect_features(test_accs, clf)

    print(f"\n{SEP}\nTRAINING\n{SEP}")
    clf.train(X_train, y_train, X_val, y_val)

    print(f"\n{SEP}\nTHRESHOLD CALIBRATION\n{SEP}")
    if X_val is not None and y_val is not None:
        best_t = clf.calibrate_threshold(X_val, y_val)
    else:
        best_t = 0.5
        print("  No val set — using default 0.5")

    if X_test is not None and y_test is not None and PROD_MODEL.exists():
        print(f"\n{SEP}\nTEST-SET COMPARISON (held-out)\n{SEP}")
        old_clf = OrfGroupClassifier()
        old_clf.load(str(PROD_MODEL))
        old_feats = old_clf.model.feature_name_
        new_feats = clf.feature_names or list(X_test.columns)
        if all(f in X_test.columns for f in old_feats):
            print_sweep(old_clf, clf, X_test, y_test, old_feats, new_feats)

    print(f"\n{SEP}\nSAVING -> {NEW_MODEL}\n{SEP}")
    clf.save(str(NEW_MODEL))
    print(f"\n  Calibrated threshold: {best_t:.3f}")
    print(f"  Run 'compare' to validate on held-out genomes, then 'promote' to go live.")
    print(SEP)


def cmd_compare(args):
    for p in (PROD_MODEL, NEW_MODEL):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    old_clf = OrfGroupClassifier()
    old_clf.load(str(PROD_MODEL))
    new_clf = OrfGroupClassifier()
    new_clf.load(str(NEW_MODEL))
    old_feats = old_clf.model.feature_name_
    new_feats = new_clf.feature_names or new_clf.model.feature_name_

    print(f"\n{SEP}\nHOLDOUT GENOMES\n{SEP}")
    available = []
    for entry in HOLDOUT_GENOMES:
        acc = entry["accession"]
        fasta = os.path.join(DATA_DIR, f"{acc}.fasta")
        gff = get_gff_path(acc)
        if os.path.exists(fasta) and os.path.exists(gff):
            available.append(entry)
            print(f"  {acc}  OK")
        elif args.download:
            print(f"  {acc}  downloading...", end=" ", flush=True)
            try:
                fp, gp = download_genome_and_reference(acc)
                if fp and gp:
                    available.append(entry)
                    print("done")
                else:
                    print("FAILED")
            except Exception as e:
                print(f"ERROR: {e}")
        else:
            print(f"  {acc}  MISSING (run with --download)")

    if len(available) < 5:
        print("Too few genomes. Run with --download.")
        sys.exit(1)

    print(f"\n{SEP}\nCOLLECTING FEATURES ({len(available)} genomes)\n{SEP}")
    all_X, all_y = [], []
    for entry in available:
        acc = entry["accession"]
        print(f"  {acc}  {entry['name'][:40]}...", end=" ", flush=True)
        genome = load_genome_sequence(os.path.join(DATA_DIR, f"{acc}.fasta"))
        if not genome:
            print("SKIP")
            continue
        seq = genome["sequence"]
        with contextlib.redirect_stdout(io.StringIO()):
            orfs = find_orfs_candidates(seq, min_length=100)
            training = create_training_set(sequence=seq, all_orfs=orfs)
            intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
            models = build_all_scoring_models(training, intergenic)
            scored = score_all_orfs(orfs, models)
            filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
            groups = organize_nested_orfs(filtered)
        ref_set = _load_ref_set(acc)
        if not ref_set:
            print("SKIP (no ref)")
            continue
        y = label_groups(groups, ref_set)
        feat_df = new_clf.extract_group_features(groups, acc, weights=START_SELECTION_WEIGHTS)
        feat_df = feat_df.drop(columns=["group_id"], errors="ignore")
        print(f"groups={len(y):,}  pos={int(y.sum())}")
        all_X.append(feat_df)
        all_y.append(y)

    if not all_X:
        print("No data collected.")
        sys.exit(1)

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    print(f"\nTotal: {len(y):,} groups  pos={int(y.sum()):,}  neg={int((y==0).sum()):,}")

    print(f"\n{SEP}\nTHRESHOLD SWEEP — old (production) vs new (v2) on held-out genomes\n{SEP}")
    print_sweep(old_clf, new_clf, X, y, old_feats, new_feats)
    print(SEP)
    print("\nIf new model is better: run 'promote --threshold T' to go live.")


def cmd_promote(args):
    if not NEW_MODEL.exists():
        print(f"ERROR: {NEW_MODEL} not found. Run 'train' first.")
        sys.exit(1)

    print(f"\nBacking up current production model -> {BACKUP_MODEL.name}")
    if PROD_MODEL.exists():
        shutil.copy2(str(PROD_MODEL), str(BACKUP_MODEL))

    print(f"Promoting {NEW_MODEL.name} -> {PROD_MODEL.name}")
    shutil.copy2(str(NEW_MODEL), str(PROD_MODEL))
    print(f"Done. New production model is {PROD_MODEL.name}.")

    if args.threshold is not None:
        print(f"\nReminder: update default threshold to {args.threshold} in:")
        print("  src/ml_models.py  api/models.py  hybrid_predictor.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Manage OrfGroupClassifier LightGBM model lifecycle.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
sub = parser.add_subparsers(dest="cmd", required=True)

p_train = sub.add_parser("train", help="Retrain model on GENOME_CATALOG genomes")
p_train.add_argument("--seed", type=int, default=None, help="Random seed (default: system entropy)")
p_train.add_argument("--limit", type=int, default=0, help="Max genomes per group (0=all)")

p_compare = sub.add_parser("compare", help="Compare old vs new on 20 held-out genomes")
p_compare.add_argument("--download", action="store_true", help="Fetch missing holdout genomes")

p_promote = sub.add_parser("promote", help="Swap v2 into production (backs up old)")
p_promote.add_argument(
    "--threshold", type=float, default=None, help="Recommended threshold for the new model"
)

args = parser.parse_args()
{"train": cmd_train, "compare": cmd_compare, "promote": cmd_promote}[args.cmd](args)
