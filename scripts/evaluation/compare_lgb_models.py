"""
Compare old vs new OrfGroupClassifier on 20 genomes neither model was trained on.

Genomes are selected to be outside GENOME_CATALOG (the new model's training pool).
Downloads any missing genomes from NCBI, then prints a threshold sweep showing
F1 / Precision / Recall for both models side by side.

Run from repo root:
    python scripts/compare_lgb_models.py [--download]
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_lgb import _load_ref_set, label_groups

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
SEP = "=" * 85

# ── 20 held-out genomes (not in GENOME_CATALOG) ───────────────────────────────
# 5 per taxonomy group for balanced coverage

HOLDOUT_GENOMES = [
    # Proteobacteria
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
    # Firmicutes
    {"accession": "NC_009009.1", "name": "Streptococcus sanguinis SK36", "group": "Firmicutes"},
    {"accession": "NC_007622.1", "name": "Staphylococcus aureus RF122", "group": "Firmicutes"},
    {"accession": "NC_012563.1", "name": "Bacillus anthracis str. A0248", "group": "Firmicutes"},
    {"accession": "NC_011375.1", "name": "Streptococcus pyogenes NZ131", "group": "Firmicutes"},
    {"accession": "NC_015975.1", "name": "Lactobacillus ruminis ATCC 27782", "group": "Firmicutes"},
    # Actinobacteria
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
    # Archaea
    {"accession": "NC_000961.1", "name": "Pyrococcus horikoshii OT3", "group": "Archaea"},
    {"accession": "NC_002689.2", "name": "Thermoplasma volcanium GSS1", "group": "Archaea"},
    {"accession": "NC_008818.1", "name": "Hyperthermus butylicus DSM 5456", "group": "Archaea"},
    {"accession": "NC_015562.1", "name": "Methanosaeta thermophila PT", "group": "Archaea"},
    {"accession": "NC_002578.1", "name": "Thermoplasma acidophilum DSM 1728", "group": "Archaea"},
]

# Sanity check: none of these should be in GENOME_CATALOG
catalog_accs = {g["accession"] for g in GENOME_CATALOG}
overlap = [g["accession"] for g in HOLDOUT_GENOMES if g["accession"] in catalog_accs]
if overlap:
    print(f"ERROR: these accessions are in GENOME_CATALOG: {overlap}")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true", help="Fetch missing genomes from NCBI")
args = parser.parse_args()

# ── Load both models ──────────────────────────────────────────────────────────

for p in (MODELS_DIR / "orf_classifier_lgb.pkl", MODELS_DIR / "orf_classifier_lgb_v2.pkl"):
    if not p.exists():
        print(f"ERROR: {p} not found. Run scripts/train_lgb.py first.")
        sys.exit(1)

old_clf = OrfGroupClassifier()
old_clf.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
new_clf = OrfGroupClassifier()
new_clf.load(str(MODELS_DIR / "orf_classifier_lgb_v2.pkl"))

old_feats = old_clf.model.feature_name_
new_feats = new_clf.feature_names or new_clf.model.feature_name_

# ── Download missing genomes ──────────────────────────────────────────────────

print(f"\n{SEP}")
print("HOLDOUT GENOMES — checking availability")
print(SEP)

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
    print(f"\nToo few genomes available ({len(available)}). Run with --download.")
    sys.exit(1)

print(f"\n{len(available)}/{len(HOLDOUT_GENOMES)} genomes available")

# ── Collect features ──────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("COLLECTING GROUP FEATURES")
print(SEP)

all_X, all_y = [], []

for entry in available:
    acc = entry["accession"]
    print(f"  {acc}  {entry['name'][:40]}...", end=" ", flush=True)

    genome = load_genome_sequence(os.path.join(DATA_DIR, f"{acc}.fasta"))
    if not genome:
        print("SKIP (load failed)")
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
        print("SKIP (no reference)")
        continue

    y = label_groups(groups, ref_set)
    feat_df = new_clf.extract_group_features(groups, acc, weights=START_SELECTION_WEIGHTS)
    feat_df = feat_df.drop(columns=["group_id"], errors="ignore")

    n_pos = int(y.sum())
    print(f"groups={len(y):,}  pos={n_pos}  neg={len(y) - n_pos}")
    all_X.append(feat_df)
    all_y.append(y)

if not all_X:
    print("No data collected.")
    sys.exit(1)

X_all = pd.concat(all_X, ignore_index=True)
y_all = np.concatenate(all_y)
print(f"\nTotal: {len(y_all):,} groups  pos={int(y_all.sum()):,}  neg={int((y_all == 0).sum()):,}")

# ── Score with both models ────────────────────────────────────────────────────

old_probs = np.asarray(old_clf.model.predict_proba(X_all[old_feats].values, num_threads=1))[:, 1]
new_probs = np.asarray(new_clf.model.predict_proba(X_all[new_feats].values, num_threads=1))[:, 1]

# Baseline recall of old model at t=0.10
old_recall_baseline = recall_score(y_all, old_probs >= 0.10, zero_division=0)

# ── Threshold sweep ───────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("THRESHOLD SWEEP — old model vs new model (held-out genomes)")
print(SEP)
print(
    f"\n  {'t':>5}  {'OLD F1':>7}  {'OLD Prec':>9}  {'OLD Rec':>8}  |  {'NEW F1':>7}  {'NEW Prec':>9}  {'NEW Rec':>8}  {'dF1':>7}"
)
print(
    f"  {'---':>5}  {'------':>7}  {'--------':>9}  {'-------':>8}  |  {'------':>7}  {'--------':>9}  {'-------':>8}  {'---':>7}"
)

for t in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.467, 0.50]:
    op = (old_probs >= t).astype(int)
    np_ = (new_probs >= t).astype(int)

    of1 = f1_score(y_all, op, zero_division=0)
    opr = precision_score(y_all, op, zero_division=0)
    ore = recall_score(y_all, op, zero_division=0)

    nf1 = f1_score(y_all, np_, zero_division=0)
    npr = precision_score(y_all, np_, zero_division=0)
    nre = recall_score(y_all, np_, zero_division=0)

    note = ""
    if abs(nre - old_recall_baseline) < 0.004:
        note = "  <- same recall as old@0.10"
    elif t == 0.10:
        note = "  <- apples-to-apples"

    print(
        f"  {t:>5.3f}  {of1:>7.4f}  {opr:>9.4f}  {ore:>8.4f}  |"
        f"  {nf1:>7.4f}  {npr:>9.4f}  {nre:>8.4f}  {nf1-of1:>+7.4f}{note}"
    )

print(f"\n  Old model recall at t=0.10: {old_recall_baseline:.4f}")
print(SEP)
