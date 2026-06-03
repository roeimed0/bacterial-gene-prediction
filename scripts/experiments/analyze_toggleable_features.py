# EXPERIMENT: Find toggleable feature candidates across all 3 ML models
# STATUS: active
# RESULT: pending
"""
For each ML model, computes the correlation of every feature with genome GC%.
Features with |r| > 0.4 have systematic GC-dependent variation that the model
may be learning incorrectly — they are candidates for gating:

  gated_feature = max(0, feature - floor)  OR
  gated_feature = feature * (genome_gc >= threshold)

Models analyzed:
  1. OrfGroupClassifier (LGB) — 34 group-level features
  2. HybridGeneFilter — 26 dense-branch ORF features
  3. StartSelectionClassifier — 40+ pairwise features

Genome set: all 20 TEST_GENOMES spanning GC 30-70% and all 4 phyla.

Run from repo root:
    python scripts/experiments/analyze_toggleable_features.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, START_SELECTION_WEIGHTS, TEST_GENOMES
from src.data_management import get_data_dir, load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    normalize_all_orf_scores,
    organize_nested_orfs,
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

HOLDOUT_META = {
    "NC_002947.4": "Proteobacteria",
    "NC_002929.2": "Proteobacteria",
    "NC_003143.1": "Proteobacteria",
    "NC_003116.1": "Proteobacteria",
    "NC_004757.1": "Proteobacteria",
    "NC_008497.1": "Firmicutes",
    "NC_004350.2": "Firmicutes",
    "NC_006270.3": "Firmicutes",
    "NC_006274.1": "Firmicutes",
    "NC_003030.1": "Firmicutes",
    "NC_003155.5": "Actinobacteria",
    "NC_003450.3": "Actinobacteria",
    "NC_002677.1": "Actinobacteria",
    "NC_008268.1": "Actinobacteria",
    "NC_006958.1": "Actinobacteria",
    "NC_008818.1": "Archaea",
    "NC_015948.1": "Archaea",
    "NC_014408.1": "Archaea",
    "NC_019977.1": "Archaea",
    "NC_007644.1": "Archaea",
}

CORR_THRESHOLD = 0.40  # |r(feature, genome_gc)| above this = GC-correlation candidate
VAR_COLLAPSE_THRESHOLD = 0.10  # if within-genome std / cross-genome mean_std < this
# for any genome, feature is variance-collapsed there

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))

hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))

SEP = "=" * 90


def analyze_model(name, feat_rows_by_gc):
    """
    Given list of (genome_gc, feature_df) pairs, compute per-feature analysis:
      1. GC correlation: |r(feature, genome_gc)| — value shifts with GC%
      2. Variance collapse: fraction of genomes where within-genome std is
         < VAR_COLLAPSE_THRESHOLD * cross-genome mean std — feature is pure
         noise for those genomes regardless of its value

    Both conditions make a feature a toggleable candidate.
    """
    if not feat_rows_by_gc:
        return []
    dfs = [df for _, df in feat_rows_by_gc]
    combined = pd.concat(dfs, ignore_index=True)
    gc_per_row = []
    for gc, df in feat_rows_by_gc:
        gc_per_row.extend([gc] * len(df))
    gc_arr = np.array(gc_per_row)

    # Per-genome std for variance collapse detection
    genome_stds: dict[float, dict] = {}
    for gc, df in feat_rows_by_gc:
        genome_stds[gc] = {}
        for col in df.columns:
            genome_stds[gc][col] = float(df[col].fillna(0).std())

    results = []
    meta_cols = {"group_id", "phylum", "accession"}
    for col in combined.columns:
        if col in meta_cols:
            continue
        vals = combined[col].fillna(0).values
        if vals.std() < 1e-9:
            continue

        # GC correlation
        r, p = pearsonr(gc_arr, vals)

        # Variance collapse: how many genomes have near-zero within-genome std?
        per_genome_stds = [genome_stds[gc].get(col, 0) for gc, _ in feat_rows_by_gc]
        mean_std = np.mean(per_genome_stds)
        if mean_std > 0:
            collapsed_fraction = sum(
                1 for s in per_genome_stds if s < VAR_COLLAPSE_THRESHOLD * mean_std
            ) / max(len(per_genome_stds), 1)
        else:
            collapsed_fraction = 0.0

        results.append((col, r, p, collapsed_fraction, mean_std))

    return sorted(results, key=lambda x: -max(abs(x[1]), x[3]))


# ── MODEL 1: LGB (OrfGroupClassifier) ─────────────────────────────────────────

print(f"\n{SEP}")
print("MODEL 1: OrfGroupClassifier (LGB) — feature correlation with genome GC%")
print(f"  Collecting group features for {len(TEST_GENOMES)} genomes...")
print(SEP)

lgb_rows = []
for acc in TEST_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    print(f"  {acc}  gc={gc*100:.1f}%", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups = organize_nested_orfs(filtered)
        feat_df = lgb.extract_group_features(
            groups, acc, weights=START_SELECTION_WEIGHTS, genome_gc=gc
        )

    feat_df = feat_df.drop(columns=["group_id"], errors="ignore")
    lgb_rows.append((gc, feat_df))

lgb_results = analyze_model("LGB", lgb_rows)

print(
    f"\n  Features (|r|>0.15 or collapsed>10%) | r=GC_correlation | col%=variance_collapse_fraction"
)
print(f"  {'Feature':<35} {'r_gc':>7} {'col%':>6}  Reason")
print(f"  {'-'*35} {'-'*7} {'-'*6}  {'-'*35}")
lgb_toggleable = []
for feat, r, p, col_frac, mean_std in lgb_results:
    gc_flag = abs(r) > CORR_THRESHOLD and p < 0.05
    var_flag = col_frac > 0.20  # >20% of genomes have collapsed variance
    if abs(r) > 0.15 or col_frac > 0.10:
        reasons = []
        if gc_flag:
            reasons.append(f"GC-corr r={r:+.2f}")
        if var_flag:
            reasons.append(f"var-collapse {col_frac*100:.0f}% genomes")
        status = " | ".join(reasons) if reasons else "monitor"
        print(f"  {feat:<35} {r:>+7.4f} {col_frac*100:>5.1f}%  {status}")
        if gc_flag or var_flag:
            lgb_toggleable.append((feat, r, p, col_frac, "GC-corr" if gc_flag else "var-collapse"))
print(
    f"\n  Summary: {len(lgb_toggleable)} toggleable candidates out of {len(lgb_results)} features"
)


# ── MODEL 2: HybridGeneFilter ──────────────────────────────────────────────────

print(f"\n{SEP}")
print("MODEL 2: HybridGeneFilter — dense branch feature correlation with genome GC%")
print(f"  Collecting ORF features for {len(TEST_GENOMES)} genomes...")
print(SEP)

hf_rows = []
for acc in TEST_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    print(f"  {acc}  gc={gc*100:.1f}%", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        # Get features for all filtered candidates (convert DataFrame to list)
        filtered_list = (
            filtered.to_dict("records") if hasattr(filtered, "to_dict") else list(filtered)
        )
        feat_df = hf.extract_features(filtered_list, genome_id=acc)

    hf_rows.append((gc, feat_df))

hf_results = analyze_model("HybridGeneFilter", hf_rows)

print(
    f"\n  Features (|r|>0.15 or collapsed>10%) | r=GC_correlation | col%=variance_collapse_fraction"
)
print(f"  {'Feature':<35} {'r_gc':>7} {'col%':>6}  Reason")
print(f"  {'-'*35} {'-'*7} {'-'*6}  {'-'*35}")
hf_toggleable = []
for feat, r, p, col_frac, mean_std in hf_results:
    gc_flag = abs(r) > CORR_THRESHOLD and p < 0.05
    var_flag = col_frac > 0.20
    if abs(r) > 0.15 or col_frac > 0.10:
        reasons = []
        if gc_flag:
            reasons.append(f"GC-corr r={r:+.2f}")
        if var_flag:
            reasons.append(f"var-collapse {col_frac*100:.0f}% genomes")
        status = " | ".join(reasons) if reasons else "monitor"
        print(f"  {feat:<35} {r:>+7.4f} {col_frac*100:>5.1f}%  {status}")
        if gc_flag or var_flag:
            hf_toggleable.append((feat, r, p, col_frac, "GC-corr" if gc_flag else "var-collapse"))
print(f"\n  Summary: {len(hf_toggleable)} toggleable candidates out of {len(hf_results)} features")


# ── MODEL 3: StartSelectionClassifier ─────────────────────────────────────────

print(f"\n{SEP}")
print("MODEL 3: StartSelectionClassifier — pairwise feature correlation with genome GC%")
print(f"  Using cached pairwise_features_v2.csv + per-genome GC%")
print(SEP)

pairwise_csv = OUT_DIR / "pairwise_features_v2.csv"
if pairwise_csv.exists():
    pf = pd.read_csv(pairwise_csv)
    # Compute genome GC% for each accession
    gc_map = {}
    for acc in pf["acc"].unique():
        fasta = f"{DATA_DIR}/{acc}.fasta"
        if Path(fasta).exists():
            g = load_genome_sequence(fasta)
            s = g["sequence"]
            gc_map[acc] = (s.count("G") + s.count("C")) / max(len(s), 1)

    pf["genome_gc"] = pf["acc"].map(gc_map)
    pf = pf.dropna(subset=["genome_gc"])

    meta = {"acc", "phylum", "label", "genome_gc"}
    feat_cols = [c for c in pf.columns if c not in meta]
    gc_arr = pf["genome_gc"].values

    ssc_results = []
    for col in feat_cols:
        vals = pf[col].fillna(0).values
        if vals.std() < 1e-9:
            continue
        r, p = pearsonr(gc_arr, vals)
        ssc_results.append((col, r, p, 0.0, 0.0))
    ssc_results.sort(key=lambda x: -abs(x[1]))

    # For SSC we use genome-level aggregation for variance collapse
    # (pairwise CSV already has per-row data, compute per-accession std)
    print(f"\n  Features (|r|>0.15) | r=GC_correlation")
    print(f"  {'Feature':<35} {'r_gc':>7} {'p':>8}  Reason")
    print(f"  {'-'*35} {'-'*7} {'-'*8}  {'-'*35}")
    ssc_toggleable = []
    for feat, r, p, col_frac, _ in ssc_results:
        gc_flag = abs(r) > CORR_THRESHOLD and p < 0.05
        if abs(r) > 0.15:
            reason = f"GC-corr r={r:+.2f}" if gc_flag else "monitor"
            print(f"  {feat:<35} {r:>+7.4f} {p:>8.4f}  {reason}")
            if gc_flag:
                ssc_toggleable.append((feat, r, p, 0.0, "GC-corr"))
    print(
        f"\n  Summary: {len(ssc_toggleable)} toggleable candidates out of {len(ssc_results)} features"
    )
else:
    print("  pairwise_features_v2.csv not found -- skipping StartSelectionClassifier")
    ssc_toggleable = []

# ── Final summary ──────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("FINAL SUMMARY: All toggleable candidates across 3 models")
print(f"  Criterion: |r(feature, genome_gc)| > {CORR_THRESHOLD} AND p < 0.05")
print(SEP)

all_candidates = (
    [("LGB", f, r, p, col, rsn) for f, r, p, col, rsn in lgb_toggleable]
    + [("HybridFilter", f, r, p, col, rsn) for f, r, p, col, rsn in hf_toggleable]
    + [("StartClassifier", f, r, p, col, rsn) for f, r, p, col, rsn in ssc_toggleable]
)
all_candidates.sort(key=lambda x: -max(abs(x[2]), x[4]))

if all_candidates:
    print(f"\n  {'Model':<16} {'Feature':<35} {'r_gc':>7} {'col%':>6}  Reason + Gate type")
    print(f"  {'-'*16} {'-'*35} {'-'*7} {'-'*6}  {'-'*40}")
    for model, feat, r, p, col_frac, reason in all_candidates:
        if reason == "GC-corr":
            if r < 0:
                gate = "max(0, floor - feature)  [suppressed at high GC]"
            else:
                gate = "max(0, feature - floor)   [active at high GC]"
        else:
            gate = "feature * (within_genome_std > threshold)"
        print(f"  {model:<16} {feat:<35} {r:>+7.4f} {col_frac*100:>5.1f}%  {reason}: {gate}")
else:
    print("\n  No strong toggleable candidates found.")

# Save results
rows = []
for model, feat, r, p, col_frac, reason in all_candidates:
    rows.append(
        {
            "model": model,
            "feature": feat,
            "r_with_gc": round(r, 4),
            "p_value": round(p, 6),
            "collapse_frac": round(col_frac, 3),
            "reason": reason,
        }
    )
pd.DataFrame(rows).to_csv(OUT_DIR / "toggleable_feature_candidates.csv", index=False)
print(f"\nSaved: {OUT_DIR}/toggleable_feature_candidates.csv")
print(SEP)
