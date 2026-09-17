# EXPERIMENT: HybridGeneFilter gc_content vs gc_deviation discrimination
# STATUS: active
# RESULT: pending
"""
Tests whether replacing gc_content (r=+0.95 with genome GC%) with
gc_deviation = gc_content - genome_gc gives better TP/FP discrimination.

gc_content: raw ORF GC fraction (0.35 in a 35% GC genome, 0.70 in a 70% GC genome)
gc_deviation: orf_gc - genome_gc (how much THIS ORF deviates from genome background)
              Real genes have different codon usage than background -> deviation != 0

The concern: with gc_content, the model learns "high GC genome -> harder to filter"
rather than "this specific ORF has anomalous GC for its genome."

Diagnostic method:
  - For each focus genome: classify candidates as TP/FP using reference GFF
  - Compute AUC-ROC for gc_content and gc_deviation separately
  - Compare: which feature better separates TP from FP?

If gc_deviation consistently outperforms gc_content -> replace in Hybrid feature set.

Run from repo root:
    python scripts/experiments/analyze_hybrid_gc_features.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import HybridGeneFilter
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
)

DATA_DIR = get_data_dir("full_dataset")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

STOP_TOL = 3

FOCUS_GENOMES = [
    ("NC_002929.2", "B. pertussis", "Proteobacteria", 67.7, 60.1),
    ("NC_002677.1", "M. leprae", "Actinobacteria", 57.8, 41.7),
    ("NC_008268.1", "Rhodococcus", "Actinobacteria", 67.5, 65.3),
    ("NC_003155.5", "S. avermitilis", "Actinobacteria", 70.7, 66.3),
    ("NC_015948.1", "Haloarcula", "Archaea", 63.7, 76.8),
    ("NC_003030.1", "C. acetobutyl.", "Firmicutes", 30.9, 88.0),
    ("NC_004350.2", "S. agalactiae", "Firmicutes", 36.8, 87.1),
]

hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = set(c["e"].astype(int).tolist())
    return exact, stops


def is_tp(cand, exact, stops):
    gs = int(cand.get("genome_start", cand.get("start", 0)))
    ge = int(cand.get("genome_end", cand.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in stops)


def auc_single_feature(feature_vals, labels):
    """Compute AUC-ROC for a single feature predicting label=1 (TP)."""
    pos = feature_vals[labels == 1]
    neg = feature_vals[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u_stat, _ = mannwhitneyu(pos, neg, alternative="greater")
    auc = u_stat / (len(pos) * len(neg))
    return max(auc, 1 - auc)  # flip so AUC > 0.5 always means "feature is informative"


SEP = "=" * 90
print(f"\n{SEP}")
print("DIAGNOSTIC: gc_content vs gc_deviation — HybridGeneFilter discrimination")
print(f"  AUC > 0.5 = feature discriminates TP from FP")
print(f"  AUC closer to 1.0 = better discrimination")
print(SEP)

all_rows = []

for acc, label, phylum, gc_pct, expected_f1 in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"  SKIP {acc}")
        continue
    exact, stops = load_ref(acc)

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    genome_gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    print(f"\n  {acc}  {label:<20}  gc={gc_pct:.1f}%  F1={expected_f1}%", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = filter_candidates(
            __import__("src.traditional_methods", fromlist=["score_all_orfs"]).score_all_orfs(
                orfs, models
            ),
            **FIRST_FILTER_THRESHOLD,
        )

    cands = scored.to_dict("records") if hasattr(scored, "to_dict") else list(scored)
    feat_df = hf.extract_features(cands, genome_id=acc, genome_gc=genome_gc)

    labels_arr = np.array([is_tp(c, exact, stops) for c in cands], dtype=int)

    if "gc_content" not in feat_df.columns or "gc_deviation" not in feat_df.columns:
        print(f"    Missing features: {feat_df.columns.tolist()[:5]}")
        continue

    gc_cont = feat_df["gc_content"].values
    gc_dev = feat_df["gc_deviation"].values
    gc3 = feat_df.get("gc3_content", pd.Series(np.zeros(len(feat_df)))).values

    # Derived: gc3 deviation from genome
    gc3_dev = gc3 - genome_gc

    auc_cont = auc_single_feature(gc_cont, labels_arr)
    auc_dev = auc_single_feature(gc_dev, labels_arr)
    auc_gc3 = auc_single_feature(gc3, labels_arr)
    auc_gc3d = auc_single_feature(gc3_dev, labels_arr)

    n_tp = labels_arr.sum()
    n_fp = len(labels_arr) - n_tp

    print(f"    n_cands={len(cands)}  TP={n_tp}  FP={n_fp}")
    print(f"    {'Feature':<25} {'AUC':>7}  {'TP mean':>9} {'FP mean':>9}  Better?")
    print(f"    {'-'*25} {'-'*7}  {'-'*9} {'-'*9}  {'-'*15}")
    for feat_name, vals, auc in [
        ("gc_content", gc_cont, auc_cont),
        ("gc_deviation", gc_dev, auc_dev),
        ("gc3_content", gc3, auc_gc3),
        ("gc3 - genome_gc", gc3_dev, auc_gc3d),
    ]:
        tp_mean = float(vals[labels_arr == 1].mean())
        fp_mean = float(vals[labels_arr == 0].mean())
        better = "BETTER" if auc > 0.54 else ("similar" if auc > 0.50 else "worse")
        print(f"    {feat_name:<25} {auc:>7.4f}  {tp_mean:>9.4f} {fp_mean:>9.4f}  {better}")
        all_rows.append(
            {
                "acc": acc,
                "label": label,
                "phylum": phylum,
                "gc_pct": gc_pct,
                "feature": feat_name,
                "auc": round(auc, 4),
                "tp_mean": round(tp_mean, 4),
                "fp_mean": round(fp_mean, 4),
            }
        )

df = pd.DataFrame(all_rows)

print(f"\n{SEP}")
print("CROSS-GENOME SUMMARY: Mean AUC per feature")
print(SEP)
for feat_name in ["gc_content", "gc_deviation", "gc3_content", "gc3 - genome_gc"]:
    g = df[df["feature"] == feat_name]
    if len(g) == 0:
        continue
    print(
        f"  {feat_name:<25}  mean_AUC={g['auc'].mean():.4f}  min={g['auc'].min():.4f}  "
        f"max={g['auc'].max():.4f}"
    )

print(f"\n  DECISION: if gc_deviation mean_AUC > gc_content mean_AUC")
print(f"    -> Replace gc_content with gc_deviation in HybridGeneFilter features")
print(f"    -> Similarly for gc3_content vs gc3-genome_gc")

out = OUT_DIR / "hybrid_gc_diagnostic.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
