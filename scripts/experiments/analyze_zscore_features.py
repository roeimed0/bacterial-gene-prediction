# EXPERIMENT: Cross-group z-score features and SD prevalence modulator
# STATUS: active
# RESULT: pending
"""
Tests two new toggleable-under-condition ideas:

A) CROSS-GROUP Z-SCORES
   z_combined_max = (combined_max - genome_mean) / genome_std
   This normalises each group's score against all other groups in the same genome.
   Hypothesis: even when IMM/codon models are contaminated (absolute scores low),
   the relative rank across groups still discriminates real genes from FPs.
   Expected benefit: problem genomes (M. leprae, Bordetella) where ALL scores
   shift downward due to training contamination.

B) SD PREVALENCE AS LGB FEATURE
   sd_prevalence = fraction of training ORFs with detectable Shine-Dalgarno motif
   Already computed de novo in predict_rbs_simple(); currently unused.
   Hypothesis: when SD prevalence is low, RBS features carry noise ->
   the model should learn to down-weight rbs_max / rbs_dominance.

C) WITHIN-GENOME VARIANCE OF KEY FEATURES
   std(combined_max across all groups) as a genome-level context feature.
   Low variance = all groups look similar = harder to classify = model should
   rely more on relative features.

For each candidate: compare AUC(raw) vs AUC(candidate) across 20 holdout genomes.
Criterion: mean AUC improvement > +0.01 -> worth adding as new LGB feature.

Run from repo root:
    python scripts/experiments/analyze_zscore_features.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, START_SELECTION_WEIGHTS, TEST_GENOMES
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import OrfGroupClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    organize_nested_orfs,
    predict_rbs_simple,
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = set(c["e"].astype(int).tolist())
    return exact, stops


def is_tp_group(gid, gdf, exact, stops):
    rows = gdf.to_dict("records") if isinstance(gdf, pd.DataFrame) else list(gdf)
    for r in rows:
        gs = int(r.get("genome_start", r.get("start", 0)))
        ge = int(r.get("genome_end", r.get("end", 0)))
        if gs > ge:
            gs, ge = ge, gs
        if (gs, ge) in exact:
            return True
        if any(abs(ge - s) <= STOP_TOL for s in stops):
            return True
    return False


def auc(vals, labels):
    pos = vals[labels == 1]
    neg = vals[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.5
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    a = u / (len(pos) * len(neg))
    return max(a, 1 - a)


def compute_sd_prevalence(seq, training_orfs):
    """
    Fraction of training ORFs with detectable SD motif upstream.
    Reuses predict_rbs_simple logic: checks for AGGAGG-like motifs in [-20,-4] window.
    """
    SD_PATTERNS = ["AGGAGG", "AGGAG", "AGGA", "GAGG", "AAGG", "AGGG"]
    hits = 0
    for orf in training_orfs[:200]:  # sample first 200 for speed
        start = orf.get("genome_start", orf.get("start", 0))
        strand = orf.get("strand", "forward")
        if strand == "forward":
            window_start = max(0, start - 20)
            window = seq[window_start : start - 4]
        else:
            end = orf.get("genome_end", orf.get("end", 0))
            window_start = end + 4
            window_end = min(len(seq), end + 20)
            window = seq[window_start:window_end]
            window = window.translate(str.maketrans("ATGCatgc", "TACGtacg"))[::-1]
        if any(pat in window.upper() for pat in SD_PATTERNS):
            hits += 1
    n = min(len(training_orfs), 200)
    return hits / n if n > 0 else 0.5


SEP = "=" * 90
print(f"\n{SEP}")
print("DIAGNOSTIC: Cross-group z-scores + SD prevalence + within-genome variance")
print(f"  Z-score: (feature - genome_mean) / genome_std  per genome")
print(f"  SD prev: fraction of training ORFs with detectable Shine-Dalgarno")
print(f"  Variance: std(feature) across all groups in genome as context feature")
print(SEP)

# Features to z-score
ZSCORE_TARGETS = [
    "combined_max",
    "rbs_max",
    "imm_max",
    "start_select_max",
    "codon_max",
    "combined_mean",
    "rbs_dominance",
]

rows = []

for acc in TEST_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    try:
        exact, stops = load_ref(acc)
    except Exception:
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
    group_ids = list(groups.keys())
    labels = np.array(
        [int(is_tp_group(gid, groups[gid], exact, stops)) for gid in group_ids],
        dtype=int,
    )
    if len(labels) != len(feat_df):
        continue

    # Compute SD prevalence for this genome
    sd_prev = compute_sd_prevalence(
        seq,
        (
            training
            if isinstance(training, list)
            else training.to_dict("records") if hasattr(training, "to_dict") else list(training)
        ),
    )

    n_groups = len(feat_df)
    n_tp = int(labels.sum())
    print(f"    n_groups={n_groups}  TP={n_tp}  sd_prevalence={sd_prev:.3f}")

    genome_row = {
        "acc": acc,
        "gc_pct": round(gc * 100, 1),
        "n_groups": n_groups,
        "n_tp": n_tp,
        "sd_prevalence": round(sd_prev, 3),
    }

    # A) Raw AUC for each target feature
    for feat in ZSCORE_TARGETS:
        if feat not in feat_df.columns:
            continue
        raw_vals = feat_df[feat].fillna(0).values
        genome_row[f"auc_raw_{feat}"] = round(auc(raw_vals, labels), 4)

    # B) Z-score AUC
    for feat in ZSCORE_TARGETS:
        if feat not in feat_df.columns:
            continue
        raw_vals = feat_df[feat].fillna(0).values
        mu = raw_vals.mean()
        sigma = raw_vals.std()
        if sigma < 1e-9:
            z_vals = np.zeros(len(raw_vals))
        else:
            z_vals = (raw_vals - mu) / sigma
        genome_row[f"auc_z_{feat}"] = round(auc(z_vals, labels), 4)

    # C) Within-genome variance of key features as a scalar (genome-level context)
    # (AUC not meaningful for scalar genome features, but we capture them for inspection)
    for feat in ["combined_max", "rbs_max", "imm_max"]:
        if feat in feat_df.columns:
            genome_row[f"std_{feat}"] = round(float(feat_df[feat].std()), 4)

    # D) SD prevalence as a potential scalar feature value
    genome_row["sd_prev_val"] = sd_prev

    rows.append(genome_row)

df = pd.DataFrame(rows)

print(f"\n{SEP}")
print("A) Z-SCORE vs RAW AUC -- cross-genome mean")
print(SEP)
print(f"  {'Feature':<30} {'Raw AUC':>8} {'Z AUC':>8} {'Delta':>8}  Verdict")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}  -------")
zscore_results = []
for feat in ZSCORE_TARGETS:
    raw_col = f"auc_raw_{feat}"
    z_col = f"auc_z_{feat}"
    if raw_col not in df.columns or z_col not in df.columns:
        continue
    raw_mean = df[raw_col].mean()
    z_mean = df[z_col].mean()
    delta = z_mean - raw_mean
    verdict = "ADD z-score" if delta > 0.01 else ("marginal" if delta > 0 else "no benefit")
    print(f"  {feat:<30} {raw_mean:>8.4f} {z_mean:>8.4f} {delta:>+8.4f}  {verdict}")
    zscore_results.append(
        {
            "feature": feat,
            "raw_auc": round(raw_mean, 4),
            "z_auc": round(z_mean, 4),
            "delta": round(delta, 4),
            "verdict": verdict,
        }
    )

print(f"\n{SEP}")
print("B) SD PREVALENCE -- does it correlate with RBS feature AUC?")
print(SEP)
if "auc_raw_rbs_max" in df.columns and "sd_prevalence" in df.columns:
    corr = df["sd_prevalence"].corr(df["auc_raw_rbs_max"])
    print(f"  corr(sd_prevalence, rbs_max AUC) = {corr:.4f}")
    corr2 = (
        df["sd_prevalence"].corr(df["auc_raw_rbs_dominance"])
        if "auc_raw_rbs_dominance" in df.columns
        else None
    )
    if corr2 is not None:
        print(f"  corr(sd_prevalence, rbs_dominance AUC) = {corr2:.4f}")

    # Show per-genome sd_prevalence and rbs AUC
    print(f"\n  {'Acc':<15} {'GC%':>5} {'sd_prev':>8} {'rbs_max AUC':>12} {'rbs_dom AUC':>12}")
    print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*12} {'-'*12}")
    for _, r in df.sort_values("gc_pct").iterrows():
        rbs_auc = r.get("auc_raw_rbs_max", float("nan"))
        rbs_dom = r.get("auc_raw_rbs_dominance", float("nan"))
        print(
            f"  {r['acc']:<15} {r['gc_pct']:>5.1f} {r['sd_prevalence']:>8.3f} "
            f"{rbs_auc:>12.4f} {rbs_dom:>12.4f}"
        )

print(f"\n{SEP}")
print("C) WITHIN-GENOME VARIANCE -- does low variance predict low AUC?")
print(SEP)
for feat in ["combined_max", "rbs_max", "imm_max"]:
    std_col = f"std_{feat}"
    auc_col = f"auc_raw_{feat}"
    if std_col in df.columns and auc_col in df.columns:
        corr = df[std_col].corr(df[auc_col])
        print(f"  corr(std({feat}), AUC({feat})) = {corr:.4f}")
print("  (Positive correlation = low variance -> low AUC -> variance-gating could help)")

print(f"\n{SEP}")
print("D) GENOME-LEVEL CONTEXT: could std(combined_max) BE a useful LGB input?")
print(SEP)
print("  std(combined_max) acts as a difficulty signal: low std = all groups similar = harder.")
if "std_combined_max" in df.columns and "auc_raw_combined_max" in df.columns:
    df_sorted = df.sort_values("std_combined_max")
    low_std = df_sorted.head(7)
    high_std = df_sorted.tail(7)
    # Compute overall F1 proxy (just from AUC, since we don't have F1 per genome here)
    print(
        f"\n  Low std(combined_max) genomes (n=7):  mean AUC(combined_max)="
        f"{low_std['auc_raw_combined_max'].mean():.4f}"
    )
    print(
        f"  High std(combined_max) genomes (n=7): mean AUC(combined_max)="
        f"{high_std['auc_raw_combined_max'].mean():.4f}"
    )
    for _, r in df_sorted.iterrows():
        print(
            f"    {r['acc']:<15} std={r['std_combined_max']:.4f}  "
            f"AUC={r['auc_raw_combined_max']:.4f}  gc={r['gc_pct']:.1f}%"
        )

out = OUT_DIR / "zscore_features_diagnostic.csv"
df.to_csv(out, index=False)
pd.DataFrame(zscore_results).to_csv(OUT_DIR / "zscore_summary.csv", index=False)
print(f"\nSaved: {out}")
print(f"Saved: {OUT_DIR}/zscore_summary.csv")
print(SEP)
