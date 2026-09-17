# EXPERIMENT: SD prevalence as LGB feature + RBS interaction terms
# STATUS: active
# RESULT: pending
"""
sd_prevalence = fraction of training ORFs with a detectable Shine-Dalgarno motif
  (computed de novo from upstream windows of the first 200 training ORFs)

Previous diagnostic confirmed:
  corr(sd_prevalence, rbs_max AUC) = 0.8957  -- very strong
  corr(sd_prevalence, rbs_dominance AUC) = 0.8388

This means: when SD prevalence is LOW, RBS features are noisy and uncorrelated
with TP/FP label. The LGB currently uses rbs_max the same way regardless.

IMPORTANT NOTE ON WITHIN-GENOME AUC:
sd_prevalence is a genome-level scalar -- every group in a genome gets the
same value. So within one genome: AUC(rbs_max * sd_prevalence) == AUC(rbs_max)
because multiplying by a positive constant preserves rank order.

The benefit must come cross-genome: the model sees sd_prevalence alongside
rbs_max and learns to discount RBS features when SD prevalence is low.

TESTS (all cross-genome pooled across all 20 holdout genomes):
  1. AUC of sd_prevalence alone (genome-level signal)
  2. AUC of rbs_max * sd_prevalence vs raw rbs_max
  3. AUC of rbs_dominance * sd_prevalence vs raw rbs_dominance
  4. Correlation of sd_prevalence with existing LGB features (must be <0.95)
  5. Cross-genome AUC with and without sd_prevalence as an additive feature

Decision criterion:
  - sd_prevalence must have <0.95 correlation with all existing features
  - rbs_max * sd_prevalence must beat rbs_max alone by >0.01 AUC (cross-genome)
  - OR: sd_prevalence itself has cross-genome AUC > 0.60 (genome-level discrimination)

Run from repo root:
    python scripts/experiments/analyze_sd_prevalence_feature.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr

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
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3

SD_PATTERNS = ["AGGAGG", "AGGAG", "AGGA", "GAGG", "AAGG", "AGGG"]


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


def compute_sd_prevalence(seq, training_orfs, n_sample=200):
    """Fraction of training ORFs with a detectable SD motif in [-20, -4] window."""
    orfs = list(training_orfs) if not isinstance(training_orfs, list) else training_orfs
    sample = orfs[:n_sample]
    hits = 0
    for orf in sample:
        start = orf.get("genome_start", orf.get("start", 0))
        strand = orf.get("strand", "forward")
        if strand == "forward":
            window = seq[max(0, start - 20) : max(0, start - 4)]
        else:
            end = orf.get("genome_end", orf.get("end", 0))
            raw = seq[end + 4 : min(len(seq), end + 20)]
            window = raw.translate(str.maketrans("ATGCatgc", "TACGtacg"))[::-1]
        if any(pat in window.upper() for pat in SD_PATTERNS):
            hits += 1
    n = min(len(orfs), n_sample)
    return hits / n if n > 0 else 0.5


def auc(vals, labels):
    pos = vals[labels == 1]
    neg = vals[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.5
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    a = u / (len(pos) * len(neg))
    return max(a, 1 - a)


SEP = "=" * 90
print(f"\n{SEP}")
print("DIAGNOSTIC: SD prevalence as LGB feature + RBS interaction terms")
print(f"  Cross-genome pooled analysis (sd_prevalence is genome-level scalar)")
print(SEP)

# Collect all data
all_feats, all_labels, all_sd_prev = [], [], []

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

    # Compute SD prevalence
    training_list = training.to_dict("records") if hasattr(training, "to_dict") else list(training)
    sd_prev = compute_sd_prevalence(seq, training_list)

    print(f"    n={len(feat_df)}  TP={labels.sum()}  sd_prevalence={sd_prev:.3f}")

    all_feats.append(feat_df)
    all_labels.append(labels)
    all_sd_prev.extend([sd_prev] * len(feat_df))

# Pool everything
df_all = pd.concat(all_feats, ignore_index=True)
y_all = np.concatenate(all_labels)
sd_arr = np.array(all_sd_prev)

print(f"\n  Total pooled: {len(df_all)} groups  TP={y_all.sum()}")

# ── 1. Correlation of sd_prevalence with existing features ────────────────────

print(f"\n{SEP}")
print("1. CORRELATION: sd_prevalence vs existing LGB features")
print(SEP)
print(f"  {'Feature':<30} {'Pearson r':>10}  Flag")
high_corr = []
for col in df_all.columns:
    r, _ = pearsonr(sd_arr, df_all[col].fillna(0).values)
    flag = "  <<HIGH CORR" if abs(r) > 0.95 else ("  <warn" if abs(r) > 0.70 else "")
    print(f"  {col:<30} {r:>10.4f}{flag}")
    if abs(r) > 0.95:
        high_corr.append(col)
if high_corr:
    print(f"\n  ALERT: sd_prevalence correlates >0.95 with: {high_corr}")
    print("  -> Cannot add as independent feature (collinear)")
else:
    print(f"\n  OK: max correlation < 0.95 -- sd_prevalence is an independent signal")

# ── 2. Cross-genome AUC of sd_prevalence and interaction terms ───────────────

print(f"\n{SEP}")
print("2. CROSS-GENOME AUC: sd_prevalence + interaction terms")
print(SEP)

results = []
for name, vals in [
    ("sd_prevalence (genome scalar)", sd_arr),
    ("rbs_max (raw)", df_all["rbs_max"].fillna(0).values),
    ("rbs_dominance (raw)", df_all["rbs_dominance"].fillna(0).values),
    ("rbs_max * sd_prevalence", df_all["rbs_max"].fillna(0).values * sd_arr),
    ("rbs_dominance * sd_prevalence", df_all["rbs_dominance"].fillna(0).values * sd_arr),
    ("combined_max (raw)", df_all["combined_max"].fillna(0).values),
    ("combined_max * sd_prevalence", df_all["combined_max"].fillna(0).values * sd_arr),
]:
    a = auc(vals, y_all)
    results.append({"feature": name, "cross_genome_auc": round(a, 4)})
    print(f"  {name:<40} AUC={a:.4f}")

# ── 3. Delta analysis: interaction vs raw ─────────────────────────────────────

print(f"\n{SEP}")
print("3. DELTA: interaction term vs raw feature (cross-genome pooled)")
print(SEP)
rdf = pd.DataFrame(results).set_index("feature")
pairs = [
    ("rbs_max * sd_prevalence", "rbs_max (raw)"),
    ("rbs_dominance * sd_prevalence", "rbs_dominance (raw)"),
    ("combined_max * sd_prevalence", "combined_max (raw)"),
]
for new, base in pairs:
    if new in rdf.index and base in rdf.index:
        delta = rdf.loc[new, "cross_genome_auc"] - rdf.loc[base, "cross_genome_auc"]
        verdict = "ADD INTERACTION" if delta > 0.01 else ("marginal" if delta > 0 else "no benefit")
        print(f"  {new:<40} delta={delta:+.4f}  -> {verdict}")

# ── 4. Per-genome breakdown for rbs_max vs rbs_max*sd_prev ───────────────────

print(f"\n{SEP}")
print("4. PER-GENOME: rbs_dominance AUC vs rbs_dominance*sd_prevalence AUC")
print("   (Both should be equal within a genome -- confirms pooling is the mechanism)")
print(SEP)
print(f"  {'Acc':<15} {'sd_prev':>8} {'rbs_dom AUC':>12} {'rbs_dom*sd AUC':>14} {'delta':>7}")
print(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*14} {'-'*7}")
for acc, feat_df, labels in zip(TEST_GENOMES, [df_all.iloc[0:0]], [y_all[0:0]]):  # placeholder
    pass  # already pooled — use per-genome data we stored

# Re-run per genome for this section
per_genome_rows = []
idx = 0
all_feats_list = []
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
        [int(is_tp_group(gid, groups[gid], exact, stops)) for gid in group_ids], dtype=int
    )
    if len(labels) != len(feat_df):
        continue
    training_list = training.to_dict("records") if hasattr(training, "to_dict") else list(training)
    sd_prev = compute_sd_prevalence(seq, training_list)

    rbs_dom = feat_df["rbs_dominance"].fillna(0).values
    auc_raw = auc(rbs_dom, labels)
    auc_int = auc(rbs_dom * sd_prev, labels)
    delta = auc_int - auc_raw
    print(f"  {acc:<15} {sd_prev:>8.3f} {auc_raw:>12.4f} {auc_int:>14.4f} {delta:>+7.4f}")
    per_genome_rows.append(
        {
            "acc": acc,
            "sd_prev": sd_prev,
            "gc_pct": round(gc * 100, 1),
            "auc_rbs_dom": round(auc_raw, 4),
            "auc_rbs_dom_x_sd": round(auc_int, 4),
            "delta": round(delta, 4),
        }
    )

# ── Final verdict ─────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SUMMARY AND DECISION")
print(SEP)
if not high_corr:
    print("  sd_prevalence: INDEPENDENT signal (max corr < 0.95 with existing features)")
else:
    print(f"  sd_prevalence: COLLINEAR with {high_corr} -- cannot add")

print("\n  Cross-genome AUC comparison:")
for r in results:
    print(f"    {r['feature']:<40} {r['cross_genome_auc']:.4f}")

out = OUT_DIR / "sd_prevalence_diagnostic.csv"
pd.DataFrame(per_genome_rows).to_csv(out, index=False)
pd.DataFrame(results).to_csv(OUT_DIR / "sd_prevalence_auc_summary.csv", index=False)
print(f"\nSaved: {out}")
print(SEP)
