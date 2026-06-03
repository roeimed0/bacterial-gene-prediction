# EXPERIMENT: Systematic gating candidate analysis — all 3 models
# STATUS: active
# RESULT: pending
"""
Tests the gating approach on all remaining toggleable candidates:
  gated_feature = max(0, feature - floor)
  where floor = 50th percentile across low-GC genomes (gc < 0.55)

This makes the feature 0 (neutral) for low-GC genomes and positive for
high-GC genomes — same pattern as genome_gc_high which gave +0.26pp.

Models and features tested:
  LGB: rel_start_select_mean (35% variance collapse)
  Hybrid: small_fraction, gc_skew, codon_bias_index, stop_codon_type, purine_content
  SSC: frac_atg, any_stop_top2, d_genome_rbs (all GC-correlated pairwise features)

For each feature: compare AUC of raw vs gated across all 20 genomes.
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
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
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
GC_FLOOR = 0.55  # same as genome_gc_high

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))


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


def is_tp_cand(cand, exact, stops):
    gs = int(cand.get("genome_start", cand.get("start", 0)))
    ge = int(cand.get("genome_end", cand.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in stops)


def auc(vals, labels):
    pos = vals[labels == 1]
    neg = vals[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.5
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    a = u / (len(pos) * len(neg))
    return max(a, 1 - a)


def gate(vals, floor):
    """max(0, value - floor) — same pattern as genome_gc_high."""
    return np.maximum(0.0, vals - floor)


SEP = "=" * 90
print(f"\n{SEP}")
print("SYSTEMATIC GATING ANALYSIS: all remaining toggleable candidates")
print(f"  Gate: max(0, feature - floor) where floor = 50th percentile for gc < {GC_FLOOR}")
print(f"  Criterion: gated AUC > raw AUC + 0.01 across most genomes -> ADD to model")
print(SEP)

# ── Collect data from all 20 genomes ──────────────────────────────────────────

lgb_data, hf_data = [], []  # (gc, feat_df, labels)

for acc in TEST_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    try:
        exact, stops = load_ref(acc)
    except:
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    print(f"  {acc}  gc={gc*100:.1f}%", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models_ = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models_)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups = organize_nested_orfs(filtered)

        lgb_feat = lgb.extract_group_features(
            groups, acc, weights=START_SELECTION_WEIGHTS, genome_gc=gc
        )
        lgb_feat = lgb_feat.drop(columns=["group_id"], errors="ignore")

        cands_list = filtered.to_dict("records") if hasattr(filtered, "to_dict") else list(filtered)
        hf_feat = hf.extract_features(cands_list, genome_id=acc, genome_gc=gc)

    # LGB labels (group level)
    gids = list(groups.keys())
    lgb_labels = np.array(
        [int(is_tp_group(gid, groups[gid], exact, stops)) for gid in gids], dtype=int
    )
    if len(lgb_labels) == len(lgb_feat):
        lgb_data.append((gc, lgb_feat, lgb_labels))

    # Hybrid labels (candidate level)
    hf_labels = np.array([int(is_tp_cand(c, exact, stops)) for c in cands_list], dtype=int)
    if len(hf_labels) == len(hf_feat):
        hf_data.append((gc, hf_feat, hf_labels))

# ── Compute floor from low-GC genomes ────────────────────────────────────────


def compute_floor(data, feat_col, gc_floor=GC_FLOOR):
    """50th percentile of feature across low-GC genomes."""
    low_vals = []
    for gc, df, _ in data:
        if gc < gc_floor and feat_col in df.columns:
            low_vals.extend(df[feat_col].fillna(0).values.tolist())
    return float(np.percentile(low_vals, 50)) if low_vals else 0.0


# ── LGB analysis ─────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("LGB (OrfGroupClassifier) — remaining candidates")
print(SEP)

lgb_candidates = [
    "rel_start_select_mean",
    "rel_rbs_mean",
    "rel_combined_mean",
    "rel_codon_mean",
    "rel_start_mean",
]

lgb_results = []
for feat in lgb_candidates:
    avail = [(gc, df, lab) for gc, df, lab in lgb_data if feat in df.columns]
    if not avail:
        continue
    floor = compute_floor(avail, feat)
    raw_aucs, gated_aucs = [], []
    for gc, df, labels in avail:
        raw = df[feat].fillna(0).values
        gt = gate(raw, floor)
        raw_aucs.append(auc(raw, labels))
        gated_aucs.append(auc(gt, labels))
    delta = np.mean(gated_aucs) - np.mean(raw_aucs)
    verdict = "GATE IT" if delta > 0.01 else ("marginal" if delta > 0 else "no benefit")
    print(
        f"  {feat:<35}  raw={np.mean(raw_aucs):.4f}  gated={np.mean(gated_aucs):.4f}  "
        f"delta={delta:+.4f}  floor={floor:.4f}  -> {verdict}"
    )
    lgb_results.append(
        {
            "model": "LGB",
            "feature": feat,
            "raw_auc": round(np.mean(raw_aucs), 4),
            "gated_auc": round(np.mean(gated_aucs), 4),
            "delta": round(delta, 4),
            "floor": round(floor, 4),
            "verdict": verdict,
        }
    )

# ── Hybrid analysis ───────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("HybridGeneFilter — GC-correlated candidates")
print(SEP)

hf_candidates = [
    "small_fraction",
    "gc_skew",
    "codon_bias_index",
    "stop_codon_type",
    "purine_content",
    "gc_content",
    "gc3_content",
]

hf_results = []
for feat in hf_candidates:
    avail = [(gc, df, lab) for gc, df, lab in hf_data if feat in df.columns]
    if not avail:
        continue
    floor = compute_floor(avail, feat)
    raw_aucs, gated_aucs = [], []
    for gc, df, labels in avail:
        raw = df[feat].fillna(0).values
        gt = gate(raw, floor)
        raw_aucs.append(auc(raw, labels))
        gated_aucs.append(auc(gt, labels))
    delta = np.mean(gated_aucs) - np.mean(raw_aucs)
    verdict = "GATE IT" if delta > 0.01 else ("marginal" if delta > 0 else "no benefit")
    print(
        f"  {feat:<35}  raw={np.mean(raw_aucs):.4f}  gated={np.mean(gated_aucs):.4f}  "
        f"delta={delta:+.4f}  floor={floor:.4f}  -> {verdict}"
    )
    hf_results.append(
        {
            "model": "Hybrid",
            "feature": feat,
            "raw_auc": round(np.mean(raw_aucs), 4),
            "gated_auc": round(np.mean(gated_aucs), 4),
            "delta": round(delta, 4),
            "floor": round(floor, 4),
            "verdict": verdict,
        }
    )

# ── SSC analysis (using pairwise CSV) ────────────────────────────────────────

print(f"\n{SEP}")
print("StartSelectionClassifier — GC-correlated pairwise features")
print(SEP)

pairwise_csv = OUT_DIR / "pairwise_features_v2.csv"
if pairwise_csv.exists():
    pf = pd.read_csv(pairwise_csv)
    gc_map = {}
    for acc in pf["acc"].unique():
        fasta = f"{DATA_DIR}/{acc}.fasta"
        if Path(fasta).exists():
            g = load_genome_sequence(fasta)
            s = g["sequence"]
            gc_map[acc] = (s.count("G") + s.count("C")) / max(len(s), 1)
    pf["genome_gc"] = pf["acc"].map(gc_map)
    pf = pf.dropna(subset=["genome_gc"])

    ssc_candidates = [
        "frac_atg",
        "any_stop_top2",
        "any_stop_top1",
        "d_genome_rbs",
        "grp_rbs_mean",
        "both_atg",
        "score_range",
        "d_ctx_pwm",
        "anti_sd_top1",
    ]
    ssc_results = []
    label_col = "label" if "label" in pf.columns else None

    if label_col:
        for feat in ssc_candidates:
            if feat not in pf.columns:
                continue
            # Floor from low-GC genomes in pairwise CSV
            low = pf[pf["genome_gc"] < GC_FLOOR][feat].fillna(0).values
            floor = float(np.percentile(low, 50)) if len(low) > 0 else 0.0

            raw = pf[feat].fillna(0).values
            labs = pf[label_col].values
            gt = gate(raw, floor)
            a_raw = auc(raw, labs)
            a_gt = auc(gt, labs)
            delta = a_gt - a_raw
            verdict = "GATE IT" if delta > 0.01 else ("marginal" if delta > 0 else "no benefit")
            print(
                f"  {feat:<35}  raw={a_raw:.4f}  gated={a_gt:.4f}  "
                f"delta={delta:+.4f}  floor={floor:.4f}  -> {verdict}"
            )
            ssc_results.append(
                {
                    "model": "SSC",
                    "feature": feat,
                    "raw_auc": round(a_raw, 4),
                    "gated_auc": round(a_gt, 4),
                    "delta": round(delta, 4),
                    "floor": round(floor, 4),
                    "verdict": verdict,
                }
            )
    else:
        print("  label column not found in pairwise CSV")
        ssc_results = []
else:
    print("  pairwise_features_v2.csv not found")
    ssc_results = []

# ── Final summary ─────────────────────────────────────────────────────────────

all_results = lgb_results + hf_results + ssc_results
df_out = pd.DataFrame(all_results)
df_out.to_csv(OUT_DIR / "gating_candidates_analysis.csv", index=False)

print(f"\n{SEP}")
print("SUMMARY: Features worth gating (delta > +0.01 AUC)")
print(SEP)
winners = df_out[df_out["verdict"] == "GATE IT"].sort_values("delta", ascending=False)
if len(winners):
    print(f"  {'Model':<12} {'Feature':<35} {'delta':>7}  {'floor':>8}")
    for _, r in winners.iterrows():
        print(f"  {r['model']:<12} {r['feature']:<35} {r['delta']:>+7.4f}  {r['floor']:>8.4f}")
else:
    print("  No features showed meaningful improvement from gating.")
print(f"\nSaved: {OUT_DIR}/gating_candidates_analysis.csv")
print(SEP)
