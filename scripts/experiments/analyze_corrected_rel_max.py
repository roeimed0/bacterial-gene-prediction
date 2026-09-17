# EXPERIMENT: Corrected rel_*_max features — dominance ratio diagnostic
# STATUS: active
# RESULT: pending
"""
Tests whether the CORRECTED rel_*_max features add discriminative power.

BROKEN (always 1.0):    rel_rbs_max = max(rbs / max_rbs) = max_rbs/max_rbs = 1.0
CORRECTED (meaningful): rel_rbs_max = rbs_max / rbs_mean   (dominance ratio)

The corrected formula answers: "how much does the best ORF's RBS outperform
the group average?" High ratio = one ORF clearly dominates = more likely real gene.

Tests whether corrected ratio has HIGHER AUC than rbs_max or rbs_mean alone.
If it does -> it adds new information beyond what's already in the model.

Run from repo root:
    python scripts/experiments/analyze_corrected_rel_max.py
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
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
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


SEP = "=" * 85
print(f"\n{SEP}")
print("DIAGNOSTIC: Corrected rel_*_max (dominance ratio) vs existing features")
print(f"  Corrected: rel_rbs_max = rbs_max / rbs_mean  (not max(rbs/max_rbs) = 1.0)")
print(SEP)

focus = [
    "NC_002929.2",
    "NC_002677.1",
    "NC_008268.1",
    "NC_003155.5",
    "NC_003030.1",
    "NC_004350.2",
    "NC_015948.1",
    "NC_008818.1",
]

results = []

for acc in focus:
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

    # Compute corrected dominance ratios from existing features
    eps = 1e-9
    rbs_ratio = feat_df["rbs_max"] / (feat_df["rbs_mean"] + eps)
    codon_ratio = feat_df["codon_max"] / (feat_df["codon_mean"] + eps)
    imm_ratio = feat_df["imm_max"] / (feat_df["imm_mean"] + eps)
    comb_ratio = feat_df["combined_max"] / (feat_df["combined_mean"] + eps)
    ss_ratio = feat_df["start_select_max"] / (feat_df["start_select_mean"] + eps)

    print(f"    n_groups={len(feat_df)}  TP={labels.sum()}")
    print(f"    {'Feature':<30} {'AUC':>7}  vs existing best")
    print(f"    {'-'*30} {'-'*7}  {'-'*20}")

    genome_results = {}
    for name, vals in [
        ("rbs_max (existing)", feat_df["rbs_max"].values),
        ("rbs_mean (existing)", feat_df["rbs_mean"].values),
        ("rbs_max/rbs_mean NEW", rbs_ratio.values),
        ("codon_max (existing)", feat_df["codon_max"].values),
        ("codon_max/mean NEW", codon_ratio.values),
        ("imm_max (existing)", feat_df["imm_max"].values),
        ("imm_max/mean NEW", imm_ratio.values),
        ("combined_max/mean NEW", comb_ratio.values),
        ("ss_max/mean NEW", ss_ratio.values),
    ]:
        a = auc(vals, labels)
        genome_results[name] = a
        tag = "NEW" if "NEW" in name else "   "
        print(f"    {name:<30} {a:>7.4f}  {tag}")

    results.append({"acc": acc, **genome_results})

print(f"\n{SEP}")
print("CROSS-GENOME MEAN AUC")
print(SEP)
df = pd.DataFrame(results)
for col in df.columns[1:]:
    print(f"  {col:<30} {df[col].mean():.4f}")

print(f"\n{SEP}")
print("DECISION: Does any corrected ratio beat its existing component features?")
print(SEP)
pairs = [
    ("rbs_max/rbs_mean NEW", ["rbs_max (existing)", "rbs_mean (existing)"]),
    ("codon_max/mean NEW", ["codon_max (existing)"]),
    ("imm_max/mean NEW", ["imm_max (existing)"]),
    ("combined_max/mean NEW", [c for c in df.columns if "combined_max" in c and "NEW" not in c]),
    ("ss_max/mean NEW", [c for c in df.columns if "ss_max" in c and "NEW" not in c]),
]
for ratio_name, compare_names in pairs:
    if ratio_name not in df.columns:
        continue
    ratio_auc = df[ratio_name].mean()
    available = [c for c in compare_names if c in df.columns]
    if not available:
        print(f"  {ratio_name:<30} AUC={ratio_auc:.4f}  (no existing comparison column found)")
        continue
    best_existing = max(df[c].mean() for c in available)
    delta = ratio_auc - best_existing
    verdict = "ADD to model" if delta > 0.01 else ("marginal" if delta > 0 else "no benefit")
    print(
        f"  {ratio_name:<30} AUC={ratio_auc:.4f}  best_existing={best_existing:.4f}  "
        f"delta={delta:+.4f}  -> {verdict}"
    )

df.to_csv(OUT_DIR / "corrected_rel_max_diagnostic.csv", index=False)
print(f"\nSaved: {OUT_DIR}/corrected_rel_max_diagnostic.csv")
print(SEP)
