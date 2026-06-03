# EXPERIMENT: LGB rel_rbs_max variance collapse diagnostic
# STATUS: active
# RESULT: pending
"""
rel_rbs_max showed 80% variance collapse — for 80% of genomes, its within-genome
std is < 10% of the cross-genome mean std. This means the feature is nearly
constant (informationless noise) for most genomes.

Why it collapses:
  rel_rbs_max = max(rbs_score / max_rbs_in_group) across ORFs in a group
  When all groups in a genome have similar RBS scores (e.g., all strong or all weak),
  this ratio approaches a constant -> zero discriminative power within that genome.

Tests:
  1. For each genome: compute within-genome std of rel_rbs_max across groups
  2. Compare AUC of rel_rbs_max vs raw rbs_max in high-variance vs low-variance genomes
  3. Test gated version: rel_rbs_max_gated = rel_rbs_max if std > threshold else 0.5 (neutral)
  4. Compare AUC of raw vs gated

If gated version has higher mean AUC -> proceed with gating implementation.

Run from repo root:
    python scripts/experiments/analyze_lgb_rbs_variance.py
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
VAR_GATE_THRESHOLD = 0.05  # std below this -> collapse -> use neutral value
NEUTRAL_VALUE = 0.5  # neutral value for gated feature

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = set(c["e"].astype(int).tolist())
    return exact, stops


def is_tp_group(gid, gdf, exact, stops):
    """Does this group contain a real gene?"""
    if isinstance(gdf, pd.DataFrame):
        rows = gdf.to_dict("records")
    else:
        rows = list(gdf)
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


def auc_roc(feature_vals, labels):
    pos = feature_vals[labels == 1]
    neg = feature_vals[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.5
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    auc = u / (len(pos) * len(neg))
    return max(auc, 1 - auc)


SEP = "=" * 90
print(f"\n{SEP}")
print("DIAGNOSTIC: rel_rbs_max variance collapse — LGB feature")
print(f"  Testing whether gating when std < {VAR_GATE_THRESHOLD} improves discrimination")
print(SEP)

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
    if "rel_rbs_max" not in feat_df.columns or "rbs_max" not in feat_df.columns:
        continue

    # Labels: which groups are real genes?
    group_ids = list(groups.keys())
    labels = np.array(
        [int(is_tp_group(gid, groups[gid], exact, stops)) for gid in group_ids if gid in groups],
        dtype=int,
    )

    if len(labels) != len(feat_df):
        continue

    rel_rbs = feat_df["rel_rbs_max"].values
    raw_rbs = feat_df["rbs_max"].values

    # Variance of rel_rbs within this genome
    rel_rbs_std = float(rel_rbs.std())

    # Gated version: use neutral if std is low (feature collapsed)
    rel_rbs_gated = rel_rbs.copy()
    if rel_rbs_std < VAR_GATE_THRESHOLD:
        rel_rbs_gated[:] = NEUTRAL_VALUE

    auc_rel = auc_roc(rel_rbs, labels)
    auc_raw = auc_roc(raw_rbs, labels)
    auc_gate = auc_roc(rel_rbs_gated, labels)

    collapsed = rel_rbs_std < VAR_GATE_THRESHOLD

    print(
        f"    n_groups={len(feat_df)}  TP={labels.sum()}  "
        f"rel_rbs_std={rel_rbs_std:.4f}  collapsed={'YES' if collapsed else 'no'}"
    )
    print(
        f"    AUC: rel_rbs_max={auc_rel:.4f}  raw_rbs_max={auc_raw:.4f}  "
        f"gated={auc_gate:.4f}  "
        f"{'gated wins' if auc_gate > auc_rel + 0.005 else 'no improvement from gating'}"
    )

    rows.append(
        {
            "acc": acc,
            "gc_pct": round(gc * 100, 1),
            "n_groups": len(feat_df),
            "n_tp": int(labels.sum()),
            "rel_rbs_std": round(rel_rbs_std, 5),
            "collapsed": collapsed,
            "auc_rel_rbs": round(auc_rel, 4),
            "auc_raw_rbs": round(auc_raw, 4),
            "auc_gated": round(auc_gate, 4),
        }
    )

df = pd.DataFrame(rows)

print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
collapsed = df[df["collapsed"]]
not_collapsed = df[~df["collapsed"]]
print(
    f"\n  Genomes with variance collapse (std < {VAR_GATE_THRESHOLD}): {len(collapsed)}/{len(df)}"
)
print(f"\n  {'Group':<25} {'n':>4}  {'AUC rel_rbs':>12} {'AUC raw_rbs':>12} {'AUC gated':>10}")
print(f"  {'-'*25} {'-'*4}  {'-'*12} {'-'*12} {'-'*10}")
for label, sub in [("Collapsed genomes", collapsed), ("Non-collapsed", not_collapsed)]:
    if len(sub) == 0:
        continue
    print(
        f"  {label:<25} {len(sub):>4}  "
        f"{sub['auc_rel_rbs'].mean():>12.4f} {sub['auc_raw_rbs'].mean():>12.4f} "
        f"{sub['auc_gated'].mean():>10.4f}"
    )

print(f"\n  Overall mean AUC:")
print(f"    rel_rbs_max (current): {df['auc_rel_rbs'].mean():.4f}")
print(f"    raw_rbs_max:           {df['auc_raw_rbs'].mean():.4f}")
print(f"    gated (threshold={VAR_GATE_THRESHOLD}): {df['auc_gated'].mean():.4f}")

winner = max(
    [
        ("rel_rbs_max", df["auc_rel_rbs"].mean()),
        ("raw_rbs_max", df["auc_raw_rbs"].mean()),
        ("gated", df["auc_gated"].mean()),
    ],
    key=lambda x: x[1],
)
print(f"\n  WINNER: {winner[0]}  (AUC={winner[1]:.4f})")
if winner[0] == "gated":
    print(
        f"  -> Implement: rel_rbs_max_gated = rel_rbs_max if within_genome_std > {VAR_GATE_THRESHOLD} else {NEUTRAL_VALUE}"
    )
elif winner[0] == "raw_rbs_max":
    print(f"  -> Consider adding raw_rbs_max alongside or replacing rel_rbs_max")
else:
    print(f"  -> Keep current rel_rbs_max as-is")

df.to_csv(OUT_DIR / "lgb_rbs_variance_diagnostic.csv", index=False)
print(f"\nSaved: {OUT_DIR}/lgb_rbs_variance_diagnostic.csv")
print(SEP)
