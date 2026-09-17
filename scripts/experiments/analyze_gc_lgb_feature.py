# EXPERIMENT: Diagnostic for ML2 — GC% as explicit LGB feature
# STATUS: active
# RESULT: pending
"""
Tests whether genome_gc (and related sequence-level properties) predict F1
at the per-genome level, WITHOUT assuming phylum is the grouping variable.
Phylum is shown for context only — the causal analysis treats each genome
as an independent data point.

Three questions:
  1. What continuous genome properties correlate with F1?
     (GC%, genome_size, n_training_orfs, mean component scores...)
  2. Is genome_gc already captured by the existing 26 LGB features?
     (Pearson < 0.95 required per role_ml_engineer.md)
  3. Would a GC-aware LGB separate the low-F1 genomes from the high-F1 genomes?

Decision criteria (all must pass):
  A. |Pearson(genome_gc, F1)| > 0.40  AND genome_gc is among the top predictors
  B. max |Pearson(genome_gc, any existing LGB feature)| < 0.95
  C. The pattern is consistent: low-F1 genomes cluster in GC% space,
     not just in phylum space

Run from repo root:
    python scripts/experiments/analyze_gc_lgb_feature.py
"""

import contextlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, GENOME_CATALOG, START_SELECTION_WEIGHTS, TEST_GENOMES
from src.data_management import get_data_dir, load_genome_sequence
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
LOG_FILE = Path(__file__).parent.parent.parent / "experiments" / "log.json"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
CATALOG_SAMPLE_PER_PHYLUM = 5

_HOLDOUT_PHYLA = {
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
_CATALOG_PHYLA = {g["accession"]: g["group"] for g in GENOME_CATALOG}

_genome_f1 = {}
_HOLDOUT_SENTINEL = "NC_002947.4"  # always in TEST_GENOMES
if LOG_FILE.exists():
    log = json.load(open(LOG_FILE))
    # Find the best entry that actually contains the holdout genomes
    # (some entries use a different 42-genome catalog set — skip those)
    holdout_entries = [
        e for e in log if any(r.get("accession") == _HOLDOUT_SENTINEL for r in e.get("results", []))
    ]
    if holdout_entries:
        best = max(holdout_entries, key=lambda e: e.get("overall", {}).get("f1", 0))
        for r in best.get("results", []):
            _genome_f1[r["accession"]] = r["f1"]

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))

rng = random.Random(RANDOM_SEED)
catalog_by_phylum: dict[str, list[str]] = {}
for g in GENOME_CATALOG:
    catalog_by_phylum.setdefault(g["group"], []).append(g["accession"])

catalog_sample = []
for phylum, accs in catalog_by_phylum.items():
    available = [a for a in accs if Path(f"{DATA_DIR}/{a}.fasta").exists()]
    rng.shuffle(available)
    catalog_sample.extend(available[:CATALOG_SAMPLE_PER_PHYLUM])

EVAL_GENOMES = [
    (acc, "holdout", _HOLDOUT_PHYLA.get(acc, "?"))
    for acc in TEST_GENOMES
    if Path(f"{DATA_DIR}/{acc}.fasta").exists()
] + [
    (acc, "catalog", _CATALOG_PHYLA.get(acc, "?"))
    for acc in catalog_sample
    if Path(f"{DATA_DIR}/{acc}.fasta").exists()
]

# ── Main loop ─────────────────────────────────────────────────────────────────

SEP = "=" * 90
print(f"\n{SEP}")
print("DIAGNOSTIC ML2: Does GC% predict F1? (per-genome, phylum = context only)")
print(f"  {len(EVAL_GENOMES)} genomes")
print(SEP)

genome_rows = []
feature_rows = []

for idx, (acc, source, phylum) in enumerate(EVAL_GENOMES, 1):
    print(f"  [{idx:>2}/{len(EVAL_GENOMES)}] {acc} [{phylum}] ({source})", flush=True)
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    genome_gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    genome_size = len(seq)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups = organize_nested_orfs(filtered)
        feat_df = lgb.extract_group_features(groups, genome_id=acc, weights=START_SELECTION_WEIGHTS)

    scored_df = pd.DataFrame(scored) if not isinstance(scored, pd.DataFrame) else scored
    mean_codon = (
        scored_df["codon_score"].mean() if "codon_score" in scored_df.columns else float("nan")
    )
    mean_imm = scored_df["imm_score"].mean() if "imm_score" in scored_df.columns else float("nan")
    mean_rbs = scored_df["rbs_score"].mean() if "rbs_score" in scored_df.columns else float("nan")

    f1 = _genome_f1.get(acc, float("nan"))
    genome_rows.append(
        {
            "accession": acc,
            "source": source,
            "phylum": phylum,  # context only, not used in regression
            "gc_pct": round(genome_gc * 100, 2),
            "genome_size_mb": round(genome_size / 1e6, 2),
            "n_training": len(training),
            "n_orfs": len(orfs),
            "n_groups": len(feat_df),
            "mean_codon": round(mean_codon, 4),
            "mean_imm": round(mean_imm, 4),
            "mean_rbs": round(mean_rbs, 4),
            "f1_pct": round(f1, 2) if not np.isnan(f1) else float("nan"),
        }
    )

    feat_df = feat_df.drop(columns=["group_id"], errors="ignore")
    feat_df["genome_gc"] = genome_gc
    feat_df["phylum"] = phylum
    feat_df["accession"] = acc
    feature_rows.append(feat_df)

gdf = pd.DataFrame(genome_rows)
all_feats = pd.concat(feature_rows, ignore_index=True)

# ── Report ────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("1. PER-GENOME PROPERTIES vs F1  (sorted by F1, phylum shown for context only)")
print(SEP)
holdout = gdf[gdf["source"] == "holdout"].copy().sort_values("f1_pct")
print(
    f"  {'Accession':<16} {'Phylum':^16} {'GC%':>5} {'SizeMb':>7} {'n_train':>8} "
    f"{'mean_codon':>11} {'mean_rbs':>9} {'F1%':>6}"
)
print(f"  {'-'*16} {'-'*16} {'-'*5} {'-'*7} {'-'*8} {'-'*11} {'-'*9} {'-'*6}")
for _, r in holdout.iterrows():
    f1s = f"{r['f1_pct']:.2f}" if not np.isnan(r["f1_pct"]) else "  n/a"
    print(
        f"  {r['accession']:<16} {r['phylum']:^16} {r['gc_pct']:>5.1f} "
        f"{r['genome_size_mb']:>7.2f} {r['n_training']:>8} "
        f"{r['mean_codon']:>11.4f} {r['mean_rbs']:>9.4f} {f1s:>6}"
    )

print(f"\n{SEP}")
print("2. PEARSON CORRELATION OF CONTINUOUS PROPERTIES WITH F1  (holdout genomes only)")
print("   Each row is a genome-level predictor — no phylum grouping")
print(SEP)
holdout_nona = holdout.dropna(subset=["f1_pct"])
predictors = [
    "gc_pct",
    "genome_size_mb",
    "n_training",
    "n_orfs",
    "n_groups",
    "mean_codon",
    "mean_imm",
    "mean_rbs",
]
print(f"  {'Predictor':<18} {'r(x, F1)':>10}  {'|r|':>6}  {'p-value':>10}  Assessment")
print(f"  {'-'*18} {'-'*10}  {'-'*6}  {'-'*10}  {'-'*25}")
gc_r = None
for pred in predictors:
    col = holdout_nona[pred]
    if col.std() < 1e-9:
        continue
    r, p = pearsonr(col, holdout_nona["f1_pct"])
    if pred == "gc_pct":
        gc_r = r
    strength = "STRONG" if abs(r) > 0.60 else ("MODERATE" if abs(r) > 0.40 else "weak")
    print(f"  {pred:<18} {r:>+10.4f}  {abs(r):>6.4f}  {p:>10.4f}  {strength}")

if gc_r is not None:
    print(
        f"\n  Criterion A: |r(gc_pct, F1)| = {abs(gc_r):.4f} "
        f"-> {'PASS (> 0.40)' if abs(gc_r) > 0.40 else 'FAIL (<= 0.40)'}"
    )

print(f"\n{SEP}")
print("3. CORRELATION OF genome_gc WITH EXISTING 26 LGB FEATURES")
print("   Criterion B: all correlations must be < 0.95")
print(SEP)
feat_cols = [c for c in all_feats.columns if c not in ("genome_gc", "phylum", "accession")]
corrs = []
for col in feat_cols:
    vals = all_feats[col].fillna(0).values
    gc_vals = all_feats["genome_gc"].values
    if vals.std() < 1e-9:
        r_val = 0.0
    else:
        r_val, _ = pearsonr(gc_vals, vals)
    corrs.append((col, r_val))

corrs.sort(key=lambda x: -abs(x[1]))
max_r = max(abs(r) for _, r in corrs)
print(f"  Top correlations (sorted by |r|):")
print(f"  {'Feature':<30} {'r(gc, feat)':>12}")
print(f"  {'-'*30} {'-'*12}")
for col, r_val in corrs[:12]:
    flag = "  <- check" if abs(r_val) > 0.70 else ""
    print(f"  {col:<30} {r_val:>+12.4f}{flag}")
print(f"\n  Max |r| with any existing feature: {max_r:.4f}")
print(
    f"  Criterion B: {'PASS (< 0.95)' if max_r < 0.95 else 'FAIL (>= 0.95) -- GC already captured'}"
)

print(f"\n{SEP}")
print("4. LOW-F1 GENOMES: What do they have in common? (genome-level clustering)")
print("   Ignoring phylum -- looking at what distinguishes F1 < 65% from F1 >= 65%")
print(SEP)
if holdout_nona["f1_pct"].notna().sum() >= 4:
    low = holdout_nona[holdout_nona["f1_pct"] < 65]
    high = holdout_nona[holdout_nona["f1_pct"] >= 65]
    print(
        f"  Low-F1 genomes  (F1 < 65%, n={len(low)}):  "
        f"GC mean={low['gc_pct'].mean():.1f}%  "
        f"size mean={low['genome_size_mb'].mean():.2f}Mb  "
        f"n_train mean={low['n_training'].mean():.0f}  "
        f"mean_codon mean={low['mean_codon'].mean():.4f}"
    )
    print(
        f"  High-F1 genomes (F1 >= 65%, n={high['f1_pct'].count()}):  "
        f"GC mean={high['gc_pct'].mean():.1f}%  "
        f"size mean={high['genome_size_mb'].mean():.2f}Mb  "
        f"n_train mean={high['n_training'].mean():.0f}  "
        f"mean_codon mean={high['mean_codon'].mean():.4f}"
    )
    print(f"\n  Low-F1 genome list (sorted by F1):")
    for _, r in low.sort_values("f1_pct").iterrows():
        print(
            f"    {r['accession']:<16} [{r['phylum']:^16}]  "
            f"GC={r['gc_pct']:.1f}%  size={r['genome_size_mb']:.2f}Mb  "
            f"n_train={r['n_training']}  F1={r['f1_pct']:.2f}%"
        )

out_path = OUT_DIR / "gc_lgb_diagnostic.csv"
gdf.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(SEP)
print("DECISION:")
print("  A PASS + B PASS -> implement genome_gc as LGB feature (+ check gc3_mean)")
print("  A FAIL -> GC% doesn't explain F1 gap; look at other predictors in Section 2")
print("  B FAIL -> GC% already implicit in existing features; won't help")
print(SEP)
