# EXPERIMENT: Diagnostic for ML1 (leaderless detection) + ML4 (adaptive weights)
# STATUS: concluded
# RESULT: REJECTED. SD prevalence is 89-98% for ALL phyla including Archaea — no threshold
#   separates leaderless from SD-using genomes. Adaptive weights are identical across all
#   phyla (adp_rbs=0.957 for all 4 phyla). Root cause: (1) our 5 Archaea holdout genomes
#   are Euryarchaeota which DO use SD, not leaderless Crenarchaeota; (2) RBS scorer
#   detects purine-richness in general, not SD motifs specifically — bimodal output
#   (-5.0 or >3.0) makes sd_prevalence uninformative at any threshold.
"""
Runs on ALL TEST_GENOMES (20) + a stratified sample of GENOME_CATALOG (5 per phylum = 20)
to measure, across all 4 phyla:

  1. sd_prevalence  — fraction of training ORFs with rbs_score > threshold
                      (proxy for whether a genome uses Shine-Dalgarno)
  2. Per-component score variance from scored ORFs
                      (signal quality; low variance → weight should decrease)
  3. What adaptive START_SELECTION_WEIGHTS would look like under ML4
  4. Current F1 from experiments/log.json for baseline comparison

Decision criteria before implementing ML1/ML4:
  A. SD prevalence must clearly separate Archaea from all 3 bacterial groups
     → Archaea mean << 0.20, bacteria mean >> 0.20, minimal overlap
  B. Adaptive RBS weight must decrease for Archaea but not for Firmicutes
  C. No bacterial phylum should trigger the leaderless RBS-zero rule

Run from repo root:
    python scripts/experiments/analyze_adaptive_scoring.py
"""

import contextlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import GENOME_CATALOG, START_SELECTION_WEIGHTS, TEST_GENOMES
from src.data_management import get_data_dir, load_genome_sequence
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    find_orfs_candidates,
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
LOG_FILE = Path(__file__).parent.parent.parent / "experiments" / "log.json"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
CATALOG_SAMPLE_PER_PHYLUM = 5  # genomes sampled from GENOME_CATALOG per phylum
SD_LEADERLESS_THRESHOLD = 0.20  # candidate threshold for ML1

# ── Phylum lookup ─────────────────────────────────────────────────────────────

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
_ALL_PHYLA = {**_HOLDOUT_PHYLA, **_CATALOG_PHYLA}

# Load benchmark F1 per genome (best run from log.json)
_genome_f1 = {}
if LOG_FILE.exists():
    log = json.load(open(LOG_FILE))
    if log:
        best = max(log, key=lambda e: e.get("overall", {}).get("f1", 0))
        for r in best.get("results", []):
            _genome_f1[r["accession"]] = r["f1"]

# Build genome list: 20 holdout + 5 per phylum from catalog (available on disk)
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


# ── Helper functions ──────────────────────────────────────────────────────────


def compute_sd_prevalence(training_orfs):
    """Fraction of training ORFs with rbs_score above 0, 1, and 2."""
    scores = [o.get("rbs_score", -5.0) for o in training_orfs]
    if not scores:
        return 0.0, 0.0, 0.0
    n = len(scores)
    return (
        sum(1 for s in scores if s > 0.0) / n,
        sum(1 for s in scores if s > 1.0) / n,
        sum(1 for s in scores if s > 2.0) / n,
    )


def compute_score_variance(scored_df):
    """Per-component variance of normalised scores across all scored ORFs."""
    cols = {
        "codon": "codon_score_norm",
        "imm": "imm_score_norm",
        "rbs": "rbs_score_norm",
        "length": "length_score_norm",
        "start": "start_score_norm",
    }
    return {
        key: float(scored_df[col].var()) if col in scored_df.columns else 0.0
        for key, col in cols.items()
    }


def compute_adaptive_weights(variances, base=START_SELECTION_WEIGHTS, cap=0.50):
    """Scale weights by variance, capped at ±50% of baseline."""
    total_var = sum(variances.values())
    if total_var < 1e-9:
        return dict(base)
    total_base = sum(base.values())
    return {
        k: float(
            np.clip(
                (variances.get(k, 0) / total_var) * total_base,
                base[k] * (1 - cap),
                base[k] * (1 + cap),
            )
        )
        for k in base
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

SEP = "=" * 95
print(f"\n{SEP}")
print(f"DIAGNOSTIC: SD Prevalence + Score Variance across all 4 phyla")
print(
    f"  Holdout genomes: {sum(1 for _, s, _ in EVAL_GENOMES if s == 'holdout')}  "
    f"Catalog sample: {sum(1 for _, s, _ in EVAL_GENOMES if s == 'catalog')}"
)
print(f"  SD leaderless threshold (candidate): {SD_LEADERLESS_THRESHOLD}")
print(SEP)

rows = []
for idx, (acc, source, phylum) in enumerate(EVAL_GENOMES, 1):
    print(f"  [{idx:>2}/{len(EVAL_GENOMES)}] {acc} [{phylum}] ({source})", flush=True)
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    genome_gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored_list = score_all_orfs(orfs, models)

    scored_df = (
        pd.DataFrame(scored_list) if not isinstance(scored_list, pd.DataFrame) else scored_list
    )

    sd0, sd1, sd2 = compute_sd_prevalence(training)
    variances = compute_score_variance(scored_df)
    adaptive_w = compute_adaptive_weights(variances)
    would_zero_rbs = sd0 < SD_LEADERLESS_THRESHOLD

    rows.append(
        {
            "accession": acc,
            "source": source,
            "phylum": phylum,
            "gc_pct": round(genome_gc * 100, 1),
            "n_train": len(training),
            "sd_prev_gt0": round(sd0, 3),
            "sd_prev_gt1": round(sd1, 3),
            "sd_prev_gt2": round(sd2, 3),
            "zero_rbs_ml1": would_zero_rbs,
            "var_codon": round(variances["codon"], 4),
            "var_imm": round(variances["imm"], 4),
            "var_rbs": round(variances["rbs"], 4),
            "var_length": round(variances["length"], 4),
            "var_start": round(variances["start"], 4),
            "adp_codon": round(adaptive_w["codon"], 3),
            "adp_imm": round(adaptive_w["imm"], 3),
            "adp_rbs": round(adaptive_w["rbs"], 3),
            "adp_length": round(adaptive_w["length"], 3),
            "adp_start": round(adaptive_w["start"], 3),
            "f1_pct": round(_genome_f1.get(acc, float("nan")), 2),
        }
    )

df = pd.DataFrame(rows)

# ── Report ────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("1. SD PREVALENCE PER GENOME  (fraction of training ORFs with rbs_score > threshold)")
print(
    f"   Baseline weights: codon={START_SELECTION_WEIGHTS['codon']:.2f}  "
    f"rbs={START_SELECTION_WEIGHTS['rbs']:.2f}  "
    f"length={START_SELECTION_WEIGHTS['length']:.2f}"
)
print(SEP)
print(
    f"  {'Accession':<16} {'Phylum':<16} {'Src':<8} {'GC%':>5} "
    f"{'SD>0':>6} {'SD>1':>6} {'SD>2':>6} {'ZeroRBS?':>9} {'F1%':>6}"
)
print(f"  {'-'*16} {'-'*16} {'-'*8} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*9} {'-'*6}")
for phylum in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
    g = df[df["phylum"] == phylum]
    for _, r in g.iterrows():
        zero = "YES !" if r["zero_rbs_ml1"] else "no"
        f1s = f"{r['f1_pct']:.2f}" if not np.isnan(r["f1_pct"]) else "  n/a"
        print(
            f"  {r['accession']:<16} {phylum:<16} {r['source']:<8} {r['gc_pct']:>5.1f} "
            f"{r['sd_prev_gt0']:>6.3f} {r['sd_prev_gt1']:>6.3f} {r['sd_prev_gt2']:>6.3f} "
            f"{zero:>9} {f1s:>6}"
        )
    print()

print(f"\n{SEP}")
print("2. PHYLUM SUMMARY: Mean SD Prevalence + Adaptive RBS Weight")
print(f"   Baseline RBS weight = {START_SELECTION_WEIGHTS['rbs']:.4f}")
print(SEP)
print(
    f"  {'Phylum':<18} {'n':>3}  {'SD>0 mean':>10} {'SD>0 min':>9} {'SD>0 max':>9}  "
    f"{'ZeroRBS':>8}  {'adp_rbs mean':>13}  {'F1 mean':>8}"
)
print(f"  {'-'*18} {'-'*3}  {'-'*10} {'-'*9} {'-'*9}  {'-'*8}  {'-'*13}  {'-'*8}")
for phylum in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
    g = df[df["phylum"] == phylum]
    if len(g) == 0:
        continue
    n_zero = g["zero_rbs_ml1"].sum()
    f1_mean = g["f1_pct"].mean()
    print(
        f"  {phylum:<18} {len(g):>3}  "
        f"{g['sd_prev_gt0'].mean():>10.3f} {g['sd_prev_gt0'].min():>9.3f} "
        f"{g['sd_prev_gt0'].max():>9.3f}  "
        f"{n_zero:>3}/{len(g):<5}  "
        f"{g['adp_rbs'].mean():>13.3f}  "
        f"{f1_mean:>8.2f}%"
    )

print(f"\n{SEP}")
print("3. THRESHOLD SWEEP: At each sd_prevalence threshold, how many genomes trigger ZeroRBS?")
print("   Goal: threshold that catches all/most Archaea and zero non-Archaea")
print(SEP)
print(
    f"  {'Threshold':>10}  {'Total':>6}  {'Archaea':>8}  {'Protek':>7}  "
    f"{'Firm':>6}  {'Actin':>7}  {'Assessment'}"
)
print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*30}")
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    hit = df["sd_prev_gt0"] < thr
    total = hit.sum()
    arch = ((df["phylum"] == "Archaea") & hit).sum()
    prot = ((df["phylum"] == "Proteobacteria") & hit).sum()
    firm = ((df["phylum"] == "Firmicutes") & hit).sum()
    acti = ((df["phylum"] == "Actinobacteria") & hit).sum()
    n_arch = (df["phylum"] == "Archaea").sum()
    assess = ""
    if prot == 0 and firm == 0 and acti == 0:
        assess = "CLEAN -- only Archaea"
    elif prot == 0 and firm == 0:
        assess = "~ Archaea + some Actinobacteria"
    else:
        assess = f"WARN: hits {prot} Prot, {firm} Firm, {acti} Acti"
    print(
        f"  {thr:>10.2f}  {total:>6}  {arch:>4}/{n_arch:<3}  {prot:>7}  "
        f"{firm:>6}  {acti:>7}  {assess}"
    )

print(f"\n{SEP}")
print("4. ADAPTIVE WEIGHT ANALYSIS: Do weights shift in the right direction?")
print(
    f"   Baseline: codon={START_SELECTION_WEIGHTS['codon']:.3f}  "
    f"imm={START_SELECTION_WEIGHTS['imm']:.3f}  "
    f"rbs={START_SELECTION_WEIGHTS['rbs']:.3f}  "
    f"length={START_SELECTION_WEIGHTS['length']:.3f}  "
    f"start={START_SELECTION_WEIGHTS['start']:.3f}"
)
print(f"   Expected: Archaea adp_rbs < baseline; others adp_rbs ~= baseline")
print(SEP)
for phylum in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
    g = df[df["phylum"] == phylum]
    if len(g) == 0:
        continue
    rbs_delta = g["adp_rbs"].mean() - START_SELECTION_WEIGHTS["rbs"]
    len_delta = g["adp_length"].mean() - START_SELECTION_WEIGHTS["length"]
    codon_delta = g["adp_codon"].mean() - START_SELECTION_WEIGHTS["codon"]
    print(
        f"  {phylum:<18}  adp_rbs={g['adp_rbs'].mean():.3f} (d={rbs_delta:+.3f})  "
        f"adp_length={g['adp_length'].mean():.3f} (d={len_delta:+.3f})  "
        f"adp_codon={g['adp_codon'].mean():.3f} (d={codon_delta:+.3f})"
    )

# Save
out_path = OUT_DIR / "adaptive_scoring_diagnostic.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(SEP)
print("INTERPRETATION GUIDE:")
print("  Section 3: find highest threshold where assess = 'CLEAN'")
print("  Section 4: adp_rbs for Archaea should be < baseline; bacteria near baseline")
print("  If both criteria met -> proceed with ML1+ML4 implementation")
print("  If not -> adjust threshold or reconsider approach before touching production code")
print(SEP)
