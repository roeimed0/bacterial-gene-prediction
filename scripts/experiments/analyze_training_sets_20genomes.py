# EXPERIMENT: Training set TP/FP breakdown — all 20 holdout genomes, baseline vs filtered
# STATUS: active
# RESULT: pending
"""
For each of the 20 TEST_GENOMES, shows the training set composition
(n_total, TP, FP, TP%) under two configurations:
  1. BASELINE: current create_training_set() (no filter)
  2. FILTERED: adaptive codon-position GC bias filter
     → if genome_gc >= 0.55: keep only ORFs with |GC3-GC12| >= 0.08
     → if genome_gc <  0.55: no filter (signal too weak)

This gives us the expected training set quality improvement before running
the full 25-minute benchmark.

Run from repo root:
    python scripts/experiments/analyze_training_sets_20genomes.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import TEST_GENOMES
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    _gc_position_bias,
    create_training_set,
    filter_training_by_codon_position_bias,
    find_orfs_candidates,
)

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3

FILTER_GC_THRESHOLD = 0.55  # only filter genomes above this GC%
FILTER_BIAS_THRESHOLD = 0.08  # |GC3-GC12| minimum

HOLDOUT_META = {
    "NC_002947.4": ("P. putida", "Proteobacteria"),
    "NC_002929.2": ("B. pertussis", "Proteobacteria"),
    "NC_003143.1": ("Y. pestis", "Proteobacteria"),
    "NC_003116.1": ("N. meningitidis", "Proteobacteria"),
    "NC_004757.1": ("N. europaea", "Proteobacteria"),
    "NC_008497.1": ("L. brevis", "Firmicutes"),
    "NC_004350.2": ("S. agalactiae", "Firmicutes"),
    "NC_006270.3": ("B. licheniformis", "Firmicutes"),
    "NC_006274.1": ("B. cereus", "Firmicutes"),
    "NC_003030.1": ("C. acetobutyl.", "Firmicutes"),
    "NC_003155.5": ("S. avermitilis", "Actinobacteria"),
    "NC_003450.3": ("C. glutamicum A", "Actinobacteria"),
    "NC_002677.1": ("M. leprae", "Actinobacteria"),
    "NC_008268.1": ("Rhodococcus", "Actinobacteria"),
    "NC_006958.1": ("C. glutamicum R", "Actinobacteria"),
    "NC_008818.1": ("Hyperthermus", "Archaea"),
    "NC_015948.1": ("Haloarcula", "Archaea"),
    "NC_014408.1": ("Methanobrevibacter", "Archaea"),
    "NC_019977.1": ("Methanosaeta", "Archaea"),
    "NC_007644.1": ("Moorella", "Archaea"),
}


def load_ref_exact(gff):
    r = pd.read_csv(gff, sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    return set(zip(c["s"].astype(int), c["e"].astype(int)))


def load_ref_stops(gff):
    r = pd.read_csv(gff, sep="\t", comment="#", header=None)
    return set(r[r[2] == "CDS"][4].astype(int).tolist())


def is_tp(orf, ref_exact, ref_stops):
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in ref_exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in ref_stops)


SEP = "=" * 105
print(f"\n{SEP}")
print("TRAINING SET QUALITY: All 20 holdout genomes — BASELINE vs CODON-BIAS FILTERED")
print(
    f"  Filter: genome_gc >= {FILTER_GC_THRESHOLD} -> keep ORFs with |GC3-GC12| >= {FILTER_BIAS_THRESHOLD}"
)
print(f"  TP rule: exact coordinate OR stop codon within {STOP_TOL}bp")
print(SEP)

header = (
    f"  {'Accession':<16} {'Organism':<22} {'Phylum':<16} {'GC%':>5} "
    f"{'Filtered?':>9}  "
    f"{'Base n':>7} {'Base TP%':>8}  "
    f"{'Filt n':>7} {'Filt TP%':>8} {'dTP%':>6} {'dN':>5}"
)
print(header)
print("  " + "-" * 103)

rows = []
for acc in TEST_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"  {acc}  SKIP (no fasta)")
        continue
    try:
        gff = get_gff_path(acc)
        ref_e = load_ref_exact(gff)
        ref_s = load_ref_stops(gff)
    except Exception:
        print(f"  {acc}  SKIP (no GFF)")
        continue

    label, phylum = HOLDOUT_META.get(acc, (acc, "?"))
    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    genome_gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        orfs_l = orfs.to_dict("records") if isinstance(orfs, pd.DataFrame) else list(orfs)
        base = create_training_set(sequence=seq, all_orfs=orfs_l)

    # Classify baseline
    b_tp = sum(1 for o in base if is_tp(o, ref_e, ref_s))
    b_fp = len(base) - b_tp
    b_tp_pct = 100 * b_tp / max(len(base), 1)

    # Apply adaptive filter
    apply_filter = genome_gc >= FILTER_GC_THRESHOLD
    if apply_filter:
        filt = filter_training_by_codon_position_bias(
            base, genome_gc, min_bias=FILTER_BIAS_THRESHOLD
        )
    else:
        filt = base

    f_tp = sum(1 for o in filt if is_tp(o, ref_e, ref_s))
    f_fp = len(filt) - f_tp
    f_tp_pct = 100 * f_tp / max(len(filt), 1)
    d_tp = f_tp_pct - b_tp_pct
    d_n = len(filt) - len(base)

    filter_str = "YES" if apply_filter else "no"
    d_str = f"{d_tp:+.1f}" if apply_filter else "  ---"
    dn_str = f"{d_n:+d}" if apply_filter else "  ---"

    print(
        f"  {acc:<16} {label:<22} {phylum:<16} {genome_gc*100:>5.1f} "
        f"{filter_str:>9}  "
        f"{len(base):>7} {b_tp_pct:>8.1f}%  "
        f"{len(filt):>7} {f_tp_pct:>8.1f}% {d_str:>6} {dn_str:>5}"
    )

    rows.append(
        {
            "accession": acc,
            "label": label,
            "phylum": phylum,
            "gc_pct": round(genome_gc * 100, 1),
            "filtered": apply_filter,
            "base_n": len(base),
            "base_tp": b_tp,
            "base_fp": b_fp,
            "base_tp_pct": round(b_tp_pct, 1),
            "filt_n": len(filt),
            "filt_tp": f_tp,
            "filt_fp": f_fp,
            "filt_tp_pct": round(f_tp_pct, 1),
            "d_tp_pct": round(d_tp, 1),
            "d_n": d_n,
        }
    )

df = pd.DataFrame(rows)
print(SEP)

# Summary by phylum
print("\nPHYLUM SUMMARY:")
print(
    f"  {'Phylum':<16}  {'Genomes':>8}  {'Base TP% mean':>14}  {'Filt TP% mean':>14}  {'Gain':>6}"
)
print(f"  {'-'*16}  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*6}")
for ph in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
    g = df[df["phylum"] == ph]
    print(
        f"  {ph:<16}  {len(g):>8}  {g['base_tp_pct'].mean():>13.1f}%  "
        f"{g['filt_tp_pct'].mean():>13.1f}%  "
        f"{g['filt_tp_pct'].mean() - g['base_tp_pct'].mean():>+5.1f}pp"
    )

print(
    f"\nOVERALL: base={df['base_tp_pct'].mean():.1f}% -> filtered={df['filt_tp_pct'].mean():.1f}% "
    f"(Δ={df['filt_tp_pct'].mean()-df['base_tp_pct'].mean():+.1f}pp)"
)
print(f"Genomes filtered: {df['filtered'].sum()}/{len(df)}")
print(
    f"Mean n_post for filtered genomes: {df[df['filtered']]['filt_n'].mean():.0f} "
    f"(was {df[df['filtered']]['base_n'].mean():.0f})"
)

out = OUT_DIR / "training_sets_20genomes.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
print("If TP% gains look good → wire filter into predict_genome() and run full benchmark")
print(SEP)
