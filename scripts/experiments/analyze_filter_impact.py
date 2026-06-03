# EXPERIMENT: Filter impact analysis — all 20 holdout genomes
# STATUS: active
# RESULT: pending
"""
For each of the 20 TEST_GENOMES, shows the training set before/after
the adaptive codon-position bias filter, with focus on:
  - What was removed: were the removed ORFs actually FPs?
  - What was kept: did TP% genuinely improve?
  - Cross-reference with F1 benchmark results

Flags regressions and diagnoses root cause.
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
GC_FLOOR = 0.55
BIAS_MIN = 0.08

META = {
    "NC_002947.4": ("P. putida", "Proteobacteria", -0.12),
    "NC_002929.2": ("B. pertussis", "Proteobacteria", +1.87),
    "NC_003143.1": ("Y. pestis", "Proteobacteria", +0.08),
    "NC_003116.1": ("N. meningitidis", "Proteobacteria", -0.19),
    "NC_004757.1": ("N. europaea", "Proteobacteria", 0.00),
    "NC_008497.1": ("L. brevis", "Firmicutes", -0.40),
    "NC_004350.2": ("S. agalactiae", "Firmicutes", -0.06),
    "NC_006270.3": ("B. licheniformis", "Firmicutes", +0.34),
    "NC_006274.1": ("B. cereus", "Firmicutes", +0.29),
    "NC_003030.1": ("C. acetobutyl.", "Firmicutes", -0.03),
    "NC_003155.5": ("S. avermitilis", "Actinobacteria", +1.14),
    "NC_003450.3": ("C. glutamicum A", "Actinobacteria", +0.12),
    "NC_002677.1": ("M. leprae", "Actinobacteria", +0.02),
    "NC_008268.1": ("Rhodococcus", "Actinobacteria", +1.05),
    "NC_006958.1": ("C. glutamicum R", "Actinobacteria", +0.17),
    "NC_008818.1": ("Hyperthermus", "Archaea", +0.45),
    "NC_015948.1": ("Haloarcula", "Archaea", -0.44),
    "NC_014408.1": ("Methanobrevibacter", "Archaea", +0.28),
    "NC_019977.1": ("Methanosaeta", "Archaea", +0.12),
    "NC_007644.1": ("Moorella", "Archaea", -0.40),
}


def load_ref(acc):
    gff = get_gff_path(acc)
    r = pd.read_csv(gff, sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = set(c["e"].astype(int).tolist())
    return exact, stops


def is_tp(orf, exact, stops):
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in stops)


def classify(orfs, exact, stops):
    tp = [o for o in orfs if is_tp(o, exact, stops)]
    fp = [o for o in orfs if not is_tp(o, exact, stops)]
    return tp, fp


SEP = "=" * 110
print(f"\n{SEP}")
print("FILTER IMPACT ANALYSIS — all 20 holdout genomes")
print(f"  Filter: gc >= {GC_FLOOR} -> |GC3-GC12| >= {BIAS_MIN}")
print(f"  dF1 = new benchmark F1 - baseline F1 (entry #4)")
print(SEP)
print(
    f"  {'Genome':<22} {'GC%':>5} {'dF1':>6}  "
    f"{'Base n':>7} {'TP%':>5} {'FP%':>5}  "
    f"{'Filt n':>7} {'TP%':>5} {'FP%':>5}  "
    f"{'Removed TP':>11} {'Removed FP':>11}  Verdict"
)
print("  " + "-" * 108)

rows = []
for acc in TEST_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    label, phylum, df1 = META.get(acc, (acc, "?", 0))
    try:
        exact, stops = load_ref(acc)
    except Exception:
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    genome_gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        orfs_l = orfs.to_dict("records") if hasattr(orfs, "to_dict") else list(orfs)
        base = create_training_set(sequence=seq, all_orfs=orfs_l)

    filtered_flag = genome_gc >= GC_FLOOR
    if filtered_flag:
        filt = filter_training_by_codon_position_bias(base, genome_gc, min_bias=BIAS_MIN)
    else:
        filt = base

    b_tp, b_fp = classify(base, exact, stops)
    f_tp, f_fp = classify(filt, exact, stops)

    # What was removed?
    filt_ids = {id(o) for o in filt}
    removed = [o for o in base if id(o) not in filt_ids]
    r_tp, r_fp = classify(removed, exact, stops)

    b_tp_pct = 100 * len(b_tp) / max(len(base), 1)
    b_fp_pct = 100 * len(b_fp) / max(len(base), 1)
    f_tp_pct = 100 * len(f_tp) / max(len(filt), 1)
    f_fp_pct = 100 * len(f_fp) / max(len(filt), 1)

    # Diagnosis
    verdict = ""
    if not filtered_flag:
        verdict = "not filtered"
    elif df1 < -0.30:
        if len(r_tp) > len(r_fp):
            verdict = "REMOVED MORE TPs THAN FPs"
        elif b_tp_pct > 83 and (f_tp_pct - b_tp_pct) < 4:
            verdict = "already clean, small gain, quantity cost"
        else:
            verdict = "regressed - investigate"
    elif df1 > 0:
        verdict = "improved"
    else:
        verdict = "neutral / noise"

    flag = " <--" if df1 < -0.30 else ""
    print(
        f"  {label:<22} {genome_gc*100:>5.1f} {df1:>+6.2f}  "
        f"{len(base):>7} {b_tp_pct:>5.1f} {b_fp_pct:>5.1f}  "
        f"{len(filt):>7} {f_tp_pct:>5.1f} {f_fp_pct:>5.1f}  "
        f"{len(r_tp):>11} {len(r_fp):>11}  {verdict}{flag}"
    )

    rows.append(
        {
            "acc": acc,
            "label": label,
            "phylum": phylum,
            "gc_pct": round(genome_gc * 100, 1),
            "filtered": filtered_flag,
            "df1": df1,
            "base_n": len(base),
            "base_tp_n": len(b_tp),
            "base_fp_n": len(b_fp),
            "base_tp_pct": round(b_tp_pct, 1),
            "filt_n": len(filt),
            "filt_tp_n": len(f_tp),
            "filt_fp_n": len(f_fp),
            "filt_tp_pct": round(f_tp_pct, 1),
            "removed_n": len(removed),
            "removed_tp": len(r_tp),
            "removed_fp": len(r_fp),
            "removed_tp_pct": (
                round(100 * len(r_tp) / max(len(removed), 1), 1) if removed else float("nan")
            ),
        }
    )

df = pd.DataFrame(rows)

print(f"\n{SEP}")
print("REMOVED ORFs ANALYSIS: What did the filter actually remove?")
print(f"  For regressing genomes: were the removed ORFs mostly TPs (bad) or FPs (good)?")
print(SEP)
reg = df[(df["filtered"]) & (df["df1"] < -0.20)]
good = df[(df["filtered"]) & (df["df1"] > 0.50)]

print(f"\n  Regressing filtered genomes (dF1 < -0.20):")
for _, r in reg.iterrows():
    pct = r["removed_tp_pct"]
    print(
        f"    {r['label']:<22}  removed {r['removed_n']:>3} ORFs: "
        f"{r['removed_tp']:>3} TP ({pct:.0f}%) + {r['removed_fp']:>3} FP ({100-pct:.0f}%)"
        f"  <- {'PROBLEM: removed too many TPs!' if pct > 50 else 'FPs removed correctly'}"
    )

print(f"\n  Well-improving filtered genomes (dF1 > +0.50):")
for _, r in good.iterrows():
    pct = r["removed_tp_pct"]
    print(
        f"    {r['label']:<22}  removed {r['removed_n']:>3} ORFs: "
        f"{r['removed_tp']:>3} TP ({pct:.0f}%) + {r['removed_fp']:>3} FP ({100-pct:.0f}%)"
        f"  -> FP removal: {100-pct:.0f}%"
    )

print(f"\n{SEP}")
print("ROOT CAUSE: The filter removes too many TPs from already-clean training sets")
print("  High-GC real genes have a RANGE of codon bias -- some real genes have lower abs_bias")
print("  When base TP% is already >83%, removing low-bias TPs hurts more than it helps")
print(f"\n  FIX OPTIONS:")
print(f"  1. Raise GC floor (e.g. 0.62 instead of 0.55) -- skips borderline genomes")
print(f"  2. Adaptive: only filter when base training set has median abs_bias < 0.10")
print(f"  3. Only filter when estimated base TP% < 82% (high FP contamination)")
print(SEP)

out = OUT_DIR / "filter_impact_analysis.csv"
df.to_csv(out, index=False)
print(f"Saved: {out}")
