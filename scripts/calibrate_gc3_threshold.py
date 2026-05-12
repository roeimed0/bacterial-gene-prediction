"""
Calibrate FIRST_FILTER_THRESHOLD['combined_threshold'] after adding GC3 score.

Uses a diverse set of GENOME_CATALOG genomes (never TEST_GENOMES) covering the
GC range.  For each candidate threshold, reports sensitivity and precision at
the first filter so we can find the value that:
  - improves sensitivity for high-GC organisms
  - does not regress precision or sensitivity for low-GC organisms
  - becomes the new FIRST_FILTER_THRESHOLD['combined_threshold']

Run from repo root:
    python scripts/calibrate_gc3_threshold.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, TEST_GENOMES
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
TEST_SET = set(TEST_GENOMES)

# Diverse calibration genomes from GENOME_CATALOG — spanning the GC range.
# NOT in TEST_GENOMES.  ~2 per GC tier to see high-GC gain and low-GC safety.
CALIBRATION_GENOMES = {
    # High-GC Actinobacteria (target group — where GC3 should help most)
    "NC_003888.3": ("Streptomyces coelicolor A3(2)", "HIGH-GC"),  # 72%
    "NC_000962.3": ("Mycobacterium tuberculosis H37Rv", "HIGH-GC"),  # 65%
    "NC_009664.1": ("Kineococcus radiotolerans", "HIGH-GC"),  # 74%
    # Medium-GC Proteobacteria
    "NC_000913.3": ("Escherichia coli K-12 MG1655", "MED-GC"),  # 51%
    "NC_002516.2": ("Pseudomonas aeruginosa PAO1", "MED-GC"),  # 66%
    "NC_002505.1": ("Vibrio cholerae", "MED-GC"),  # 47%
    # Low-GC Firmicutes (no regression here)
    "NC_000964.3": ("Bacillus subtilis 168", "LOW-GC"),  # 43%
    "NC_003210.1": ("Listeria monocytogenes EGD-e", "LOW-GC"),  # 38%
}

# Verify none are in TEST_GENOMES
for acc in CALIBRATION_GENOMES:
    assert acc not in TEST_SET, f"{acc} is in TEST_GENOMES — remove it!"

THRESHOLDS = [0.10, 0.15, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35]
CURRENT = 0.26

print("Calibrating combined_threshold with GC3 score added")
print(f"Current threshold: {CURRENT}")
print(f"Calibration genomes: {len(CALIBRATION_GENOMES)}\n")

results = []  # list of (acc, gc_tier, threshold, has_gc3, sens, prec, tp, n_ref)

for acc, (name, gc_tier) in CALIBRATION_GENOMES.items():
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"SKIP {acc} — fasta not found")
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc_pct = (seq.count("G") + seq.count("C")) / len(seq)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)

    ref = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    ref_set = set(zip(ref[ref[2] == "CDS"][3].astype(int), ref[ref[2] == "CDS"][4].astype(int)))
    n_ref = len(ref_set)

    # Score WITHOUT gc3: monkey-patch _gc3_content to zero
    import src.traditional_methods as tm
    from src.traditional_methods import _gc3_content as _gc3_real

    tm._gc3_content = lambda s: 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        scored_no_gc3 = score_all_orfs(orfs.copy(), models)
    tm._gc3_content = _gc3_real

    # Score WITH gc3
    with contextlib.redirect_stdout(io.StringIO()):
        scored_gc3 = score_all_orfs(orfs.copy(), models)

    print(f"{acc}  {name:<40}  GC={gc_pct:.0%}  [{gc_tier}]  n_ref={n_ref}")

    for threshold in THRESHOLDS:
        thresh_params = dict(FIRST_FILTER_THRESHOLD)
        thresh_params["combined_threshold"] = threshold

        for gc3_on, scored in [(False, scored_no_gc3), (True, scored_gc3)]:
            filtered = filter_candidates(scored, **thresh_params)
            fids = set(
                zip(filtered["genome_start"].astype(int), filtered["genome_end"].astype(int))
            )
            tp = sum(1 for r in ref_set if r in fids)
            fp = len(fids) - tp
            sens = 100 * tp / max(n_ref, 1)
            prec = 100 * tp / max(len(fids), 1)
            results.append(
                {
                    "acc": acc,
                    "name": name,
                    "gc_tier": gc_tier,
                    "threshold": threshold,
                    "gc3": gc3_on,
                    "sens": sens,
                    "prec": prec,
                    "tp": tp,
                    "n_ref": n_ref,
                }
            )

# ── Summary: mean sens/prec per threshold per tier ─────────────────────────
print("\n" + "=" * 80)
print("SUMMARY — mean sensitivity / precision at filter1 by GC tier and threshold")
print("=" * 80)
print(
    f"\n{'Threshold':>10}  {'gc3':>5}  "
    f"{'HIGH-GC sens':>13}  {'HIGH-GC prec':>13}  "
    f"{'LOW-GC sens':>12}  {'LOW-GC prec':>12}  "
    f"{'MED-GC sens':>12}  {'MED-GC prec':>12}"
)
print("-" * 100)

import collections

for threshold in THRESHOLDS:
    for gc3_on in [False, True]:
        rows = [r for r in results if r["threshold"] == threshold and r["gc3"] == gc3_on]
        by_tier = collections.defaultdict(list)
        for r in rows:
            by_tier[r["gc_tier"]].append(r)

        def avg(tier, key):
            g = by_tier.get(tier, [])
            return f"{np.mean([r[key] for r in g]):.1f}%" if g else "  n/a"

        flag = ""
        hi_sens_gc3 = np.mean([r["sens"] for r in by_tier.get("HIGH-GC", [])])
        lo_prec_gc3 = np.mean([r["prec"] for r in by_tier.get("LOW-GC", [])])
        if gc3_on and threshold != CURRENT:
            no_gc3_rows = [
                r
                for r in results
                if r["threshold"] == threshold and not r["gc3"] and r["gc_tier"] == "HIGH-GC"
            ]
            if no_gc3_rows:
                hi_sens_no = np.mean([r["sens"] for r in no_gc3_rows])
                if hi_sens_gc3 > hi_sens_no + 1.0 and lo_prec_gc3 >= 85.0:
                    flag = " << CANDIDATE"

        print(
            f"  t={threshold:.2f}    {'YES' if gc3_on else 'no ':>5}  "
            f"{avg('HIGH-GC','sens'):>13}  {avg('HIGH-GC','prec'):>13}  "
            f"{avg('LOW-GC','sens'):>12}  {avg('LOW-GC','prec'):>12}  "
            f"{avg('MED-GC','sens'):>12}  {avg('MED-GC','prec'):>12}{flag}"
        )
    if threshold == CURRENT:
        print(f"  {'  ^ current ^':^100}")

# Best candidate
print("\n" + "=" * 80)
candidates = []
for threshold in THRESHOLDS:
    gc3_rows = [r for r in results if r["threshold"] == threshold and r["gc3"]]
    base_rows = [r for r in results if r["threshold"] == CURRENT and not r["gc3"]]
    if not gc3_rows or not base_rows:
        continue
    hi_gc3 = [r for r in gc3_rows if r["gc_tier"] == "HIGH-GC"]
    lo_gc3 = [r for r in gc3_rows if r["gc_tier"] == "LOW-GC"]
    hi_base = [r for r in base_rows if r["gc_tier"] == "HIGH-GC"]
    lo_base = [r for r in base_rows if r["gc_tier"] == "LOW-GC"]
    if not hi_gc3 or not lo_gc3:
        continue
    delta_sens_hi = np.mean([r["sens"] for r in hi_gc3]) - np.mean([r["sens"] for r in hi_base])
    delta_prec_lo = np.mean([r["prec"] for r in lo_gc3]) - np.mean([r["prec"] for r in lo_base])
    candidates.append((threshold, delta_sens_hi, delta_prec_lo))
    print(
        f"  t={threshold:.2f}  HIGH-GC sens delta={delta_sens_hi:+.1f}pp  LOW-GC prec delta={delta_prec_lo:+.1f}pp"
    )

# Pick best: max HIGH-GC sensitivity gain with LOW-GC prec regression < 2pp
good = [(t, ds, dp) for t, ds, dp in candidates if dp >= -2.0 and ds > 0]
if good:
    best = max(good, key=lambda x: x[1])
    print(f"\nRECOMMENDED combined_threshold: {best[0]}")
    print(f"  HIGH-GC sensitivity gain: {best[1]:+.1f}pp")
    print(f"  LOW-GC  precision change: {best[2]:+.1f}pp")
else:
    print("\nNo threshold strictly improves HIGH-GC without LOW-GC regression.")
