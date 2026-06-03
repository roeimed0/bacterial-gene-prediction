# EXPERIMENT: RBS score distribution of TP vs FP training ORFs
# STATUS: concluded
# RESULT: REJECTED. rbs_score is bimodal (either -5.0 or >8.0). Threshold sweep
#   from -4 to +3 gives identical results at every step — all training ORFs
#   (TP and FP alike) have rbs_score > 3.0. Max FP removal: 4-11%, need 40%.
#   Root cause: Glimmer+Flexible already selects for purine-rich upstream ORFs,
#   so FPs look like TPs by all sequence-local RBS criteria. Fix requires
#   iterative training (Glimmer-style multi-pass), not a filter.
"""
For each focus genome, classifies every training ORF as TP or FP using the
reference GFF, then measures the rbs_score distribution for each class.

Hypothesis: FP training ORFs have systematically lower rbs_score than TPs.
If confirmed, a rbs_score threshold filter on the training set will:
  - Remove most FPs (improving model quality)
  - Retain most TPs (preserving model coverage)
  - Not make the small-genome training set problem worse

Decision criteria (all must hold at the chosen threshold T):
  A. tp_retention(T) >= 0.80  — keep at least 80% of TPs
  B. fp_removal(T)   >= 0.40  — remove at least 40% of FPs
  C. n_training_post(T) >= 400 — don't starve small-genome models

Threshold sweep: T in {-5, -4, -3, -2, -1, 0, 1, 2, 3}

Run from repo root:
    python scripts/experiments/analyze_training_rbs_quality.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    create_training_set,
    find_orfs_candidates,
)

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

STOP_TOLERANCE = 3  # bp tolerance on stop codon for TP match

FOCUS_GENOMES = [
    # Problem cases — low F1, contaminated training sets
    ("NC_002677.1", "M. leprae", "Actinobacteria", 41.74, 66.2),
    ("NC_002929.2", "B. pertussis", "Proteobacteria", 60.06, 47.3),
    ("NC_008818.1", "Hyperthermus", "Archaea", 60.88, 77.4),
    ("NC_008268.1", "Rhodococcus", "Actinobacteria", 65.34, 57.4),
    ("NC_003155.5", "S. avermitilis", "Actinobacteria", 66.34, 47.9),
    # Reference cases — high F1, clean training sets
    ("NC_003030.1", "C. acetobutyl.", "Firmicutes", 88.01, 91.0),
    ("NC_004350.2", "S. agalactiae", "Firmicutes", 87.08, 91.1),
]

THRESHOLDS = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def load_reference_stops(gff_path: str) -> set:
    """Load reference CDS stop coordinates for TP matching (stop is reliable)."""
    ref = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
    cds = ref[ref[2] == "CDS"][[3, 4]].rename(columns={3: "start", 4: "end"})
    return set(cds["end"].astype(int).tolist())


def load_reference_exact(gff_path: str) -> set:
    """Load reference CDS as (start, end) pairs for exact matching."""
    ref = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
    cds = ref[ref[2] == "CDS"][[3, 4]].rename(columns={3: "start", 4: "end"})
    cds = cds.drop_duplicates()
    return set(zip(cds["start"].astype(int), cds["end"].astype(int)))


def classify_training_orf(orf: dict, ref_exact: set, ref_stops: set) -> str:
    """Return 'TP' if orf matches a reference CDS, 'FP' otherwise."""
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in ref_exact:
        return "TP"
    # Stop codon match within tolerance
    for stop in ref_stops:
        if abs(ge - stop) <= STOP_TOLERANCE:
            return "TP"
    return "FP"


# ── Main ──────────────────────────────────────────────────────────────────────

SEP = "=" * 95
print(f"\n{SEP}")
print("DIAGNOSTIC T3-A: RBS score — TP vs FP distribution in training set")
print(f"  Hypothesis: FP training ORFs have lower rbs_score than TP training ORFs")
print(f"  Decision: threshold T acceptable if tp_retention>=80%, fp_removal>=40%, n_post>=400")
print(SEP)

all_orfs_data = []  # per-ORF rows for threshold sweep
genome_summaries = []

for acc, label, phylum, f1_pct, known_tp_pct in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"  SKIP {acc}: no fasta")
        continue
    try:
        gff_path = get_gff_path(acc)
        ref_exact = load_reference_exact(gff_path)
        ref_stops = load_reference_stops(gff_path)
    except Exception:
        print(f"  SKIP {acc}: no reference GFF")
        continue

    print(f"\n  {acc}  {label:<20}  [{phylum}]  F1={f1_pct:.1f}%  known_TP%={known_tp_pct:.1f}%")

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        orfs_list = orfs.to_dict("records") if isinstance(orfs, pd.DataFrame) else list(orfs)
        training = create_training_set(sequence=seq, all_orfs=orfs_list)

    # Classify each training ORF
    tp_rbs, fp_rbs = [], []
    for orf in training:
        label_tp = classify_training_orf(orf, ref_exact, ref_stops)
        rbs = float(orf.get("rbs_score", -5.0))
        if label_tp == "TP":
            tp_rbs.append(rbs)
        else:
            fp_rbs.append(rbs)
        all_orfs_data.append(
            {
                "accession": acc,
                "genome_label": label,
                "phylum": phylum,
                "f1_pct": f1_pct,
                "classification": label_tp,
                "rbs_score": rbs,
            }
        )

    n_tp, n_fp = len(tp_rbs), len(fp_rbs)
    n_total = n_tp + n_fp
    tp_pct = 100 * n_tp / max(n_total, 1)

    print(
        f"    Training: {n_total} total | TP={n_tp} ({tp_pct:.1f}%) | FP={n_fp} ({100-tp_pct:.1f}%)"
    )
    if tp_rbs:
        print(
            f"    TP  rbs: median={np.median(tp_rbs):+.2f}  mean={np.mean(tp_rbs):+.2f}  "
            f"min={min(tp_rbs):+.2f}  max={max(tp_rbs):+.2f}"
        )
    if fp_rbs:
        print(
            f"    FP  rbs: median={np.median(fp_rbs):+.2f}  mean={np.mean(fp_rbs):+.2f}  "
            f"min={min(fp_rbs):+.2f}  max={max(fp_rbs):+.2f}"
        )
    if tp_rbs and fp_rbs:
        separation = np.median(tp_rbs) - np.median(fp_rbs)
        print(
            f"    Median separation (TP - FP): {separation:+.2f}  "
            f"{'GOOD signal' if separation > 1.0 else 'WEAK signal' if separation > 0 else 'NO signal'}"
        )

    # Threshold sweep for this genome
    print(f"\n    Threshold sweep (rbs_score >= T):")
    print(
        f"    {'T':>6} {'n_kept':>7} {'tp_kept':>8} {'fp_kept':>8} "
        f"{'tp_ret%':>8} {'fp_rem%':>8}  Decision"
    )
    print(f"    {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")
    for T in THRESHOLDS:
        tp_kept = sum(1 for r in tp_rbs if r >= T)
        fp_kept = sum(1 for r in fp_rbs if r >= T)
        n_kept = tp_kept + fp_kept
        tp_ret = 100 * tp_kept / max(n_tp, 1)
        fp_rem = 100 * (n_fp - fp_kept) / max(n_fp, 1)
        ok = []
        if tp_ret >= 80:
            ok.append("A")
        if fp_rem >= 40:
            ok.append("B")
        if n_kept >= 400:
            ok.append("C")
        verdict = (
            "PASS (A+B+C)" if len(ok) == 3 else (f"PARTIAL ({'+'.join(ok)})" if ok else "FAIL")
        )
        print(
            f"    {T:>6.1f} {n_kept:>7} {tp_kept:>8} {fp_kept:>8} "
            f"{tp_ret:>8.1f}% {fp_rem:>8.1f}%  {verdict}"
        )

    genome_summaries.append(
        {
            "accession": acc,
            "label": label,
            "phylum": phylum,
            "f1_pct": f1_pct,
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_total": n_total,
            "tp_pct": round(tp_pct, 1),
            "tp_median_rbs": round(np.median(tp_rbs), 3) if tp_rbs else float("nan"),
            "fp_median_rbs": round(np.median(fp_rbs), 3) if fp_rbs else float("nan"),
            "separation": (
                round(np.median(tp_rbs) - np.median(fp_rbs), 3)
                if tp_rbs and fp_rbs
                else float("nan")
            ),
        }
    )

# ── Cross-genome threshold summary ────────────────────────────────────────────

print(f"\n{SEP}")
print("CROSS-GENOME THRESHOLD ANALYSIS: Does any threshold pass A+B+C for ALL genomes?")
print(SEP)

df_orfs = pd.DataFrame(all_orfs_data)
print(f"  {'T':>6}  {'Genomes passing A+B+C':>22}  Failures")
print(f"  {'-'*6}  {'-'*22}  {'-'*40}")

best_threshold = None
for T in THRESHOLDS:
    passes, fails = [], []
    for row in genome_summaries:
        acc = row["accession"]
        genome_orfs = df_orfs[df_orfs["accession"] == acc]
        tp_orfs = genome_orfs[genome_orfs["classification"] == "TP"]["rbs_score"]
        fp_orfs = genome_orfs[genome_orfs["classification"] == "FP"]["rbs_score"]
        n_tp = len(tp_orfs)
        n_fp = len(fp_orfs)
        tp_kept = (tp_orfs >= T).sum()
        fp_kept = (fp_orfs >= T).sum()
        n_kept = tp_kept + fp_kept
        tp_ret = 100 * tp_kept / max(n_tp, 1)
        fp_rem = 100 * (n_fp - fp_kept) / max(n_fp, 1)
        ok = (tp_ret >= 80) and (fp_rem >= 40) and (n_kept >= 400)
        (passes if ok else fails).append(row["label"])
    n_pass = len(passes)
    n_total_g = len(genome_summaries)
    fail_str = ", ".join(fails) if fails else "none"
    mark = "  <-- CANDIDATE" if n_pass == n_total_g else ""
    print(f"  {T:>6.1f}  {n_pass:>2}/{n_total_g} pass               {fail_str}{mark}")
    if n_pass == n_total_g and best_threshold is None:
        best_threshold = T

print(f"\n{SEP}")
print("SUMMARY: TP vs FP RBS medians per genome")
print(SEP)
print(
    f"  {'Genome':<20} {'F1%':>5} {'TP%':>5} {'TP med RBS':>11} {'FP med RBS':>11} "
    f"{'Separation':>11}  Signal"
)
print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*11} {'-'*11} {'-'*11}  {'-'*12}")
for row in genome_summaries:
    signal = (
        "STRONG"
        if row["separation"] > 2
        else "MODERATE" if row["separation"] > 1 else "WEAK" if row["separation"] > 0 else "NONE"
    )
    print(
        f"  {row['label']:<20} {row['f1_pct']:>5.1f} {row['tp_pct']:>5.1f} "
        f"{row['tp_median_rbs']:>11.3f} {row['fp_median_rbs']:>11.3f} "
        f"{row['separation']:>11.3f}  {signal}"
    )

if best_threshold is not None:
    print(f"\n  RECOMMENDED THRESHOLD: rbs_score >= {best_threshold:.1f}")
    print(
        f"  All {len(genome_summaries)} genomes pass criteria A (tp_ret>=80%), "
        f"B (fp_rem>=40%), C (n_post>=400)"
    )
else:
    print(f"\n  No single threshold passes A+B+C for ALL genomes.")
    print(f"  Consider: adaptive threshold per genome, or only apply to genomes with n_train > 600")

out_path = OUT_DIR / "training_rbs_diagnostic.csv"
df_orfs.to_csv(out_path, index=False)
print(f"\nSaved per-ORF data: {out_path}")
print(SEP)
