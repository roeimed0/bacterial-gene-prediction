# EXPERIMENT: Comprehensive TP vs FP metric analysis — pseudogene discrimination
# STATUS: active
# RESULT: pending
"""
For each focus genome, classifies every training ORF as TP or FP using the
reference GFF, then compares ALL available de novo metrics between the two groups.

Goal: find which combination of metrics best identifies pseudogenes and IS elements
contaminating the training set, across ALL genome types (not just high-GC).

Metrics tested (all computable from ORF sequence + genome alone — no trained model):
  From ORF dict:   rbs_score, rbs_spacing, start_codon, length, strand
  From sequence:   gc1, gc2, gc3, abs_bias=|GC3-GC12|, gc_orf, Nc (eff.num.codons),
                   upstream_in_frame_stops, upstream_gc

Decision criteria for a filter to be useful:
  A. tp_retention >= 80%   (keep most TPs)
  B. fp_removal   >= 40%   (remove substantial FPs)
  C. n_post       >= 350   (don't starve small-genome models)

Run from repo root:
    python scripts/experiments/analyze_tp_fp_comprehensive.py
"""

import contextlib
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    _gc_position_bias,
    create_training_set,
    find_orfs_candidates,
)

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

STOP_TOL = 3

FOCUS_GENOMES = [
    ("NC_002677.1", "M. leprae", "Actinobacteria", 41.74, 66.2),
    ("NC_002929.2", "B. pertussis", "Proteobacteria", 60.06, 47.3),
    ("NC_008818.1", "Hyperthermus", "Archaea", 60.88, 77.4),
    ("NC_008268.1", "Rhodococcus", "Actinobacteria", 65.34, 57.4),
    ("NC_003155.5", "S. avermitilis", "Actinobacteria", 66.34, 47.9),
    ("NC_003030.1", "C. acetobutyl.", "Firmicutes", 88.01, 91.0),
    ("NC_004350.2", "S. agalactiae", "Firmicutes", 87.08, 91.1),
]


# ── Metric computation helpers ─────────────────────────────────────────────────


def _nc(seq: str) -> float:
    """Effective number of codons (Nc / ENC). Lower = more biased = more likely coding."""
    if len(seq) < 9:
        return 61.0  # max = uniform usage
    codons = [seq[i : i + 3] for i in range(0, len(seq) - 2, 3) if len(seq[i : i + 3]) == 3]
    # Group synonymous codons by amino acid (simplified: by 1st+2nd position)
    family_counts: dict = {}
    for c in codons:
        key = c[:2]
        family_counts.setdefault(key, Counter())
        family_counts[key][c] += 1
    # Nc = sum over families: n_aa / chi_aa  (Wright 1990 simplified)
    nc_sum = 0.0
    n_families = 0
    for fam, cnt in family_counts.items():
        n = sum(cnt.values())
        if n < 2:
            continue
        p = np.array(list(cnt.values()), dtype=float) / n
        chi = sum(p**2)
        if chi > 0:
            nc_sum += 1.0 / chi
            n_families += 1
    return nc_sum / max(n_families, 1)


def _upstream_stops(genome_seq: str, orf: dict, window: int = 300) -> int:
    """Count in-frame stop codons in the upstream window (same frame as ORF start)."""
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    strand = orf.get("strand", "forward")
    if gs > ge:
        gs, ge = ge, gs

    if strand == "forward":
        region = genome_seq[max(0, gs - window - 1) : gs - 1].upper()
    else:
        _RC = str.maketrans("ACGT", "TGCA")
        raw = genome_seq[ge : min(len(genome_seq), ge + window)].upper()
        region = raw.translate(_RC)[::-1]

    stops = {"TAA", "TAG", "TGA"}
    count = 0
    for i in range(0, len(region) - 2, 3):
        if region[i : i + 3] in stops:
            count += 1
    return count


def _upstream_gc(genome_seq: str, orf: dict, window: int = 25) -> float:
    """GC% of the upstream window (same region RBS is scored from)."""
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    strand = orf.get("strand", "forward")
    if gs > ge:
        gs, ge = ge, gs

    if strand == "forward":
        region = genome_seq[max(0, gs - window - 1) : gs - 1].upper()
    else:
        _RC = str.maketrans("ACGT", "TGCA")
        raw = genome_seq[ge : min(len(genome_seq), ge + window)].upper()
        region = raw.translate(_RC)[::-1]

    if not region:
        return 0.5
    return (region.count("G") + region.count("C")) / len(region)


def compute_metrics(orf: dict, genome_seq: str) -> dict:
    """Compute all de novo metrics for a single training ORF."""
    seq = orf.get("sequence", "")
    gc1, gc2, gc3, abs_bias = _gc_position_bias(seq) if seq else (0, 0, 0, 0)
    gc_orf = (seq.count("G") + seq.count("C")) / max(len(seq), 1) if seq else 0.5
    nc = _nc(seq) if seq else 61.0

    return {
        # From ORF dict
        "rbs_score": float(orf.get("rbs_score", -5.0)),
        "rbs_spacing": float(orf.get("rbs_spacing", 0)),
        "length": float(orf.get("length", 0)),
        "is_atg": int(orf.get("start_codon", "ATG") == "ATG"),
        "is_forward": int(orf.get("strand", "forward") == "forward"),
        # From sequence
        "gc1": gc1,
        "gc2": gc2,
        "gc3": gc3,
        "gc12": (gc1 + gc2) / 2,
        "abs_bias": abs_bias,  # |GC3 - GC12|
        "gc_orf": gc_orf,
        "nc": nc,  # lower = more biased = more likely coding
        "upstream_stops": _upstream_stops(genome_seq, orf),
        "upstream_gc": _upstream_gc(genome_seq, orf),
    }


def load_ref_exact(gff):
    r = pd.read_csv(gff, sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    return set(zip(c["s"].astype(int), c["e"].astype(int)))


def load_ref_stops(gff):
    r = pd.read_csv(gff, sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[4]].astype(int)
    return set(c[4].tolist())


def is_tp(orf, ref_exact, ref_stops):
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in ref_exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in ref_stops)


# ── Main ──────────────────────────────────────────────────────────────────────

SEP = "=" * 100
METRIC_COLS = [
    "rbs_score",
    "rbs_spacing",
    "length",
    "is_atg",
    "is_forward",
    "gc1",
    "gc2",
    "gc3",
    "gc12",
    "abs_bias",
    "gc_orf",
    "nc",
    "upstream_stops",
    "upstream_gc",
]

all_rows = []

print(f"\n{SEP}")
print("COMPREHENSIVE TP vs FP METRIC ANALYSIS — Pseudogene discrimination")
print(f"  Classifies training ORFs as TP/FP using reference GFF")
print(f"  Tests {len(METRIC_COLS)} de novo metrics for pseudogene signal")
print(SEP)

for acc, label, phylum, f1, known_tp in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    try:
        gff = get_gff_path(acc)
        ref_e = load_ref_exact(gff)
        ref_s = load_ref_stops(gff)
    except Exception:
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    genome_gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        orfs_l = orfs.to_dict("records") if isinstance(orfs, pd.DataFrame) else list(orfs)
        training = create_training_set(sequence=seq, all_orfs=orfs_l)

    print(
        f"\n  {acc}  {label:<20}  GC={genome_gc*100:.1f}%  F1={f1:.1f}%  known_TP={known_tp:.1f}%"
    )

    tp_metrics = {m: [] for m in METRIC_COLS}
    fp_metrics = {m: [] for m in METRIC_COLS}

    for orf in training:
        m = compute_metrics(orf, seq)
        target = tp_metrics if is_tp(orf, ref_e, ref_s) else fp_metrics
        for k, v in m.items():
            target[k].append(v)
        all_rows.append(
            {
                "acc": acc,
                "label": label,
                "phylum": phylum,
                "f1": f1,
                "genome_gc": round(genome_gc, 4),
                "cls": "TP" if is_tp(orf, ref_e, ref_s) else "FP",
                **m,
            }
        )

    n_tp = len(tp_metrics["length"])
    n_fp = len(fp_metrics["length"])
    n_tot = n_tp + n_fp
    print(f"    Total: {n_tot}  TP: {n_tp} ({100*n_tp/max(n_tot,1):.0f}%)  FP: {n_fp}")

    # Rank metrics by separation
    separations = []
    for m in METRIC_COLS:
        tv = np.array(tp_metrics[m])
        fv = np.array(fp_metrics[m])
        if len(tv) < 2 or len(fv) < 2 or tv.std() + fv.std() < 1e-9:
            separations.append((m, 0.0, float("nan"), float("nan"), float("nan"), ""))
            continue
        med_sep = float(np.median(tv) - np.median(fv))
        try:
            _, p = mannwhitneyu(tv, fv, alternative="two-sided")
        except Exception:
            p = 1.0
        signal = (
            "STRONG"
            if abs(med_sep) > 0.05 and p < 0.01
            else "MODERATE" if abs(med_sep) > 0.02 and p < 0.05 else "WEAK"
        )
        separations.append((m, med_sep, np.median(tv), np.median(fv), p, signal))

    separations.sort(key=lambda x: -abs(x[1]))
    print(
        f"\n    {'Metric':<20} {'TP median':>11} {'FP median':>11} {'Separation':>12} {'p-value':>10}  Signal"
    )
    print(f"    {'-'*20} {'-'*11} {'-'*11} {'-'*12} {'-'*10}  {'-'*10}")
    for m, sep, tp_med, fp_med, p, sig in separations:
        pstr = f"{p:.4f}" if not np.isnan(p) else "  n/a"
        print(f"    {m:<20} {tp_med:>11.4f} {fp_med:>11.4f} {sep:>+12.4f} {pstr:>10}  {sig}")

print(f"\n{SEP}")
print("CROSS-GENOME RANKING: Which metrics are consistently discriminative?")
print(f"  Counting genomes where |separation| > threshold AND p < 0.05")
print(SEP)

df = pd.DataFrame(all_rows)

# For each metric: compute separation per genome and count "significant" ones
metric_scores = []
for m in METRIC_COLS:
    n_sig = 0
    mean_abs_sep = 0.0
    for acc, label, phylum, f1, _ in FOCUS_GENOMES:
        g = df[df["acc"] == acc]
        if len(g) < 5:
            continue
        tv = g[g["cls"] == "TP"][m].values
        fv = g[g["cls"] == "FP"][m].values
        if len(tv) < 2 or len(fv) < 2:
            continue
        sep = abs(float(np.median(tv) - np.median(fv)))
        mean_abs_sep += sep
        try:
            _, p = mannwhitneyu(tv, fv, alternative="two-sided")
        except Exception:
            p = 1.0
        if sep > 0.02 and p < 0.05:
            n_sig += 1
    n_genomes = sum(1 for acc, _, _, _, _ in FOCUS_GENOMES if len(df[df["acc"] == acc]) >= 5)
    metric_scores.append((m, n_sig, mean_abs_sep / max(n_genomes, 1)))

metric_scores.sort(key=lambda x: (-x[1], -x[2]))
print(f"\n  {'Metric':<20} {'n_sig/7':>8} {'mean|sep|':>11}  Rank")
print(f"  {'-'*20} {'-'*8} {'-'*11}  {'-'*20}")
for rank, (m, n_sig, mean_sep) in enumerate(metric_scores, 1):
    star = "  <-- TOP SIGNAL" if rank <= 3 else ""
    print(f"  {m:<20} {n_sig:>3}/{len(FOCUS_GENOMES):<4} {mean_sep:>11.4f}  #{rank}{star}")

print(f"\n{SEP}")
print("THRESHOLD SWEEP: Top-3 metrics individually and in combination")
print(f"  A: tp_ret>=80%  B: fp_rem>=40%  C: n_post>=350")
print(SEP)

top3 = [m for m, _, _ in metric_scores[:3]]
print(f"  Top 3 metrics: {top3}")

for acc, label, phylum, f1, _ in FOCUS_GENOMES:
    g = df[df["acc"] == acc]
    if len(g) < 5:
        continue
    tp = g[g["cls"] == "TP"]
    fp = g[g["cls"] == "FP"]
    n_tp, n_fp = len(tp), len(fp)
    print(f"\n  {acc}  {label}  (TP={n_tp}, FP={n_fp})")

    for m in top3:
        tv = tp[m].values
        fv = fp[m].values
        direction = "ge" if np.median(tv) > np.median(fv) else "le"
        thresholds = np.percentile(np.concatenate([tv, fv]), [10, 20, 30, 40, 50, 60, 70, 80])
        best_verdict, best_T = "FAIL", None
        for T in thresholds:
            if direction == "ge":
                tp_k = (tv >= T).sum()
                fp_k = (fv >= T).sum()
            else:
                tp_k = (tv <= T).sum()
                fp_k = (fv <= T).sum()
            n_k = tp_k + fp_k
            tp_r = 100 * tp_k / max(n_tp, 1)
            fp_r = 100 * (n_fp - fp_k) / max(n_fp, 1)
            ok = []
            if tp_r >= 80:
                ok.append("A")
            if fp_r >= 40:
                ok.append("B")
            if n_k >= 350:
                ok.append("C")
            v = "PASS" if len(ok) == 3 else (f"PARTIAL({'+'.join(ok)})" if ok else "FAIL")
            if len(ok) > len(best_verdict.split("+")):
                best_verdict = v
                best_T = T
        t_str = f"{best_T:.4f}" if best_T is not None else "none"
        print(f"    {m:<20}  best_T={t_str}  {best_verdict}")

out = OUT_DIR / "tp_fp_comprehensive.csv"
df.to_csv(out, index=False)
print(f"\nSaved per-ORF data: {out}")
print(SEP)
print("NEXT STEPS:")
print("  1. Design filter using top-ranked metric(s)")
print("  2. Wire into create_training_set() / predict_genome() with adaptive thresholds")
print("  3. Run full 20-genome benchmark")
print(SEP)
