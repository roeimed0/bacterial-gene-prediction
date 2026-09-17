# EXPERIMENT: Downstream coding leakage — detect gene fragments in training set
# STATUS: concluded
# RESULT: REJECTED. Corrupted initial model (47-66% FP training) cannot detect
#   coding signal downstream. All downstream scores are negative for both TP and
#   FP classes. 0/6 genomes show significant signal. Circular dependency: need
#   clean model to detect coding, need clean training to build clean model.
"""
For each training ORF, score the region DOWNSTREAM of its stop codon (in the
same reading frame) using the initial IMM/codon model built from the training set.

Hypothesis:
  - Gene-fragment FPs (ORFs stopping inside a real gene at a premature stop):
    the downstream region IS the real gene → should score positively with the
    coding model, even with a corrupted initial model.

  - Real TPs (complete genes):
    the downstream region is non-coding (next gene, intergenic, or different frame)
    → downstream coding score should be near zero or negative.

  - Intergenic FPs (random ORFs):
    downstream is also random → neutral or negative score.

Signal: downstream_imm_score > 0 → gene fragment → FP candidate

This is a targeted iterative approach: we score NOT the ORF itself (which gene
fragments score high, causing iterative training to fail) but the region AFTER
the stop — which only gene fragments will show as coding.

Run from repo root:
    python scripts/experiments/analyze_downstream_leakage.py
"""

import bisect
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    find_orfs_candidates,
    score_codon_bias_ratio,
    score_imm_ratio,
)

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3
DOWNSTREAM_BP = 120  # score next N bp after stop (= 40 codons)

FOCUS_GENOMES = [
    ("NC_002677.1", "M. leprae", "Actinobacteria", 41.74),
    ("NC_002929.2", "B. pertussis", "Proteobacteria", 60.06),
    ("NC_008818.1", "Hyperthermus", "Archaea", 60.88),
    ("NC_008268.1", "Rhodococcus", "Actinobacteria", 65.34),
    ("NC_003155.5", "S. avermitilis", "Actinobacteria", 66.34),
    ("NC_003030.1", "C. acetobutyl.", "Firmicutes", 88.01),
]

_RC = str.maketrans("ACGT", "TGCA")


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = sorted(set(c["e"].astype(int).tolist()))
    return exact, stops


def is_tp(orf, exact, stops):
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in stops)


def nearest_stop_dist(orf, stops):
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    if gs > ge:
        gs, ge = ge, gs
    i = bisect.bisect_left(stops, ge)
    best = 999999
    for j in [max(0, i - 1), i, min(len(stops) - 1, i + 1)]:
        best = min(best, abs(ge - stops[j]))
    return best


def get_downstream_seq(orf, genome_seq, n_bp=DOWNSTREAM_BP):
    """Get N bp of sequence downstream of ORF's stop, in the same reading frame."""
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    strand = orf.get("strand", "forward")
    if gs > ge:
        gs, ge = ge, gs

    if strand == "forward":
        # Stop codon ends at ge (exclusive in genome coords); downstream = ge+1 onwards
        start = ge  # ge is 1-based exclusive end, so downstream starts at ge
        end = min(start + n_bp, len(genome_seq))
        return genome_seq[start:end].upper()
    else:
        # Reverse strand: stop codon is at gs-1 (upstream in genome = downstream in gene)
        end = gs - 1
        start = max(0, end - n_bp)
        raw = genome_seq[start:end].upper()
        return raw.translate(_RC)[::-1]


SEP = "=" * 90
print(f"\n{SEP}")
print("DOWNSTREAM CODING LEAKAGE — gene fragment detection in training set")
print(f"  Score {DOWNSTREAM_BP}bp downstream of each training ORF's stop (same reading frame)")
print(f"  Gene fragments: downstream should look CODING (continuing real gene)")
print(f"  Complete genes: downstream should look NON-CODING")
print(SEP)

all_rows = []

for acc, label, phylum, f1 in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    try:
        exact, ref_stops = load_ref(acc)
    except Exception:
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    print(f"\n  {acc}  {label:<20}  [{phylum}]  F1={f1:.1f}%  gc={gc*100:.1f}%")

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        orfs_l = orfs.to_dict("records") if hasattr(orfs, "to_dict") else list(orfs)
        training = create_training_set(sequence=seq, all_orfs=orfs_l)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs_l)
        models = build_all_scoring_models(training, intergenic)

    # Score each training ORF's downstream region
    tp_down, fp_frag_down, fp_other_down = [], [], []

    for orf in training:
        tp = is_tp(orf, exact, ref_stops)
        stop_d = nearest_stop_dist(orf, ref_stops)
        down_seq = get_downstream_seq(orf, seq, DOWNSTREAM_BP)

        if len(down_seq) < 9:
            continue

        # IMM score of downstream region (coding vs noncoding)
        imm_score = score_imm_ratio(
            down_seq,
            models["coding_imm"],
            models["noncoding_imm"],
            models["max_order"],
        )
        # Codon score of downstream region
        codon_score = score_codon_bias_ratio(
            down_seq,
            models["codon_model"],
            models["background_codon_model"],
        )
        combined = (imm_score + codon_score) / 2

        row = {
            "acc": acc,
            "label": label,
            "phylum": phylum,
            "f1": f1,
            "cls": "TP" if tp else "FP",
            "stop_dist": stop_d,
            "fp_type": (
                "TP"
                if tp
                else (
                    "near"
                    if stop_d < 30
                    else "frag" if stop_d < 100 else "neighbor" if stop_d < 300 else "intergenic"
                )
            ),
            "imm_down": imm_score,
            "codon_down": codon_score,
            "combined_down": combined,
        }
        all_rows.append(row)

        if tp:
            tp_down.append(combined)
        elif stop_d < 100:  # gene fragment FPs
            fp_frag_down.append(combined)
        else:
            fp_other_down.append(combined)

    print(
        f"    n_train={len(training)}  TP={sum(1 for o in training if is_tp(o, exact, ref_stops))}"
    )

    if tp_down and fp_frag_down:
        _, p1 = mannwhitneyu(tp_down, fp_frag_down, alternative="two-sided")
        sep1 = np.median(fp_frag_down) - np.median(tp_down)
        print(f"    TP downstream:        median={np.median(tp_down):+.4f}")
        print(
            f"    FP-fragment downstream: median={np.median(fp_frag_down):+.4f}  "
            f"sep={sep1:+.4f}  p={p1:.4f}  "
            f"{'SIGNAL - fragments score MORE CODING than TPs!' if sep1>0.01 and p1<0.05 else 'weak'}"
        )

    if tp_down and fp_other_down:
        _, p2 = mannwhitneyu(tp_down, fp_other_down, alternative="two-sided")
        sep2 = np.median(fp_other_down) - np.median(tp_down)
        print(
            f"    FP-other downstream:    median={np.median(fp_other_down):+.4f}  "
            f"sep={sep2:+.4f}  p={p2:.4f}  "
            f"{'SIGNAL' if abs(sep2)>0.01 and p2<0.05 else 'weak'}"
        )

    # Threshold sweep: remove training ORFs where downstream looks too coding
    # (to catch gene fragments)
    if fp_frag_down and tp_down:
        print(f"\n    Threshold sweep (remove if combined_down > T — catches gene fragments):")
        print(f"    {'T':>8} {'tp_ret':>8} {'frag_rem':>10} {'other_rem':>11}  Decision")
        thresholds = [np.percentile(tp_down, p) for p in [50, 60, 70, 75, 80, 85, 90]]
        for T in sorted(set(round(t, 3) for t in thresholds)):
            tp_k = sum(1 for v in tp_down if v <= T)
            frag_k = sum(1 for v in fp_frag_down if v <= T)
            oth_k = sum(1 for v in fp_other_down if v <= T)
            tr = 100 * tp_k / max(len(tp_down), 1)
            fr = 100 * (len(fp_frag_down) - frag_k) / max(len(fp_frag_down), 1)
            or_ = 100 * (len(fp_other_down) - oth_k) / max(len(fp_other_down), 1)
            ok = (tr >= 80) and (fr >= 30)
            print(
                f"    {T:>8.4f} {tr:>8.1f}% {fr:>10.1f}% {or_:>11.1f}%  "
                f"{'PASS' if ok else 'partial' if tr>=80 else 'FAIL'}"
            )

df = pd.DataFrame(all_rows)
out = OUT_DIR / "downstream_leakage_diagnostic.csv"
df.to_csv(out, index=False)
print(f"\n{SEP}")
print(f"CROSS-GENOME SUMMARY:")
for ftype in ["frag", "neighbor", "intergenic"]:
    n_sig = 0
    for acc, label, _, _ in FOCUS_GENOMES:
        g = df[df["acc"] == acc]
        tp = g[g["cls"] == "TP"]["combined_down"].values
        fp = g[g["fp_type"] == ftype]["combined_down"].values
        if len(tp) < 2 or len(fp) < 2:
            continue
        _, p = mannwhitneyu(tp, fp, alternative="two-sided")
        sep = np.median(fp) - np.median(tp)
        if sep > 0.01 and p < 0.05:
            n_sig += 1
    print(f"  FP-{ftype:<12} downstream > TP: {n_sig}/{len(FOCUS_GENOMES)} genomes significant")
print(f"Saved: {out}")
print(SEP)
