# EXPERIMENT: GC3 codon position bias as training set pre-filter
# STATUS: active
# RESULT: pending
"""
Tests whether |GC3 - GC12| (codon position GC asymmetry) discriminates
TP from FP training ORFs better than RBS score.

Real genes: translational selection creates GC3 != GC12.
Spurious ORFs: GC3 ~= GC12 ~= genome_GC (no selection pressure).

Run from repo root:
    python scripts/experiments/analyze_training_codon_bias.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import _gc_position_bias, create_training_set, find_orfs_candidates

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3

FOCUS_GENOMES = [
    ("NC_002677.1", "M. leprae", 41.74),
    ("NC_002929.2", "B. pertussis", 60.06),
    ("NC_008818.1", "Hyperthermus", 60.88),
    ("NC_008268.1", "Rhodococcus", 65.34),
    ("NC_003155.5", "S. avermitilis", 66.34),
    ("NC_003030.1", "C. acetobutyl.", 88.01),
    ("NC_004350.2", "S. agalactiae", 87.08),
]

BIAS_THRESHOLDS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


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


SEP = "=" * 90
print(f"\n{SEP}")
print("DIAGNOSTIC: Codon-position GC bias (|GC3-GC12|) — TP vs FP discrimination")
print(SEP)

rows = []

for acc, label, f1 in FOCUS_GENOMES:
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

    tp_bias, fp_bias = [], []
    for orf in training:
        s = orf.get("sequence", "")
        if not s:
            continue
        _, _, _, bias = _gc_position_bias(s)
        (tp_bias if is_tp(orf, ref_e, ref_s) else fp_bias).append(bias)

    n_tp, n_fp = len(tp_bias), len(fp_bias)
    n_tot = n_tp + n_fp
    print(
        f"\n  {acc}  {label:<20}  GC={genome_gc*100:.1f}%  F1={f1:.1f}%  "
        f"TP={n_tp} ({100*n_tp/max(n_tot,1):.0f}%)  FP={n_fp}"
    )
    if tp_bias and fp_bias:
        sep_med = np.median(tp_bias) - np.median(fp_bias)
        print(f"    TP bias: median={np.median(tp_bias):.4f}  mean={np.mean(tp_bias):.4f}")
        print(f"    FP bias: median={np.median(fp_bias):.4f}  mean={np.mean(fp_bias):.4f}")
        print(
            f"    Separation (TP-FP medians): {sep_med:+.4f}  "
            f"{'GOOD' if sep_med>0.02 else 'WEAK' if sep_med>0 else 'NONE'}"
        )

    print(f"\n    Threshold sweep  (tp_retention A>=80%, fp_removal B>=40%, n_post C>=400):")
    print(f"    {'bias>=':>8} {'n_post':>7} {'tp_ret%':>8} {'fp_rem%':>8}  Decision")
    print(f"    {'-'*8} {'-'*7} {'-'*8} {'-'*8}  {'-'*20}")
    for T in BIAS_THRESHOLDS:
        tp_k = sum(1 for b in tp_bias if b >= T)
        fp_k = sum(1 for b in fp_bias if b >= T)
        n_k = tp_k + fp_k
        tp_r = 100 * tp_k / max(n_tp, 1)
        fp_r = 100 * (n_fp - fp_k) / max(n_fp, 1)
        ok = []
        if tp_r >= 80:
            ok.append("A")
        if fp_r >= 40:
            ok.append("B")
        if n_k >= 400:
            ok.append("C")
        verdict = (
            "PASS (A+B+C)" if len(ok) == 3 else (f"PARTIAL ({'+'.join(ok)})" if ok else "FAIL")
        )
        print(f"    {T:>8.2f} {n_k:>7} {tp_r:>8.1f}% {fp_r:>8.1f}%  {verdict}")
        rows.append(
            {
                "acc": acc,
                "label": label,
                "f1": f1,
                "genome_gc": round(genome_gc, 4),
                "threshold": T,
                "n_post": n_k,
                "tp_ret": round(tp_r, 2),
                "fp_rem": round(fp_r, 2),
                "n_tp": n_tp,
                "n_fp": n_fp,
            }
        )

print(f"\n{SEP}")
print("CROSS-GENOME: best threshold passing A+B+C for most genomes")
print(SEP)
df = pd.DataFrame(rows)
print(f"  {'bias>=':>8}  {'pass A+B+C':>12}  Failures")
print(f"  {'-'*8}  {'-'*12}  {'-'*40}")
for T in BIAS_THRESHOLDS:
    g = df[df["threshold"] == T]
    pass_mask = (g["tp_ret"] >= 80) & (g["fp_rem"] >= 40) & (g["n_post"] >= 400)
    n_pass = pass_mask.sum()
    fails = g[~pass_mask]["label"].tolist()
    mark = "  <-- CANDIDATE" if n_pass == len(FOCUS_GENOMES) else ""
    print(
        f"  {T:>8.2f}  {n_pass:>3}/{len(FOCUS_GENOMES)} pass     "
        f"{', '.join(fails) if fails else 'none'}{mark}"
    )

out = OUT_DIR / "training_codon_bias_diagnostic.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
print("If a threshold passes A+B+C for all genomes -> set min_codon_position_bias=T")
print("in create_training_set() call in predict_genome() and run full benchmark.")
print(SEP)
