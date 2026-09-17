# EXPERIMENT: Multi-feature pseudogene filter — abs_bias + Nc + length
# STATUS: active
# RESULT: pending
"""
Tests combinations of de novo features to find a filter that removes more FPs
while keeping more TPs than any single feature alone.

Logic: remove an ORF only when MULTIPLE signals agree it is a pseudogene
(AND rule = conservative = fewer TP losses).

Examples:
  Single:  abs_bias < 0.08
  Double:  abs_bias < 0.08 AND nc > 3.5
  Triple:  abs_bias < 0.08 AND nc > 3.5 AND length < 800

The AND rule means a real gene with low codon bias (low abs_bias) is KEPT
if it has low Nc (still biased in absolute terms) — reducing false TP removal.

Uses per-ORF data from tp_fp_comprehensive.csv (already classified as TP/FP).
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
CSV = OUT_DIR / "tp_fp_comprehensive.csv"

if not CSV.exists():
    print(f"ERROR: {CSV} not found. Run analyze_tp_fp_comprehensive.py first.")
    sys.exit(1)

df = pd.read_csv(CSV)

GENOMES = [
    ("NC_002677.1", "M. leprae", 57.8),
    ("NC_002929.2", "B. pertussis", 67.7),
    ("NC_008818.1", "Hyperthermus", 53.7),
    ("NC_008268.1", "Rhodococcus", 67.5),
    ("NC_003155.5", "S. avermitilis", 70.7),
    ("NC_003030.1", "C. acetobutyl.", 30.9),
    ("NC_004350.2", "S. agalactiae", 36.8),
]

SEP = "=" * 95


def evaluate_filter(tp_mask, fp_mask, n_tp, n_fp):
    """Given boolean masks of which ORFs PASS the filter, compute stats."""
    tp_kept = tp_mask.sum()
    fp_kept = fp_mask.sum()
    n_kept = tp_kept + fp_kept
    tp_ret = 100 * tp_kept / max(n_tp, 1)
    fp_rem = 100 * (n_fp - fp_kept) / max(n_fp, 1)
    tp_pct = 100 * tp_kept / max(n_kept, 1)
    ok = (tp_ret >= 85) and (fp_rem >= 40) and (n_kept >= 350)
    return tp_kept, fp_kept, n_kept, tp_ret, fp_rem, tp_pct, ok


print(f"\n{SEP}")
print("MULTI-FEATURE FILTER ANALYSIS")
print("  Strategy: remove ORF only when MULTIPLE signals agree (AND rule)")
print("  Criteria: tp_retention >= 85%, fp_removal >= 40%, n_post >= 350")
print(SEP)

# ── Per-genome grid search ─────────────────────────────────────────────────────
print("\n1. PER-GENOME: Best filter per genome (searching abs_bias x nc x length)")
print(SEP)

# Candidate thresholds
BIAS_T = [0.04, 0.06, 0.08, 0.10, 0.12]  # min abs_bias (keep if >=)
NC_T = [2.5, 3.0, 3.5, 4.0, 5.0]  # max Nc      (keep if <=)
LEN_T = [300, 500, 700, 900]  # min length  (keep if >=)

best_configs = {}
for acc, label, gc_pct in GENOMES:
    g = df[df["acc"] == acc]
    tp = g[g["cls"] == "TP"]
    fp = g[g["cls"] == "FP"]
    n_tp, n_fp = len(tp), len(fp)
    if n_tp == 0 or n_fp == 0:
        continue

    print(
        f"\n  {label} ({gc_pct:.1f}% GC)  TP={n_tp}  FP={n_fp}  base_TP%={100*n_tp/(n_tp+n_fp):.1f}%"
    )
    print(f"  {'Filter':<50} {'tp_ret':>7} {'fp_rem':>7} {'n_post':>7} {'new_TP%':>8}  Result")
    print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7} {'-'*8}  {'-'*12}")

    best = None
    best_score = 0

    # Single: abs_bias only
    for bt in BIAS_T:
        tp_m = tp["abs_bias"] >= bt
        fp_m = fp["abs_bias"] >= bt
        tpk, fpk, nk, tr, fr, tp_pct, ok = evaluate_filter(tp_m, fp_m, n_tp, n_fp)
        score = fr * tr
        if ok and score > best_score:
            best_score = score
            best = (f"abs_bias>={bt}", bt, None, None, tr, fr, nk, tp_pct)

    # Double: abs_bias AND nc
    for bt, nt in product(BIAS_T, NC_T):
        tp_m = (tp["abs_bias"] >= bt) | (tp["nc"] <= nt)  # OR = more inclusive
        fp_m = (fp["abs_bias"] >= bt) | (fp["nc"] <= nt)
        tpk, fpk, nk, tr, fr, tp_pct, ok = evaluate_filter(tp_m, fp_m, n_tp, n_fp)
        score = fr * tr
        if ok and score > best_score:
            best_score = score
            best = (f"abs_bias>={bt} OR nc<={nt}", bt, nt, None, tr, fr, nk, tp_pct)

        # AND rule (more aggressive)
        tp_m2 = (tp["abs_bias"] >= bt) & (tp["nc"] <= nt)
        fp_m2 = (fp["abs_bias"] >= bt) & (fp["nc"] <= nt)
        tpk2, fpk2, nk2, tr2, fr2, tp_pct2, ok2 = evaluate_filter(tp_m2, fp_m2, n_tp, n_fp)
        score2 = fr2 * tr2
        if ok2 and score2 > best_score:
            best_score = score2
            best = (f"abs_bias>={bt} AND nc<={nt}", bt, nt, None, tr2, fr2, nk2, tp_pct2)

    # Triple: abs_bias OR nc OR long enough
    for bt, nt, lt in product(BIAS_T, NC_T, LEN_T):
        for rule in ["OR", "AND"]:
            if rule == "OR":
                tp_m = (tp["abs_bias"] >= bt) | (tp["nc"] <= nt) | (tp["length"] >= lt)
                fp_m = (fp["abs_bias"] >= bt) | (fp["nc"] <= nt) | (fp["length"] >= lt)
            else:
                tp_m = (tp["abs_bias"] >= bt) & (tp["nc"] <= nt) | (tp["length"] >= lt)
                fp_m = (fp["abs_bias"] >= bt) & (fp["nc"] <= nt) | (fp["length"] >= lt)
            tpk, fpk, nk, tr, fr, tp_pct, ok = evaluate_filter(tp_m, fp_m, n_tp, n_fp)
            score = fr * tr
            if ok and score > best_score:
                best_score = score
                best = (
                    f"(bias>={bt} {rule} nc<={nt}) OR len>={lt}",
                    bt,
                    nt,
                    lt,
                    tr,
                    fr,
                    nk,
                    tp_pct,
                )

    if best:
        desc, bt, nt, lt, tr, fr, nk, tp_pct = best
        print(f"  BEST: {desc:<50} {tr:>7.1f}% {fr:>7.1f}% {nk:>7}  {tp_pct:>7.1f}%  PASS")
        best_configs[acc] = best
    else:
        print(f"  No combination passes all criteria for {label}")

    # Show top-3 rules
    scored = []
    for bt in BIAS_T:
        for nt in NC_T:
            for rule in ["OR", "AND"]:
                if rule == "OR":
                    tp_m = (tp["abs_bias"] >= bt) | (tp["nc"] <= nt)
                    fp_m = (fp["abs_bias"] >= bt) | (fp["nc"] <= nt)
                else:
                    tp_m = (tp["abs_bias"] >= bt) & (tp["nc"] <= nt)
                    fp_m = (fp["abs_bias"] >= bt) & (fp["nc"] <= nt)
                tpk, fpk, nk, tr, fr, tp_pct, ok = evaluate_filter(tp_m, fp_m, n_tp, n_fp)
                scored.append((fr * tr, f"abs_bias>={bt} {rule} nc<={nt}", tr, fr, nk, tp_pct, ok))
    scored.sort(reverse=True)
    for sc, desc, tr, fr, nk, tp_pct, ok in scored[:3]:
        tag = "PASS" if ok else "FAIL"
        print(f"  {desc:<50} {tr:>7.1f}% {fr:>7.1f}% {nk:>7}  {tp_pct:>7.1f}%  {tag}")

print(f"\n{SEP}")
print("2. CROSS-GENOME: Which abs_bias + nc combination passes for MOST genomes?")
print(SEP)
print(f"  {'Rule':<45} {'pass/7':>7}  Failures")
print(f"  {'-'*45} {'-'*7}  {'-'*35}")

cross_results = {}
for bt in BIAS_T:
    for nt in NC_T:
        for rule in ["OR", "AND"]:
            passes, fails = [], []
            for acc, label, gc_pct in GENOMES:
                g = df[df["acc"] == acc]
                tp = g[g["cls"] == "TP"]
                fp = g[g["cls"] == "FP"]
                if len(tp) == 0 or len(fp) == 0:
                    continue
                if rule == "OR":
                    tp_m = (tp["abs_bias"] >= bt) | (tp["nc"] <= nt)
                    fp_m = (fp["abs_bias"] >= bt) | (fp["nc"] <= nt)
                else:
                    tp_m = (tp["abs_bias"] >= bt) & (tp["nc"] <= nt)
                    fp_m = (fp["abs_bias"] >= bt) & (fp["nc"] <= nt)
                _, _, _, tr, fr, _, ok = evaluate_filter(tp_m, fp_m, len(tp), len(fp))
                (passes if ok else fails).append(label)
            key = f"abs_bias>={bt} {rule} nc<={nt}"
            cross_results[key] = (len(passes), fails)

top = sorted(cross_results.items(), key=lambda x: -x[1][0])
for rule, (n_pass, fails) in top[:10]:
    mark = "  <-- BEST" if n_pass == max(v[0] for v in cross_results.values()) else ""
    print(f"  {rule:<45} {n_pass:>3}/7   {', '.join(fails) if fails else 'none'}{mark}")

out = OUT_DIR / "multifeature_filter_results.csv"
rows = []
for acc, label, gc_pct in GENOMES:
    g = df[df["acc"] == acc]
    tp = g[g["cls"] == "TP"]
    fp = g[g["cls"] == "FP"]
    for bt in BIAS_T:
        for nt in NC_T:
            for rule in ["OR", "AND"]:
                if len(tp) == 0 or len(fp) == 0:
                    continue
                if rule == "OR":
                    tp_m = (tp["abs_bias"] >= bt) | (tp["nc"] <= nt)
                    fp_m = (fp["abs_bias"] >= bt) | (fp["nc"] <= nt)
                else:
                    tp_m = (tp["abs_bias"] >= bt) & (tp["nc"] <= nt)
                    fp_m = (fp["abs_bias"] >= bt) & (fp["nc"] <= nt)
                tpk, fpk, nk, tr, fr, tp_pct, ok = evaluate_filter(tp_m, fp_m, len(tp), len(fp))
                rows.append(
                    {
                        "acc": acc,
                        "label": label,
                        "gc_pct": gc_pct,
                        "bias_t": bt,
                        "nc_t": nt,
                        "rule": rule,
                        "tp_ret": round(tr, 1),
                        "fp_rem": round(fr, 1),
                        "n_post": nk,
                        "tp_pct": round(tp_pct, 1),
                        "pass": ok,
                    }
                )
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
