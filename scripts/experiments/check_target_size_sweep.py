"""
Sweep glimmer_max_size / flexible_target_size across all 20 holdout genomes.
Tests targets 2000 / 4000 / 6000 / 8000, each WITH and WITHOUT pass-2 top-50% filter.

Goal: find the combination that maximises TP% while maintaining or increasing
      n_training vs the current baseline (target=2000, no pass-2).

Run from repo root:
    python scripts/experiments/check_target_size_sweep.py
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
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_training_adaptive,
    find_orfs_candidates,
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
STOP_TOL = 3
MIN_P2 = 150  # minimum training size after pass-2 filter

TARGETS = [2000, 4000, 6000, 8000]


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
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


SEP = "=" * 105
print(f"\n{SEP}")
print("TARGET SIZE SWEEP — all 20 holdout genomes, with and without pass-2 top-50%")
print(f"  Targets tested : {TARGETS}")
print(f"  For each target: (A) no pass-2  (B) pass-2 top-50%")
print(SEP)

all_rows = []

for acc in TEST_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    try:
        exact, stops = load_ref(acc)
    except Exception:
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    print(f"\n  {acc}  gc={gc*100:.1f}%", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)

    print(
        f"    {'Target':>7}  {'no-p2: n':>9} {'TP%':>6} {'FP':>5}  |  "
        f"{'p2 top50%: n':>13} {'TP%':>6} {'FP':>5}"
    )
    print(f"    {'-------':>7}  {'-'*9} {'-'*6} {'-'*5}  |  {'-'*13} {'-'*6} {'-'*5}")

    base_n = None  # track target=2000 no-p2 size as baseline

    for target in TARGETS:
        with contextlib.redirect_stdout(io.StringIO()):
            training_raw = create_training_set(
                sequence=seq,
                all_orfs=orfs,
                glimmer_max_size=target,
                flexible_target_size=target,
            )
            training_filt = filter_training_adaptive(list(training_raw), gc)

        # ── No pass-2 ────────────────────────────────────────────────────────
        labels_a = np.array([int(is_tp(o, exact, stops)) for o in training_filt], dtype=int)
        n_a = len(labels_a)
        tp_a = int(labels_a.sum())
        fp_a = n_a - tp_a
        tpf_a = tp_a / n_a if n_a > 0 else 0.0

        if target == 2000:
            base_n = n_a

        # ── Pass-2 top-50% ───────────────────────────────────────────────────
        with contextlib.redirect_stdout(io.StringIO()):
            intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
            models1 = build_all_scoring_models(training_filt, intergenic)
            tdf = pd.DataFrame(training_filt)
            scored = score_all_orfs(tdf, models1)

        if "combined_score" in scored.columns and n_a > MIN_P2 * 2:
            thresh = float(np.percentile(scored["combined_score"].values, 50))
            keep = scored["combined_score"].values >= thresh
            n_b = int(keep.sum())
            if n_b >= MIN_P2:
                labels_b = labels_a[keep]
                tp_b = int(labels_b.sum())
                fp_b = n_b - tp_b
                tpf_b = tp_b / n_b
            else:
                n_b, tp_b, fp_b, tpf_b = n_a, tp_a, fp_a, tpf_a
        else:
            n_b, tp_b, fp_b, tpf_b = n_a, tp_a, fp_a, tpf_a

        flag = ""
        if target > 2000 and n_a >= base_n and tpf_a > (labels_a[:base_n].sum() / max(base_n, 1)):
            flag = " +"  # bigger AND cleaner than baseline

        print(
            f"    {target:>7}  {n_a:>9} {tpf_a*100:>5.1f}% {fp_a:>5}  |  "
            f"{n_b:>13} {tpf_b*100:>5.1f}% {fp_b:>5}{flag}"
        )

        all_rows.append(
            {
                "acc": acc,
                "gc_pct": round(gc * 100, 1),
                "target": target,
                "n_no_p2": n_a,
                "tp_no_p2": tp_a,
                "fp_no_p2": fp_a,
                "tpfrac_no_p2": round(tpf_a, 3),
                "n_p2": n_b,
                "tp_p2": tp_b,
                "fp_p2": fp_b,
                "tpfrac_p2": round(tpf_b, 3),
            }
        )

df = pd.DataFrame(all_rows)

print(f"\n{SEP}")
print("CROSS-GENOME SUMMARY: mean n_training and mean TP% across all 20 genomes")
print(SEP)
print(
    f"\n  {'Target':>7}  {'no-p2: mean_n':>14} {'mean_TP%':>9} {'mean_FP':>8}  |  "
    f"{'p2: mean_n':>11} {'mean_TP%':>9} {'mean_FP':>8}"
)
print(f"  {'-------':>7}  {'-'*14} {'-'*9} {'-'*8}  |  {'-'*11} {'-'*9} {'-'*8}")
for target in TARGETS:
    sub = df[df["target"] == target]
    flag = " <-- baseline" if target == 2000 else ""
    print(
        f"  {target:>7}  {sub['n_no_p2'].mean():>14.0f} {sub['tpfrac_no_p2'].mean()*100:>8.1f}%"
        f" {sub['fp_no_p2'].mean():>8.1f}  |  "
        f"{sub['n_p2'].mean():>11.0f} {sub['tpfrac_p2'].mean()*100:>8.1f}%"
        f" {sub['fp_p2'].mean():>8.1f}{flag}"
    )

print(f"\n{SEP}")
print("VERDICT: Best config per genome (max TP% with n >= baseline n)")
print(SEP)
base = df[df["target"] == 2000][["acc", "n_no_p2", "tpfrac_no_p2"]].rename(
    columns={"n_no_p2": "base_n", "tpfrac_no_p2": "base_tp"}
)
df2 = df.merge(base, on="acc")

# Consider all 8 variants (4 targets x 2 filter options)
winners = []
for acc in df2["acc"].unique():
    sub = df2[df2["acc"] == acc]
    base_n = sub["base_n"].iloc[0]
    base_tp = sub["base_tp"].iloc[0]
    best_tp = 0.0
    best_conf = None
    for _, r in sub.iterrows():
        for variant, n, tp in [
            ("no-p2", r["n_no_p2"], r["tpfrac_no_p2"]),
            ("p2-50%", r["n_p2"], r["tpfrac_p2"]),
        ]:
            if n >= base_n * 0.80 and tp > best_tp:  # allow slight shrink (80%)
                best_tp = tp
                best_conf = (r["target"], variant, int(n), tp, base_n, base_tp)
    if best_conf:
        target, variant, n, tp, bn, btp = best_conf
        delta = (tp - btp) * 100
        winners.append(
            {
                "acc": acc,
                "best_target": target,
                "variant": variant,
                "n": n,
                "tp_pct": round(tp * 100, 1),
                "base_n": bn,
                "base_tp_pct": round(btp * 100, 1),
                "delta_tp_pp": round(delta, 1),
            }
        )

wdf = pd.DataFrame(winners)
print(
    f"\n  {'Acc':<15} {'best_target':>12} {'variant':>8} {'n':>6} {'base_n':>7} "
    f"{'TP%':>6} {'base_TP%':>9} {'delta':>7}"
)
print(f"  {'-'*15} {'-'*12} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*9} {'-'*7}")
for _, r in wdf.sort_values("delta_tp_pp", ascending=False).iterrows():
    print(
        f"  {r['acc']:<15} {r['best_target']:>12} {r['variant']:>8} {r['n']:>6} "
        f"{r['base_n']:>7} {r['tp_pct']:>5.1f}% {r['base_tp_pct']:>8.1f}% {r['delta_tp_pp']:>+6.1f}pp"
    )

out = Path(__file__).parent.parent.parent / "lgb_attribution_results" / "target_size_sweep.csv"
df.to_csv(out, index=False)
wdf.to_csv(out.with_name("target_size_winners.csv"), index=False)
print(f"\nSaved: {out}")
print(SEP)
