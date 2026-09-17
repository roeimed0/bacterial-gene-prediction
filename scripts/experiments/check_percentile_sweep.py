"""
At target=6000, sweep pass-2 percentile cuts from keep-top-10% to keep-top-90%
across all 20 holdout genomes.  Finds the cut that maximises TP% while keeping
n_training closest to the current baseline (target=2000, no pass-2).

Run from repo root:
    python scripts/experiments/check_percentile_sweep.py
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
TARGET = 6000
KEEP_PCTS = list(range(10, 100, 10))  # 10, 20, 30, ... 90


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


SEP = "=" * 115
print(f"\n{SEP}")
print(f"PERCENTILE SWEEP at target={TARGET}: keep top X% by combined score")
print(f"  Percentiles: {KEEP_PCTS}")
print(f"  Baseline: target=2000, no pass-2")
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
        # Baseline: target=2000, no p2
        t2000 = create_training_set(
            sequence=seq, all_orfs=orfs, glimmer_max_size=2000, flexible_target_size=2000
        )
        t2000 = filter_training_adaptive(list(t2000), gc)
        # Expanded: target=6000
        t6000 = create_training_set(
            sequence=seq, all_orfs=orfs, glimmer_max_size=TARGET, flexible_target_size=TARGET
        )
        t6000 = filter_training_adaptive(list(t6000), gc)
        # Build pass-1 model from t6000
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models1 = build_all_scoring_models(t6000, intergenic)
        tdf6 = pd.DataFrame(t6000)
        scored6 = score_all_orfs(tdf6, models1)

    base_labels = np.array([int(is_tp(o, exact, stops)) for o in t2000], dtype=int)
    base_n = len(base_labels)
    base_tp = int(base_labels.sum())
    base_tpf = base_tp / base_n if base_n > 0 else 0.0

    full_labels = np.array([int(is_tp(o, exact, stops)) for o in t6000], dtype=int)
    scores = (
        scored6["combined_score"].values
        if "combined_score" in scored6.columns
        else np.zeros(len(t6000))
    )

    print(
        f"    baseline(t=2000): n={base_n}  TP={base_tp} ({base_tpf*100:.1f}%)  FP={base_n-base_tp}"
    )
    print(
        f"    expanded(t=6000): n={len(t6000)}  TP={full_labels.sum()} ({full_labels.mean()*100:.1f}%)  "
        f"FP={int((1-full_labels).sum())}"
    )
    print(
        f"    {'keep%':>6} {'n':>6} {'TP':>6} {'FP':>5} {'TP%':>6} {'vs_base_n':>10} {'vs_base_TP%':>12}"
    )
    print(
        f"    {'-----':>6} {'--':>6} {'--':>6} {'--':>5} {'---':>6} {'---------':>10} {'-----------':>12}"
    )

    for keep_pct in KEEP_PCTS:
        drop_pct = 100 - keep_pct
        thresh = float(np.percentile(scores, drop_pct))  # discard bottom drop_pct%
        keep = scores >= thresh
        n_after = int(keep.sum())
        if n_after < 50:
            continue
        labs_after = full_labels[keep]
        tp_after = int(labs_after.sum())
        fp_after = n_after - tp_after
        tpf_after = tp_after / n_after
        n_delta = n_after - base_n
        tp_delta = (tpf_after - base_tpf) * 100

        flag = ""
        if n_after >= base_n * 0.80 and tpf_after > base_tpf:
            flag = " VIABLE"
        if n_after >= base_n and tpf_after > base_tpf:
            flag = " BIGGER+CLEANER"

        print(
            f"    {keep_pct:>5}%  {n_after:>6} {tp_after:>6} {fp_after:>5} {tpf_after*100:>5.1f}%"
            f" {n_delta:>+10} {tp_delta:>+11.1f}pp{flag}"
        )

        all_rows.append(
            {
                "acc": acc,
                "gc_pct": round(gc * 100, 1),
                "base_n": base_n,
                "base_tpf": round(base_tpf, 3),
                "keep_pct": keep_pct,
                "n": n_after,
                "tp": tp_after,
                "fp": fp_after,
                "tpf": round(tpf_after, 3),
                "n_delta": n_after - base_n,
                "tp_delta_pp": round((tpf_after - base_tpf) * 100, 2),
                "viable": n_after >= base_n * 0.80 and tpf_after > base_tpf,
            }
        )

df = pd.DataFrame(all_rows)

print(f"\n{SEP}")
print("CROSS-GENOME SUMMARY: mean n and mean TP% by keep% (target=6000)")
print(SEP)
print(
    f"  {'keep%':>6} {'mean_n':>8} {'vs_base':>8} {'mean_TP%':>10} {'mean_FP':>9} {'n_viable':>9}"
)
print(
    f"  {'-----':>6} {'------':>8} {'-------':>8} {'--------':>10} {'-------':>9} {'--------':>9}"
)
base_mean_n = df[df["keep_pct"] == KEEP_PCTS[0]]["base_n"].mean()
for pct in KEEP_PCTS:
    sub = df[df["keep_pct"] == pct]
    if len(sub) == 0:
        continue
    n_viable = sub["viable"].sum()
    print(
        f"  {pct:>5}%  {sub['n'].mean():>8.0f} {sub['n'].mean()-base_mean_n:>+8.0f}"
        f" {sub['tpf'].mean()*100:>9.1f}% {sub['fp'].mean():>9.1f} {n_viable:>9}/{len(sub)}"
    )

print(
    f"\n  Baseline (t=2000, no-p2): mean_n={base_mean_n:.0f}  mean_TP%={df[df['keep_pct']==10]['base_tpf'].mean()*100:.1f}%"
)

print(f"\n{SEP}")
print("BEST PERCENTILE per genome (max TP% with n >= 80% of baseline)")
print(SEP)
viable = df[df["viable"]]
if len(viable) > 0:
    best = viable.loc[viable.groupby("acc")["tpf"].idxmax()]
    print(
        f"  {'Acc':<15} {'gc%':>5} {'best_keep%':>10} {'n':>6} {'base_n':>7} "
        f"{'TP%':>6} {'base_TP%':>9} {'delta':>8}"
    )
    print(f"  {'-'*15} {'-'*5} {'-'*10} {'-'*6} {'-'*7} {'-'*6} {'-'*9} {'-'*8}")
    for _, r in best.sort_values("tp_delta_pp", ascending=False).iterrows():
        print(
            f"  {r['acc']:<15} {r['gc_pct']:>5.1f} {r['keep_pct']:>9}%  {r['n']:>6} "
            f"{r['base_n']:>7} {r['tpf']*100:>5.1f}% {r['base_tpf']*100:>8.1f}% {r['tp_delta_pp']:>+7.1f}pp"
        )
else:
    print("  No viable configurations found (all shrink below 80% of baseline)")

print(f"\n{SEP}")
print("OPTIMAL GLOBAL PERCENTILE: which keep% is best across all genomes?")
print(SEP)
for pct in KEEP_PCTS:
    sub = df[df["keep_pct"] == pct]
    n_viable = sub["viable"].sum()
    mean_tp_del = sub["tp_delta_pp"].mean()
    mean_n_del = sub["n_delta"].mean()
    print(
        f"  keep {pct:>2}%:  viable={n_viable:>2}/{len(sub)}  "
        f"mean_n_delta={mean_n_del:>+7.0f}  mean_tp_delta={mean_tp_del:>+6.2f}pp"
    )

out = Path(__file__).parent.parent.parent / "lgb_attribution_results" / "percentile_sweep_t6000.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
