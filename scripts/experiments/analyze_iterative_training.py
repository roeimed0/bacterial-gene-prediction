# EXPERIMENT: Iterative self-training — pass-2 contamination purification
# STATUS: active
# RESULT: pending
"""
Tests whether a 2-pass training strategy cleans contaminated training sets.

ROOT CAUSE (confirmed):
  Problem genomes have 47-66% FP in their training sets.
  Glimmer+Flexible picks pseudogenes / IS elements because they look geometrically
  identical to real genes (long, non-overlapping). The codon/IMM models trained
  on this garbage then score real genes poorly -> low sensitivity.

GATING ANALOGY (same pattern as genome_gc_high):
  Feature gating:       max(0, feature - floor)       -- active only when reliable
  Training set gating:  if contaminated -> pass-2     -- activate only when needed
  Gate signal:          frac_low = fraction of training ORFs where the pass-1
                        model gives a negative score (combined_score < threshold)

PASS-2 FILTER LOGIC:
  1. Build pass-1 model from geometric training set (current pipeline)
  2. Score all training ORFs with pass-1 model
  3. Keep only top-P percentile training ORFs by combined score
  4. Rebuild model from this cleaner set (pass-2)
  5. Gate: only fire pass-2 if frac_low_before > gate_threshold (e.g., 0.20)

WHAT THIS SCRIPT MEASURES (per genome):
  - n_training_before, tp_before, fp_before, tp_frac_before
  - frac_negative (fraction of training ORFs scoring below 0 by pass-1 model)
  - For each percentile cut (25, 50, 75):
      n_after, tp_after, fp_after, tp_frac_after
  - Combined: does pass-2 improve TP fraction meaningfully?

SUCCESS CRITERION:
  tp_frac_after >= tp_frac_before + 0.10 on problem genomes
  tp_frac_before unchanged (or better) on clean genomes (no unnecessary filtering)

Run from repo root:
    python scripts/experiments/analyze_iterative_training.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, START_SELECTION_WEIGHTS, TEST_GENOMES
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    filter_training_adaptive,
    find_orfs_candidates,
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

STOP_TOL = 3
GATE_THRESHOLD = 0.20  # fire pass-2 if >20% of training scores negative
PERCENTILES = [25, 50, 75]  # pass-2 filter aggressiveness options


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = set(c["e"].astype(int).tolist())
    return exact, stops


def is_tp_orf(orf, exact, stops):
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in stops)


SEP = "=" * 100
print(f"\n{SEP}")
print("DIAGNOSTIC: Iterative self-training -- pass-2 contamination purification")
print(f"  Gate: if frac_negative > {GATE_THRESHOLD} -> trigger pass-2 filter")
print(f"  Percentile cuts tested: {PERCENTILES}")
print(f"  Success: tp_frac_after >= tp_frac_before + 0.10 on problem genomes")
print(SEP)

rows = []

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

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training_raw = create_training_set(sequence=seq, all_orfs=orfs)
        training = filter_training_adaptive(list(training_raw), gc)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models_1 = build_all_scoring_models(training, intergenic)
        # Score the training ORFs with pass-1 model — convert list to DataFrame first
        training_df = pd.DataFrame(training) if not isinstance(training, pd.DataFrame) else training
        scored_training = score_all_orfs(training_df, models_1)

    training_list = training_df.to_dict("records")

    # Label training ORFs as TP/FP
    labels = np.array([int(is_tp_orf(o, exact, stops)) for o in training_list], dtype=int)
    n_total = len(labels)
    n_tp_before = int(labels.sum())
    n_fp_before = n_total - n_tp_before
    tp_frac_before = n_tp_before / n_total if n_total > 0 else 0.0

    # combined_score from pass-1 model
    if hasattr(scored_training, "to_dict"):
        scores = (
            scored_training["combined_score"].values
            if "combined_score" in scored_training.columns
            else np.zeros(n_total)
        )
    else:
        scores = np.zeros(n_total)

    frac_negative = float((scores < 0).sum()) / max(n_total, 1)
    gate_fires = frac_negative > GATE_THRESHOLD

    row = {
        "acc": acc,
        "gc_pct": round(gc * 100, 1),
        "n_training": n_total,
        "n_tp_before": n_tp_before,
        "n_fp_before": n_fp_before,
        "tp_frac_before": round(tp_frac_before, 3),
        "frac_negative": round(frac_negative, 3),
        "gate_fires": gate_fires,
    }

    print(
        f"\n  {acc}  gc={gc*100:.1f}%  n={n_total}  "
        f"TP={n_tp_before} ({tp_frac_before*100:.1f}%)  FP={n_fp_before}  "
        f"frac_neg={frac_negative:.3f}  gate={'FIRES' if gate_fires else 'closed'}"
    )

    # Pass-2 filter at each percentile
    for pct in PERCENTILES:
        thresh = float(np.percentile(scores, 100 - pct))  # keep TOP pct%
        keep_mask = scores >= thresh
        n_after = int(keep_mask.sum())
        if n_after == 0:
            n_after = n_total
            keep_mask = np.ones(n_total, dtype=bool)

        labels_after = labels[keep_mask]
        n_tp_after = int(labels_after.sum())
        n_fp_after = n_after - n_tp_after
        tp_frac_after = n_tp_after / n_after if n_after > 0 else 0.0
        delta = tp_frac_after - tp_frac_before

        row[f"n_after_p{pct}"] = n_after
        row[f"tp_after_p{pct}"] = n_tp_after
        row[f"fp_after_p{pct}"] = n_fp_after
        row[f"tp_frac_after_p{pct}"] = round(tp_frac_after, 3)
        row[f"delta_p{pct}"] = round(delta, 3)

        verdict = "GOOD" if delta > 0.10 else ("ok" if delta > 0.05 else "weak")
        print(
            f"    top-{pct}%: n={n_after:>5}  TP={n_tp_after} ({tp_frac_after*100:.1f}%)  "
            f"FP={n_fp_after}  delta={delta:+.3f}  {verdict}"
        )

    rows.append(row)

df = pd.DataFrame(rows)

print(f"\n{SEP}")
print("SUMMARY TABLE: tp_frac before -> after at each percentile cut")
print(SEP)
print(
    f"  {'Acc':<15} {'GC%':>5} {'n':>6} {'TP%_before':>10} {'frac_neg':>9} "
    f"{'TP%@top25':>10} {'TP%@top50':>10} {'TP%@top75':>10}  Gate"
)
print(f"  {'-'*15} {'-'*5} {'-'*6} {'-'*10} {'-'*9} {'-'*10} {'-'*10} {'-'*10}  ----")
for _, r in df.sort_values("tp_frac_before").iterrows():
    g = "FIRES" if r["gate_fires"] else "closed"
    print(
        f"  {r['acc']:<15} {r['gc_pct']:>5.1f} {r['n_training']:>6} "
        f"{r['tp_frac_before']*100:>9.1f}% {r['frac_negative']:>9.3f} "
        f"{r.get('tp_frac_after_p25',0)*100:>9.1f}% "
        f"{r.get('tp_frac_after_p50',0)*100:>9.1f}% "
        f"{r.get('tp_frac_after_p75',0)*100:>9.1f}%  {g}"
    )

print(f"\n{SEP}")
print("ANALYSIS: Best percentile cut per genome (max TP% improvement)")
print(SEP)
problem = df[df["tp_frac_before"] < 0.75].copy()
clean = df[df["tp_frac_before"] >= 0.75].copy()

for label, subset in [
    ("Problem genomes (TP% < 75%)", problem),
    ("Clean genomes  (TP% >= 75%)", clean),
]:
    if len(subset) == 0:
        continue
    print(f"\n  {label}  (n={len(subset)})")
    for _, r in subset.sort_values("tp_frac_before").iterrows():
        best_pct = max(PERCENTILES, key=lambda p: r.get(f"delta_p{p}", -99))
        best_delta = r.get(f"delta_p{best_pct}", 0)
        best_n = r.get(f"n_after_p{best_pct}", 0)
        best_tp = r.get(f"tp_frac_after_p{best_pct}", 0)
        print(
            f"    {r['acc']:<15} gc={r['gc_pct']:4.1f}%  "
            f"before: {r['tp_frac_before']*100:.1f}% TP ({r['n_training']} ORFs)  "
            f"best: top-{best_pct}% -> {best_tp*100:.1f}% TP "
            f"({best_n} ORFs, delta={best_delta:+.3f})"
        )

print(f"\n{SEP}")
print("GATE ANALYSIS: Would the gate correctly fire/suppress?")
print(SEP)
print(f"  Gate threshold: frac_negative > {GATE_THRESHOLD}")
n_fires = df["gate_fires"].sum()
n_total_g = len(df)
print(f"  Gate fires on {n_fires}/{n_total_g} genomes")
print(f"\n  Problem genomes where gate fires (CORRECT):")
for _, r in df[(df["gate_fires"]) & (df["tp_frac_before"] < 0.75)].iterrows():
    print(
        f"    {r['acc']:<15} tp_before={r['tp_frac_before']*100:.1f}%  frac_neg={r['frac_negative']:.3f}"
    )
print(f"\n  Problem genomes where gate does NOT fire (MISSED):")
for _, r in df[(~df["gate_fires"]) & (df["tp_frac_before"] < 0.75)].iterrows():
    print(
        f"    {r['acc']:<15} tp_before={r['tp_frac_before']*100:.1f}%  frac_neg={r['frac_negative']:.3f}"
    )
print(f"\n  Clean genomes where gate incorrectly fires (FALSE ALARM):")
for _, r in df[(df["gate_fires"]) & (df["tp_frac_before"] >= 0.75)].iterrows():
    print(
        f"    {r['acc']:<15} tp_before={r['tp_frac_before']*100:.1f}%  frac_neg={r['frac_negative']:.3f}"
    )

out = OUT_DIR / "iterative_training_diagnostic.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
