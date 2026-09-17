# EXPERIMENT: Move pass-2 rejects into intergenic set instead of discarding
# STATUS: active
# RESULT: pending
"""
Tests whether adding pass-2 rejected ORFs (likely pseudogenes/IS elements)
to the intergenic (non-coding) set improves scoring model discrimination.

HYPOTHESIS:
  IMM score = log P(seq | coding) - log P(seq | non-coding)
  Adding pseudogenes to non-coding model makes it include pseudogene patterns.
  Result: real genes score higher (more distinct from non-coding), pseudogenes
  score lower (more similar to the enriched non-coding model).

THREE MODELS COMPARED:
  A: target=2000, no pass-2 (current baseline)
  B: target=6000, top-70% coding only (pass-2 rejects discarded)
  C: target=6000, top-70% coding + bottom-30% added to intergenic (new idea)

METRIC: AUC(combined_score) for separating TP from FP across all ORF candidates.

Run from repo root:
    python scripts/experiments/analyze_fp_to_intergenic.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

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
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = set(c["e"].astype(int).tolist())
    return exact, stops


def is_tp(cand, exact, stops):
    gs = int(cand.get("genome_start", cand.get("start", 0)))
    ge = int(cand.get("genome_end", cand.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in stops)


def auc(vals, labels):
    pos = vals[labels == 1]
    neg = vals[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.5
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    a = u / (len(pos) * len(neg))
    return max(a, 1 - a)


def orfs_to_intergenic_format(orf_dicts):
    """Convert ORF dicts to the intergenic region format expected by build_all_scoring_models."""
    result = []
    for o in orf_dicts:
        seq = o.get("sequence", "")
        if not seq:
            continue
        result.append(
            {
                "start": o.get("genome_start", o.get("start", 0)),
                "end": o.get("genome_end", o.get("end", 0)),
                "length": len(seq),
                "sequence": seq,
                "type": "intergenic",  # mark as non-coding background
            }
        )
    return result


SEP = "=" * 100
print(f"\n{SEP}")
print("DIAGNOSTIC: FP ORFs -> intergenic set (enrich non-coding model)")
print(f"  Model A: baseline (target=2000, no pass-2)")
print(f"  Model B: top-70% coding only (rejects discarded)")
print(f"  Model C: top-70% coding + bottom-30% added to intergenic")
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
    print(f"\n  {acc}  gc={gc*100:.1f}%", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)

    orfs_df = pd.DataFrame(orfs) if not isinstance(orfs, pd.DataFrame) else orfs
    orfs_list = orfs_df.to_dict("records")
    labels = np.array([int(is_tp(o, exact, stops)) for o in orfs_list], dtype=int)
    n_tp, n_fp = int(labels.sum()), len(labels) - int(labels.sum())

    # ── Model A: baseline (target=2000, no pass-2) ────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        train_a = create_training_set(sequence=seq, all_orfs=orfs)
        train_a = filter_training_adaptive(list(train_a), gc)
        models_a = build_all_scoring_models(train_a, intergenic)
        scored_a = score_all_orfs(orfs_df, models_a)

    # ── Build expanded pass-1 for B and C ────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        train_exp = create_training_set(
            sequence=seq,
            all_orfs=orfs,
            glimmer_max_size=6000,
            flexible_target_size=6000,
        )
        train_exp = filter_training_adaptive(list(train_exp), gc)
        models_p1 = build_all_scoring_models(train_exp, intergenic)
        scored_exp = score_all_orfs(pd.DataFrame(train_exp), models_p1)

    if "combined_score" not in scored_exp.columns or len(train_exp) <= 300:
        continue
    thresh = float(np.percentile(scored_exp["combined_score"].values, 30))
    keep = scored_exp["combined_score"].values >= thresh
    n_keep = int(keep.sum())
    n_reject = len(keep) - n_keep

    if n_keep < 150:
        continue

    train_exp_df = pd.DataFrame(train_exp)
    coding_p2 = train_exp_df[keep].to_dict("records")
    rejected = train_exp_df[~keep].to_dict("records")

    # ── Model B: top-70% coding only ─────────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        models_b = build_all_scoring_models(coding_p2, intergenic)
        scored_b = score_all_orfs(orfs_df, models_b)

    # ── Model C: top-70% coding + bottom-30% into intergenic ─────────────────
    intergenic_enriched = intergenic + orfs_to_intergenic_format(rejected)
    with contextlib.redirect_stdout(io.StringIO()):
        models_c = build_all_scoring_models(coding_p2, intergenic_enriched)
        scored_c = score_all_orfs(orfs_df, models_c)

    # AUC comparison
    auc_a = (
        auc(scored_a["combined_score"].fillna(0).values, labels)
        if "combined_score" in scored_a.columns
        else 0.5
    )
    auc_b = (
        auc(scored_b["combined_score"].fillna(0).values, labels)
        if "combined_score" in scored_b.columns
        else 0.5
    )
    auc_c = (
        auc(scored_c["combined_score"].fillna(0).values, labels)
        if "combined_score" in scored_c.columns
        else 0.5
    )

    d_b = auc_b - auc_a
    d_c = auc_c - auc_a
    d_bc = auc_c - auc_b

    labels_coding = np.array([int(is_tp(o, exact, stops)) for o in coding_p2], dtype=int)
    labels_rejected = np.array([int(is_tp(o, exact, stops)) for o in rejected], dtype=int)
    labels_train_a = np.array([int(is_tp(o, exact, stops)) for o in train_a], dtype=int)

    tp_frac_a = labels_train_a.mean()
    tp_frac_coding = labels_coding.mean()
    tp_frac_rejected = labels_rejected.mean() if len(labels_rejected) > 0 else 0.0

    print(f"    TP/FP purity:")
    print(
        f"      baseline train (A):  n={len(train_a):>5}  TP={labels_train_a.sum():>5} ({tp_frac_a*100:.1f}%)  FP={int((1-labels_train_a).sum()):>5}"
    )
    print(
        f"      coding_p2  (top70%): n={n_keep:>5}  TP={labels_coding.sum():>5} ({tp_frac_coding*100:.1f}%)  FP={int((1-labels_coding).sum()):>5}"
    )
    print(
        f"      rejected (->interg): n={n_reject:>5}  TP={labels_rejected.sum():>5} ({tp_frac_rejected*100:.1f}%)  FP={int((1-labels_rejected).sum()):>5}  <-- goes to intergenic"
    )
    print(f"    AUC comparison:")
    print(
        f"    AUC_A={auc_a:.4f}  AUC_B={auc_b:.4f} (delta={d_b:+.4f})  "
        f"AUC_C={auc_c:.4f} (delta_vs_A={d_c:+.4f}, delta_vs_B={d_bc:+.4f})"
    )
    verdict = (
        "C WINS"
        if d_bc > 0.005
        else ("marginal" if d_bc > 0 else ("B wins" if d_b > d_c else "tie"))
    )
    print(f"    -> {verdict}")

    rows.append(
        {
            "acc": acc,
            "gc_pct": round(gc * 100, 1),
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_train_a": len(train_a),
            "tp_frac_a": round(tp_frac_a, 3),
            "n_coding_p2": n_keep,
            "tp_frac_coding": round(tp_frac_coding, 3),
            "n_rejected": n_reject,
            "tp_frac_rejected": round(tp_frac_rejected, 3),
            "auc_a": round(auc_a, 4),
            "auc_b": round(auc_b, 4),
            "auc_c": round(auc_c, 4),
            "delta_b": round(d_b, 4),
            "delta_c": round(d_c, 4),
            "delta_bc": round(d_bc, 4),
        }
    )

df = pd.DataFrame(rows)

print(f"\n{SEP}")
print("SUMMARY: Mean AUC(combined_score) across all 20 genomes")
print(SEP)
print(f"  Model A (baseline t=2000):          mean AUC = {df['auc_a'].mean():.4f}")
print(
    f"  Model B (t=6000, top-70% only):     mean AUC = {df['auc_b'].mean():.4f}  (delta vs A = {df['delta_b'].mean():+.4f})"
)
print(
    f"  Model C (t=6000, top-70% + FP->int): mean AUC = {df['auc_c'].mean():.4f}  (delta vs A = {df['delta_c'].mean():+.4f}, vs B = {df['delta_bc'].mean():+.4f})"
)

print(f"\n  C beats A by >0.005: {(df['delta_c']>0.005).sum()}/20 genomes")
print(f"  C beats B by >0.005: {(df['delta_bc']>0.005).sum()}/20 genomes")
print(f"  C beats A on any:    {(df['delta_c']>0).sum()}/20 genomes")

print(f"\n  Problem genomes (low TP in baseline):")
for acc in ["NC_003155.5", "NC_002929.2", "NC_002677.1", "NC_008268.1"]:
    r = df[df["acc"] == acc]
    if len(r):
        r = r.iloc[0]
        print(
            f"    {acc}: A={r['auc_a']:.4f}  B={r['auc_b']:.4f}  C={r['auc_c']:.4f}  "
            f"C-A={r['delta_c']:+.4f}  C-B={r['delta_bc']:+.4f}"
        )

out = OUT_DIR / "fp_to_intergenic_diagnostic.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
