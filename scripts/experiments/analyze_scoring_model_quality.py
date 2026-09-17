# EXPERIMENT: Does 2-pass training set produce better scoring models?
# STATUS: active
# RESULT: pending
"""
Tests whether the cleaner training set (target=6000, keep-top-70%) actually
produces IMM/codon/RBS models that better separate real genes from FPs.

If the scoring models improve --> worth retraining Hybrid and SSC.
If not --> codon/IMM models are robust to contamination --> skip Hybrid/SSC retrain.

For each holdout genome, builds TWO scoring models:
  Model A (baseline): target=2000, no pass-2
  Model B (2-pass):   target=6000, keep top-70%

Then scores ALL ORF candidates with each model and computes:
  - AUC(combined_score) for separating TP from FP candidates
  - AUC(rbs_score), AUC(codon_score), AUC(imm_score) separately
  - Delta AUC (Model B - Model A): does 2-pass improve discrimination?

Success criterion: mean delta AUC > +0.01 on problem genomes.

Run from repo root:
    python scripts/experiments/analyze_scoring_model_quality.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, TEST_GENOMES
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


def is_tp_cand(cand, exact, stops):
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


def build_pass2_models(seq, orfs, gc, intergenic):
    """Build scoring models using target=6000, keep-top-70% pass-2."""
    with contextlib.redirect_stdout(io.StringIO()):
        training = create_training_set(
            sequence=seq,
            all_orfs=orfs,
            glimmer_max_size=6000,
            flexible_target_size=6000,
        )
        training = filter_training_adaptive(list(training), gc)
        models1 = build_all_scoring_models(training, intergenic)
        tdf = pd.DataFrame(training)
        scored_t = score_all_orfs(tdf, models1)

    if "combined_score" in scored_t.columns and len(tdf) > 300:
        thresh = float(np.percentile(scored_t["combined_score"].values, 30))
        keep = scored_t["combined_score"].values >= thresh
        if keep.sum() >= 150:
            training_p2 = tdf[keep].to_dict("records")
            with contextlib.redirect_stdout(io.StringIO()):
                models2 = build_all_scoring_models(training_p2, intergenic)
            return models2, len(training_p2)
    return models1, len(training)


def build_baseline_models(seq, orfs, gc, intergenic):
    """Build scoring models using target=2000, no pass-2 (current baseline)."""
    with contextlib.redirect_stdout(io.StringIO()):
        training = create_training_set(sequence=seq, all_orfs=orfs)
        training = filter_training_adaptive(list(training), gc)
        models = build_all_scoring_models(training, intergenic)
    return models, len(training)


SEP = "=" * 100
print(f"\n{SEP}")
print("DIAGNOSTIC: 2-pass training set -> better scoring models?")
print(f"  Model A: target=2000, no pass-2 (current baseline)")
print(f"  Model B: target=6000, keep top-70% (2-pass)")
print(f"  Metric: AUC(combined_score) for TP vs FP ORF candidates")
print(f"  Success: mean delta AUC > +0.01 on problem genomes")
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

    # Build both models
    models_a, n_a = build_baseline_models(seq, orfs, gc, intergenic)
    models_b, n_b = build_pass2_models(seq, orfs, gc, intergenic)

    # Score ALL ORF candidates with each model
    orfs_df = pd.DataFrame(orfs) if not isinstance(orfs, pd.DataFrame) else orfs
    orfs_list = orfs_df.to_dict("records")
    with contextlib.redirect_stdout(io.StringIO()):
        scored_a = score_all_orfs(orfs_df, models_a)
        scored_b = score_all_orfs(orfs_df, models_b)

    # Label each ORF as TP or FP
    labels = np.array([int(is_tp_cand(o, exact, stops)) for o in orfs_list], dtype=int)
    n_tp = labels.sum()
    n_fp = len(labels) - n_tp

    score_cols = ["combined_score", "rbs_score", "codon_score_norm", "imm_score_norm"]
    results_a, results_b = {}, {}
    for col in score_cols:
        if col in scored_a.columns:
            results_a[col] = auc(scored_a[col].fillna(0).values, labels)
        if col in scored_b.columns:
            results_b[col] = auc(scored_b[col].fillna(0).values, labels)

    print(f"    n_orfs={len(orfs):,}  TP={n_tp:,}  FP={n_fp:,}")
    print(f"    training: A={n_a}  B={n_b}")
    print(f"    {'Score':<25} {'AUC_A':>7} {'AUC_B':>7} {'delta':>8}  Verdict")
    print(f"    {'-'*25} {'-'*7} {'-'*7} {'-'*8}  -------")

    row = {
        "acc": acc,
        "gc_pct": round(gc * 100, 1),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_train_a": n_a,
        "n_train_b": n_b,
    }
    for col in score_cols:
        a_val = results_a.get(col, 0.5)
        b_val = results_b.get(col, 0.5)
        delta = b_val - a_val
        verdict = "BETTER" if delta > 0.01 else ("marginal" if delta > 0 else "no benefit")
        print(f"    {col:<25} {a_val:>7.4f} {b_val:>7.4f} {delta:>+8.4f}  {verdict}")
        row[f"auc_a_{col}"] = round(a_val, 4)
        row[f"auc_b_{col}"] = round(b_val, 4)
        row[f"delta_{col}"] = round(delta, 4)
    rows.append(row)

df = pd.DataFrame(rows)

print(f"\n{SEP}")
print("SUMMARY: Mean delta AUC (2-pass vs baseline) across all 20 genomes")
print(SEP)
for col in score_cols:
    dc = f"delta_{col}"
    if dc not in df.columns:
        continue
    mean_d = df[dc].mean()
    n_better = (df[dc] > 0.01).sum()
    verdict = "WORTH RETRAINING" if mean_d > 0.01 else ("marginal" if mean_d > 0 else "no benefit")
    print(f"  {col:<25} mean_delta={mean_d:>+7.4f}  n_better={n_better}/20  -> {verdict}")

print(f"\n{SEP}")
print("PROBLEM GENOMES (low TP% in baseline training set):")
print(SEP)
problem_accs = ["NC_003155.5", "NC_002929.2", "NC_002677.1", "NC_008268.1"]
prob = df[df["acc"].isin(problem_accs)]
if len(prob):
    for col in ["combined_score"]:
        dc = f"delta_{col}"
        if dc in prob.columns:
            print(f"  {col}: mean delta on problem genomes = {prob[dc].mean():>+.4f}")
            for _, r in prob.iterrows():
                a = r.get(f"auc_a_{col}", 0.5)
                b = r.get(f"auc_b_{col}", 0.5)
                print(f"    {r['acc']}  AUC_A={a:.4f}  AUC_B={b:.4f}  delta={b-a:>+.4f}")

out = OUT_DIR / "scoring_model_quality.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
