# EXPERIMENT: Can the existing start classifier correctly fix B4-rank2 errors?
# STATUS: active
# RESULT: pending
"""
ML-NEW1 feasibility check: rank-0 vs rank-2 contest.

The start selection diagnostic (analyze_start_selection_errors.py) identified:
  - B4-rank2: correct start is at baseline rank 2 (not rank 0 or 1).
    The classifier only sees rank-0 vs rank-1, so the correct start is
    never contested.  On problem genomes this accounts for ~256 errors/genome.
  - Of these, 78% have gap(rank0, correct) < contest_t=1.0 -- they WOULD
    be contestable if the classifier were called on rank-0 vs rank-2.

This script answers the de-novo question:
  "If we feed the existing classifier the rank-0 vs rank-2 pair instead of
   rank-0 vs rank-1, does it pick rank-2 (the correct start) purely from
   intrinsic sequence features?"

For each B4-rank2 group on problem + clean genomes:
  1. Build pairwise features for (rank-0, rank-2) -- same _compute_features path
  2. Run the existing classifier (no retraining)
  3. Record: did it flip? (prob_keep < 1 - flip_t = 0.20)
  4. Compare feature distributions to the standard rank-0 vs rank-1 case

If the classifier is already right on >50% of B4-rank2 pairs:
  -> Extending the contest window is a cheap win; no retrain needed yet.
If accuracy is near chance (50%):
  -> Need to add GC-corrected features and retrain before extending.

Run from repo root:
    python scripts/experiments/analyze_rank2_contest_feasibility.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, START_SELECTION_WEIGHTS
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import OrfGroupClassifier, StartSelectionClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    organize_nested_orfs,
    score_all_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3

FOCUS_GENOMES = [
    ("NC_002677.1", "M.leprae", "problem"),
    ("NC_003155.5", "Streptomyces", "problem"),
    ("NC_002929.2", "Bordetella", "problem"),
    ("NC_003030.1", "Clostridium", "clean"),
    ("NC_004350.2", "Streptococcus", "clean"),
]


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    cds = r[r[2] == "CDS"]
    if 8 in cds.columns:
        is_pseudo = cds[8].str.contains("pseudo=true|pseudogene", case=False, na=False)
        cds = cds[~is_pseudo]
    c = cds[[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    return exact


def normalize_coords(gs, ge):
    return (min(gs, ge), max(gs, ge))


def get_orf_coords(row):
    gs = int(row.get("genome_start", row.get("start", 0)))
    ge = int(row.get("genome_end", row.get("end", 0)))
    return normalize_coords(gs, ge)


def find_ref_match(gdf, exact):
    """Return (row_index, ref_coords) for the ORF that matches a reference gene, or (None, None)."""
    for idx, row in gdf.iterrows():
        coords = get_orf_coords(row)
        if coords in exact:
            return idx, coords
    return None, None


def group_stop(gdf):
    ends = [int(r.get("genome_end", r.get("end", 0))) for r in gdf.to_dict("records")]
    return max(ends)


def stop_matches_ref(gdf, exact):
    """True if any ORF in the group shares a stop codon (within STOP_TOL) with a reference gene."""
    stop = group_stop(gdf)
    return any(abs(re - stop) <= STOP_TOL for rs, re in exact)


# ── Load models ──────────────────────────────────────────────────────────────
lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
ss = StartSelectionClassifier()
ss.load(str(MODELS_DIR / "start_selector.pkl"))

SEP = "=" * 100
print(f"\n{SEP}")
print(
    f"RANK-0 vs RANK-2 CONTEST FEASIBILITY  "
    f"(contest_t={ss.contest_t:.2f}  flip_t={ss.flip_t:.2f})"
)
print(SEP)

all_records = []

for acc, name, genome_cat in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"\n  SKIP {acc}")
        continue

    exact = load_ref(acc)
    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc_pct = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups0 = organize_nested_orfs(filtered)
        groups = lgb.filter_groups(
            groups=groups0,
            genome_id=acc,
            weights=START_SELECTION_WEIGHTS,
            threshold=0.07,
            genome_gc=gc_pct,
        )

    # Per-genome classifier context (mirrors select_best_starts internals)
    rbs_pwm = ss._build_rbs_pwm(groups, seq)
    ctx_pwm = ss._build_ctx_pwm(groups, seq)
    len_mean, len_std = ss._build_len_prior(groups)

    n_b4r2_total = 0  # B4-rank2 groups (correct at rank 2)
    n_contestable = 0  # subset where gap(rank0, correct) < contest_t
    n_clf_correct = 0  # classifier flips to rank-2 (correct)
    n_clf_wrong = 0  # classifier keeps rank-0 (wrong)

    # Also track standard B2 accuracy for comparison
    n_b2_total = 0
    n_b2_clf_correct = 0

    group_rows = []

    for gid, gdf in groups.items():
        if isinstance(gdf, list):
            gdf = pd.DataFrame(gdf)
        if len(gdf) < 2:
            continue
        if not stop_matches_ref(gdf, exact):
            continue

        gdf = gdf.copy()
        gdf["_base"] = gdf.apply(lambda r: ss._baseline_score(r, START_SELECTION_WEIGHTS), axis=1)
        sorted_idx = gdf["_base"].sort_values(ascending=False).index.tolist()
        n = len(sorted_idx)

        correct_idx, _ = find_ref_match(gdf, exact)
        if correct_idx is None:
            continue

        try:
            correct_rank = sorted_idx.index(correct_idx)
        except ValueError:
            continue

        # ── B2: correct at rank 1, contested ─────────────────────────────────
        if correct_rank == 1:
            gap_01 = float(gdf.loc[sorted_idx[0], "_base"]) - float(gdf.loc[sorted_idx[1], "_base"])
            if gap_01 < ss.contest_t:
                n_b2_total += 1
                t1 = gdf.loc[sorted_idx[0]]
                t2 = gdf.loc[sorted_idx[1]]
                fv = ss._compute_features(
                    t1,
                    t2,
                    gdf,
                    seq,
                    models,
                    rbs_pwm,
                    ctx_pwm,
                    len_mean,
                    len_std,
                    gc_pct,
                    gap_01,
                )
                X = ss.scaler.transform(np.array([[fv.get(c, 0.0) for c in ss.features]]))
                prob = float(ss.clf.predict_proba(X)[0, 1])
                if prob < (1.0 - ss.flip_t):
                    n_b2_clf_correct += 1
                row = {
                    "type": "B2",
                    "acc": acc,
                    "name": name,
                    "genome_cat": genome_cat,
                    "correct_rank": 1,
                    "gap_01": gap_01,
                    "gap_to_correct": gap_01,
                    "prob_keep": prob,
                    "clf_flipped": prob < (1.0 - ss.flip_t),
                }
                group_rows.append(row)

        # ── B4-rank2: correct at rank 2 ───────────────────────────────────────
        if correct_rank == 2 and n >= 3:
            n_b4r2_total += 1
            gap_01 = float(gdf.loc[sorted_idx[0], "_base"]) - float(gdf.loc[sorted_idx[1], "_base"])
            gap_to_correct = float(gdf.loc[sorted_idx[0], "_base"]) - float(
                gdf.loc[correct_idx, "_base"]
            )

            if gap_to_correct < ss.contest_t:
                n_contestable += 1
                # Feed rank-0 vs rank-2 to the classifier (same feature computation)
                t1 = gdf.loc[sorted_idx[0]]  # rank-0 (wrong)
                t2 = gdf.loc[correct_idx]  # rank-2 (correct)
                fv = ss._compute_features(
                    t1,
                    t2,
                    gdf,
                    seq,
                    models,
                    rbs_pwm,
                    ctx_pwm,
                    len_mean,
                    len_std,
                    gc_pct,
                    gap_to_correct,
                )
                X = ss.scaler.transform(np.array([[fv.get(c, 0.0) for c in ss.features]]))
                prob = float(ss.clf.predict_proba(X)[0, 1])
                flipped = prob < (1.0 - ss.flip_t)

                if flipped:
                    n_clf_correct += 1
                else:
                    n_clf_wrong += 1

                row = {
                    "type": "B4r2",
                    "acc": acc,
                    "name": name,
                    "genome_cat": genome_cat,
                    "correct_rank": 2,
                    "gap_01": gap_01,
                    "gap_to_correct": gap_to_correct,
                    "prob_keep": prob,
                    "clf_flipped": flipped,
                }
                # Key diagnostic features
                for feat in [
                    "d_rbs",
                    "d_anti_sd",
                    "d_codon",
                    "d_imm",
                    "d_length",
                    "d_f4",
                    "d_baseline",
                    "gc_pct",
                    "d_len_zscore",
                ]:
                    row[feat] = fv.get(feat, np.nan)
                group_rows.append(row)

    print(f"\n{SEP}")
    print(f"  {name} ({acc})  GC={gc_pct*100:.1f}%  [{genome_cat}]")
    print(f"\n  B4-rank2 groups (correct at rank 2):  {n_b4r2_total}")
    if n_b4r2_total > 0:
        pct_cont = n_contestable / n_b4r2_total * 100
        print(f"    Contestable at gap<{ss.contest_t:.1f}: {n_contestable} ({pct_cont:.1f}%)")
    if n_contestable > 0:
        acc_pct = n_clf_correct / n_contestable * 100
        print(f"    Classifier accuracy on rank-0 vs rank-2 pairs:")
        print(f"      Correct flips:  {n_clf_correct}/{n_contestable} ({acc_pct:.1f}%)")
        print(
            f"      Wrong keeps:    {n_clf_wrong}/{n_contestable} ({n_clf_wrong/n_contestable*100:.1f}%)"
        )

    print(f"\n  B2 groups (correct at rank 1, contested):  {n_b2_total}")
    if n_b2_total > 0:
        b2_acc = n_b2_clf_correct / n_b2_total * 100
        print(
            f"    Classifier accuracy (current, rank-0 vs rank-1): {n_b2_clf_correct}/{n_b2_total} ({b2_acc:.1f}%)"
        )

    all_records.extend(group_rows)

# ── Cross-genome summary ──────────────────────────────────────────────────────
df = pd.DataFrame(all_records)

print(f"\n{SEP}")
print("CROSS-GENOME SUMMARY")
print(SEP)

for genome_cat in ["problem", "clean"]:
    sub = df[df["genome_cat"] == genome_cat]
    print(f"\n  {genome_cat.upper()} genomes:")

    b4r2 = sub[sub["type"] == "B4r2"]
    if not b4r2.empty:
        acc_pct = b4r2["clf_flipped"].mean() * 100
        print(
            f"    B4-rank2 contestable pairs (n={len(b4r2)}):  "
            f"classifier correct = {acc_pct:.1f}%"
        )
        print(
            f"    prob_keep distribution:  "
            f"median={b4r2['prob_keep'].median():.3f}  "
            f"p25={b4r2['prob_keep'].quantile(0.25):.3f}  "
            f"p75={b4r2['prob_keep'].quantile(0.75):.3f}"
        )

        # Feature signal: d_rbs, d_anti_sd for correct-flip vs wrong-keep
        correct = b4r2[b4r2["clf_flipped"]]
        wrong = b4r2[~b4r2["clf_flipped"]]
        print(f"    Feature signal (correct flips vs wrong keeps):")
        for feat in ["d_rbs", "d_anti_sd", "d_codon", "d_f4", "d_length"]:
            if feat in b4r2.columns:
                c_med = correct[feat].median() if not correct.empty else float("nan")
                w_med = wrong[feat].median() if not wrong.empty else float("nan")
                print(f"      {feat:<14} correct={c_med:>+7.3f}  wrong={w_med:>+7.3f}")

    b2 = sub[sub["type"] == "B2"]
    if not b2.empty:
        b2_acc = b2["clf_flipped"].mean() * 100
        print(f"    B2 (rank-0 vs rank-1, n={len(b2)}):  " f"classifier correct = {b2_acc:.1f}%")

# ── Decision rule ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DECISION")
print(SEP)

if not df.empty:
    b4r2_all = df[df["type"] == "B4r2"]
    if not b4r2_all.empty:
        overall_acc = b4r2_all["clf_flipped"].mean() * 100
        problem_b4r2 = b4r2_all[b4r2_all["genome_cat"] == "problem"]
        problem_acc = problem_b4r2["clf_flipped"].mean() * 100 if not problem_b4r2.empty else 0

        print(f"\n  Overall B4-rank2 classifier accuracy:  {overall_acc:.1f}%")
        print(f"  Problem-genome B4-rank2 accuracy:       {problem_acc:.1f}%")
        print()
        if problem_acc >= 60:
            print("  VERDICT: Extend contest window (rank-0 vs rank-2).")
            print("  The existing classifier already identifies the correct start")
            print("  for the majority of B4-rank2 pairs from intrinsic features alone.")
            print("  Expected sensitivity gain: ~3-5pp on problem genomes.")
        elif problem_acc >= 40:
            print("  VERDICT: Marginal.  Add GC-corrected length feature to classifier")
            print("  before extending the window -- accuracy too low to help as-is.")
        else:
            print("  VERDICT: Classifier is not reliable for rank-0 vs rank-2 pairs.")
            print("  Retrain with GC-stratified data + new features FIRST.")

out = OUT_DIR / "rank2_contest_feasibility.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
