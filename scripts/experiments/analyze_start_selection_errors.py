# EXPERIMENT: Start selection error diagnosis -- why does start selection fail on problem genomes?
# STATUS: active
# RESULT: pending
"""
ML-NEW1: Diagnoses start selection errors by error root cause.

For each post-LGB group that matches a reference stop codon, this script:
  1. Finds the correct start codon (from pseudogene-excluded reference)
  2. Ranks all candidate starts by the baseline weighted-sum score
  3. Runs the start classifier logic identically to production
  4. Classifies the outcome into one of these categories:

  CORRECT   : final selected start = reference start (no error)
  A         : correct start was baseline #1, classifier wrongly flipped it
              → fix: raise flip_t or retrain classifier
  B1        : correct start was baseline #2, gap >= contest_t=1.0 (uncontested)
              → fix: widen contest window (lower contest_t)
  B2        : correct start was baseline #2, contested, classifier failed to flip
              → fix: retrain classifier or lower flip_t
  B3        : correct start was baseline #3+, gap >= contest_t
              → fix: improve baseline scoring weights (structural failure)
  B4        : correct start was baseline #3+, contested
              → fix: improve baseline scoring (classifier can't help rank 3+)
  C         : correct start not a candidate in group at all
              → fix: upstream (ORF detection or first filter)

Reports per-genome counts in each category, plus:
  - Baseline score distributions for correct vs chosen starts in error groups
  - Contest threshold sensitivity (how many B1 errors would move to B2 at contest_t=2.0/3.0)
  - Start codon type breakdown for correct vs chosen starts

Run from repo root:
    python scripts/experiments/analyze_start_selection_errors.py
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
    # (accession, label, category)
    ("NC_002677.1", "M.leprae", "problem"),
    ("NC_003155.5", "Streptomyces", "problem"),
    ("NC_002929.2", "Bordetella", "problem"),
    ("NC_003030.1", "Clostridium", "clean"),
    ("NC_004350.2", "Streptococcus", "clean"),
]


def load_ref(acc):
    """Load reference CDS, excluding pseudogenes, returning exact set and stop set."""
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    cds = r[r[2] == "CDS"]
    if 8 in cds.columns:
        is_pseudo = cds[8].str.contains("pseudo=true|pseudogene", case=False, na=False)
        cds = cds[~is_pseudo]
    c = cds[[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    return exact, len(exact)


def normalize(gs, ge):
    return (min(gs, ge), max(gs, ge))


def get_orf_coords(row):
    gs = int(row.get("genome_start", row.get("start", 0)))
    ge = int(row.get("genome_end", row.get("end", 0)))
    return normalize(gs, ge)


def get_group_stop(gdf):
    """Get the shared stop coordinate for a group (max end across all ORFs)."""
    ends = [int(r.get("genome_end", r.get("end", 0))) for r in gdf.to_dict("records")]
    return max(ends)


def find_matching_ref(group_end, exact):
    """
    Find all reference (start, end) pairs whose end is within STOP_TOL of group_end.
    Returns list of (ref_start, ref_end) tuples.
    """
    matches = []
    for rs, re in exact:
        if abs(re - group_end) <= STOP_TOL:
            matches.append((rs, re))
    return matches


def find_correct_orf_in_group(gdf, ref_pairs):
    """
    For each reference pair, find matching ORF in group (exact coords after normalize).
    Returns (orf_row_index, ref_pair) or None.
    """
    ref_set = set(ref_pairs)
    for idx, row in gdf.iterrows():
        coords = get_orf_coords(row)
        if coords in ref_set:
            return idx, coords
    return None, None


def analyze_group(gdf, exact, ss, seq, models, gc_pct, rbs_pwm, ctx_pwm, len_mean, len_std):
    """
    Analyze start selection for one group. Returns a dict with diagnostic info.
    Returns None if the group has no stop-codon match to any reference gene.

    rbs_pwm/ctx_pwm/len_mean/len_std are pre-built per-genome context (mirrors
    what ss.select_best_starts() builds once before the per-group loop).
    """
    gdf = gdf.copy()

    # Get group's shared stop coordinate
    group_stop = get_group_stop(gdf)

    # Find matching reference genes at this stop
    ref_matches = find_matching_ref(group_stop, exact)
    if not ref_matches:
        return None  # not a TP group -- skip

    # Compute baseline scores for all ORFs in group
    gdf["_base"] = gdf.apply(lambda r: ss._baseline_score(r, START_SELECTION_WEIGHTS), axis=1)
    sorted_idx = gdf["_base"].sort_values(ascending=False).index.tolist()

    gap = (
        float(gdf.loc[sorted_idx[0], "_base"]) - float(gdf.loc[sorted_idx[1], "_base"])
        if len(sorted_idx) > 1
        else 999.0
    )

    contested = gap < ss.contest_t
    n_orfs = len(sorted_idx)

    # Find correct start rank
    correct_row_idx, correct_ref = find_correct_orf_in_group(gdf, ref_matches)
    if correct_row_idx is None:
        correct_rank = -1  # not in group (category C)
    else:
        correct_rank = sorted_idx.index(correct_row_idx) if correct_row_idx in sorted_idx else -1

    # Determine classifier decision (if contested) -- use pre-built genome-level context
    baseline_winner_idx = sorted_idx[0]
    final_winner_idx = baseline_winner_idx
    clf_prob_keep = None
    clf_flipped = False

    if contested and len(sorted_idx) >= 2:
        t1 = gdf.loc[sorted_idx[0]]
        t2 = gdf.loc[sorted_idx[1]]

        fv = ss._compute_features(
            t1, t2, gdf, seq, models, rbs_pwm, ctx_pwm, len_mean, len_std, gc_pct, gap
        )
        X = ss.scaler.transform(np.array([[fv.get(c, 0.0) for c in ss.features]]))
        clf_prob_keep = float(ss.clf.predict_proba(X)[0, 1])

        if clf_prob_keep < (1.0 - ss.flip_t):
            final_winner_idx = sorted_idx[1]
            clf_flipped = True

    # Determine if final selection is correct
    final_correct = correct_row_idx is not None and final_winner_idx == correct_row_idx
    baseline_correct = correct_row_idx is not None and baseline_winner_idx == correct_row_idx

    # Classify error category
    if correct_rank == -1:
        category = "C"  # correct start not a candidate
    elif final_correct:
        category = "CORRECT"
    elif correct_rank == 0:
        # baseline was correct but classifier flipped it wrong
        category = "A"
    elif correct_rank == 1:
        if not contested:
            category = "B1"  # uncontested, correct at rank 1 -- widen contest window
        else:
            category = "B2"  # contested but classifier failed to flip
    else:
        # correct start at rank 2+
        if not contested:
            category = "B3"  # uncontested, structural baseline failure
        else:
            category = "B4"  # contested but classifier can't recover rank 2+

    # Score of baseline winner vs correct ORF (for error analysis)
    winner_score = float(gdf.loc[baseline_winner_idx, "_base"])
    correct_score = (
        float(gdf.loc[correct_row_idx, "_base"]) if correct_row_idx is not None else None
    )

    # Start codon types
    winner_codon = str(gdf.loc[baseline_winner_idx].get("start_codon", "?"))
    correct_codon = (
        str(gdf.loc[correct_row_idx].get("start_codon", "?"))
        if correct_row_idx is not None
        else "?"
    )

    # Score gap at the correct rank (for B1 sensitivity: if contest_t were higher, would B1 → B2?)
    # gap_to_correct = winner_score - correct_score (how far is the correct start below #1)
    gap_to_correct = (winner_score - correct_score) if correct_score is not None else None

    return {
        "category": category,
        "n_orfs": n_orfs,
        "correct_rank": correct_rank,
        "gap": gap,
        "gap_to_correct": gap_to_correct,
        "contested": contested,
        "clf_flipped": clf_flipped,
        "clf_prob_keep": clf_prob_keep,
        "winner_score": winner_score,
        "correct_score": correct_score,
        "winner_codon": winner_codon,
        "correct_codon": correct_codon,
        "group_stop": group_stop,
    }


# Load models
lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
ss = StartSelectionClassifier()
ss.load(str(MODELS_DIR / "start_selector.pkl"))

SEP = "=" * 100
print(f"\n{SEP}")
print("START SELECTION ERROR DIAGNOSIS -- ML-NEW1")
print(f"  contest_t={ss.contest_t:.2f}  flip_t={ss.flip_t:.2f}")
print(SEP)

CATEGORIES = ["CORRECT", "A", "B1", "B2", "B3", "B4", "C"]
CAT_LABELS = {
    "CORRECT": "Correct selection",
    "A": "Clf wrongly flipped (baseline was right)",
    "B1": "Correct@rank1, gap>=contest_t (widen window)",
    "B2": "Correct@rank1, contested, clf failed to flip",
    "B3": "Correct@rank2+, uncontested (baseline structural)",
    "B4": "Correct@rank2+, contested (baseline structural)",
    "C": "Correct start not a candidate",
}

all_records = []
summary_rows = []

for acc, name, category_label in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"\n  SKIP {acc} (no fasta)")
        continue

    exact, n_ref = load_ref(acc)
    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc_pct = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    print(f"\n{SEP}")
    print(f"  {name} ({acc})  GC={gc_pct*100:.1f}%  [{category_label}]  ref_genes={n_ref:,}")

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

    print(f"  Post-LGB groups: {len(groups):,}")

    # Build per-genome context once (mirrors ss.select_best_starts internals)
    rbs_pwm_global = ss._build_rbs_pwm(groups, seq)
    ctx_pwm_global = ss._build_ctx_pwm(groups, seq)
    len_mean_global, len_std_global = ss._build_len_prior(groups)

    cat_counts = {c: 0 for c in CATEGORIES}
    group_records = []

    for gid, gdf in groups.items():
        if isinstance(gdf, list):
            gdf = pd.DataFrame(gdf)
        if len(gdf) == 0:
            continue

        result = analyze_group(
            gdf,
            exact,
            ss,
            seq,
            models,
            gc_pct,
            rbs_pwm_global,
            ctx_pwm_global,
            len_mean_global,
            len_std_global,
        )
        if result is None:
            continue  # not a TP group

        result["acc"] = acc
        result["name"] = name
        result["genome_category"] = category_label
        group_records.append(result)
        cat_counts[result["category"]] += 1
        all_records.append(result)

    n_tp_groups = sum(cat_counts.values())
    print(f"  TP groups (stop-matched): {n_tp_groups:,}")
    print()
    print(f"  {'Category':<8}  {'Count':>6}  {'Pct':>6}  Description")
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*45}")
    for cat in CATEGORIES:
        n = cat_counts[cat]
        pct = n / max(n_tp_groups, 1) * 100
        marker = " <<<" if cat not in ("CORRECT", "C") and n > 0 else ""
        print(f"  {cat:<8}  {n:>6}  {pct:>5.1f}%  {CAT_LABELS[cat]}{marker}")

    # Error analysis: score gap for B1 errors (how close were they to contest_t?)
    b1_gaps = [
        r["gap_to_correct"]
        for r in group_records
        if r["category"] == "B1" and r["gap_to_correct"] is not None
    ]
    if b1_gaps:
        # B1 = correct start at rank 1, gap (winner - correct) >= contest_t=1.0
        # gap_to_correct = winner_score - correct_score
        # If we raised contest_t to X, those with gap_to_correct < X would become B2
        print(f"\n  B1 gap_to_correct distribution (winner - correct start score):")
        print(
            f"    n={len(b1_gaps)}  min={min(b1_gaps):.3f}  "
            f"median={np.median(b1_gaps):.3f}  max={max(b1_gaps):.3f}"
        )
        for threshold in [1.5, 2.0, 3.0, 5.0]:
            rescued = sum(g < threshold for g in b1_gaps)
            print(
                f"    contest_t={threshold:.1f}: {rescued}/{len(b1_gaps)} B1 errors become B2 (contestable)"
            )

    # Start codon breakdown for error groups
    error_records = [
        r for r in group_records if r["category"] != "CORRECT" and r["category"] != "C"
    ]
    if error_records:
        winner_codons = [r["winner_codon"] for r in error_records]
        correct_codons = [r["correct_codon"] for r in error_records]
        print(
            f"\n  Wrong-start codon: ATG={winner_codons.count('ATG')}  "
            f"GTG={winner_codons.count('GTG')}  TTG={winner_codons.count('TTG')}"
        )
        print(
            f"  Correct codon:     ATG={correct_codons.count('ATG')}  "
            f"GTG={correct_codons.count('GTG')}  TTG={correct_codons.count('TTG')}"
        )

    row = {
        "acc": acc,
        "name": name,
        "category": category_label,
        "n_ref": n_ref,
        "n_tp_groups": n_tp_groups,
    }
    for cat in CATEGORIES:
        row[cat] = cat_counts[cat]
    summary_rows.append(row)

# Cross-genome summary
df_all = pd.DataFrame(all_records) if all_records else pd.DataFrame()
df_summary = pd.DataFrame(summary_rows)

print(f"\n{SEP}")
print("CROSS-GENOME SUMMARY")
print(SEP)

if not df_summary.empty:
    header = f"  {'Genome':<15} {'N_ref':>6} {'N_TP':>6}  " + "  ".join(
        f"{c:>7}" for c in CATEGORIES
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in df_summary.iterrows():
        vals = "  ".join(f"{r[c]:>7}" for c in CATEGORIES)
        print(f"  {r['name']:<15} {r['n_ref']:>6} {r['n_tp_groups']:>6}  {vals}")

    # Problem vs clean averages (pct)
    print()
    for cat_grp in ["problem", "clean"]:
        subset = df_summary[df_summary["category"] == cat_grp]
        if subset.empty:
            continue
        print(f"  Mean % for {cat_grp} genomes:")
        for cat in CATEGORIES:
            pcts = [row[cat] / max(row["n_tp_groups"], 1) * 100 for _, row in subset.iterrows()]
            print(f"    {cat:<8}: {np.mean(pcts):>5.1f}%")

# Actionable summary
print(f"\n{SEP}")
print("ACTIONABLE FINDINGS")
print(SEP)

if not df_all.empty:
    for genome_cat in ["problem", "clean"]:
        subset = df_all[df_all["genome_category"] == genome_cat]
        if subset.empty:
            continue
        total = len(subset)
        print(f"\n  {genome_cat.upper()} genomes ({total:,} stop-matched groups):")
        for cat in ["A", "B1", "B2", "B3", "B4", "C"]:
            n = (subset["category"] == cat).sum()
            if n > 0:
                print(f"    {cat} ({CAT_LABELS[cat]}): {n} ({n/total*100:.1f}%)")

    # B1 sensitivity to contest_t change (pooled problem genomes)
    b1_all = df_all[(df_all["category"] == "B1") & (df_all["genome_category"] == "problem")]
    if not b1_all.empty:
        print(f"\n  B1 contest_t sensitivity (problem genomes, n={len(b1_all)}):")
        for t in [1.5, 2.0, 3.0, 5.0]:
            rescuable = (b1_all["gap_to_correct"] < t).sum()
            print(f"    contest_t={t:.1f}: {rescuable}/{len(b1_all)} become contestable")

    # Classifier accuracy on contested groups
    contested_subset = df_all[df_all["contested"]]
    if not contested_subset.empty:
        for genome_cat in ["problem", "clean"]:
            sub = contested_subset[contested_subset["genome_category"] == genome_cat]
            if sub.empty:
                continue
            correct_after_clf = (sub["category"] == "CORRECT").sum()
            total_c = len(sub)
            print(
                f"\n  Classifier accuracy on contested groups ({genome_cat}): "
                f"{correct_after_clf}/{total_c} ({correct_after_clf/total_c*100:.1f}%)"
            )
            # Of contested, how many had baseline correct vs incorrect?
            baseline_correct = sub[sub["category"].isin(["CORRECT", "A"])]
            baseline_wrong = sub[sub["category"].isin(["B2"])]
            print(
                f"    Baseline was correct in contested group: {len(baseline_correct)} "
                f"(classifier kept: {(sub['category']=='CORRECT').sum()}, "
                f"wrongly flipped: {(sub['category']=='A').sum()})"
            )
            print(
                f"    Baseline was wrong (correct@rank1) in contested: {len(baseline_wrong)} "
                f"(classifier fixed: 0 shown as B2)"
            )

# Save detailed CSV
if all_records:
    out_path = OUT_DIR / "start_selection_error_analysis.csv"
    pd.DataFrame(all_records).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

print(SEP)
