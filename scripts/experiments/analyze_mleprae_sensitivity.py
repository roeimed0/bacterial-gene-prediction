# EXPERIMENT: Stage-by-stage sensitivity analysis for M. leprae
# STATUS: active
# RESULT: pending
"""
M. leprae has F1=41.5%, Sens=30.4% — catastrophically low sensitivity.
The scoring models have AUC=0.849 for combined_score (discrimination is fine).
So where are real genes being LOST in the pipeline?

Stage-by-stage analysis:
  Stage 0: Reference genes (ground truth)
  Stage 1: After find_orfs_candidates — do the ORFs even exist?
  Stage 2: After first filter — are they surviving the score threshold?
  Stage 3: After organize_nested_orfs — are they in groups?
  Stage 4: After LGB filter — is LGB rejecting them?
  Stage 5: After select_best_starts — is start selection failing?
  Stage 6: After second filter — final attrition?
  Stage 7: After hybrid filter — final cut?

Also runs on a few comparison genomes (Clostridium/Bordetella) to show contrast.

Run from repo root:
    python scripts/experiments/analyze_mleprae_sensitivity.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    FIRST_FILTER_THRESHOLD,
    SECOND_FILTER_THRESHOLD,
    START_SELECTION_WEIGHTS,
)
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    filter_training_adaptive,
    find_orfs_candidates,
    organize_nested_orfs,
    score_all_orfs,
    select_best_starts,
)

DATA_DIR = get_data_dir("full_dataset")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3

# Load models
lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    cds = r[r[2] == "CDS"]
    # Exclude pseudogene CDS: NCBI marks them with "pseudo=true" or "pseudogene"
    # in the attributes column (col 8). Filter these out so we only count
    # real functional protein-coding genes.
    if 8 in cds.columns:
        is_pseudo = cds[8].str.contains("pseudo=true|pseudogene", case=False, na=False)
        n_pseudo = int(is_pseudo.sum())
        cds = cds[~is_pseudo]
        if n_pseudo > 0:
            print(f"    (filtered {n_pseudo} pseudogene CDS entries from reference)")
    c = cds[[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = set(c["e"].astype(int).tolist())
    return exact, stops, len(c)


def covered_by(candidates, exact, stops, key_start="genome_start", key_end="genome_end"):
    """How many reference stop codons are covered by at least one candidate?"""
    predicted_stops = set()
    for c in candidates:
        if isinstance(c, dict):
            ge = int(c.get(key_end, c.get("end", 0)))
        else:
            ge = int(getattr(c, key_end, getattr(c, "end", 0)))
        predicted_stops.add(ge)
    # Count reference genes with their stop codon in predicted set
    return sum(1 for s in stops if any(abs(s - ps) <= STOP_TOL for ps in predicted_stops))


def covered_exact(candidates, exact, key_start="genome_start", key_end="genome_end"):
    """Exact coordinate matches."""
    matched = set()
    for c in candidates:
        if isinstance(c, dict):
            gs = int(c.get(key_start, c.get("start", 0)))
            ge = int(c.get(key_end, c.get("end", 0)))
        else:
            gs = int(getattr(c, key_start, getattr(c, "start", 0)))
            ge = int(getattr(c, key_end, getattr(c, "end", 0)))
        if gs > ge:
            gs, ge = ge, gs
        matched.add((gs, ge))
    return sum(1 for pair in exact if pair in matched)


def groups_with_tp(groups, exact, stops):
    count = 0
    for gdf in groups.values():
        rows = gdf.to_dict("records") if hasattr(gdf, "to_dict") else list(gdf)
        for r in rows:
            gs = int(r.get("genome_start", r.get("start", 0)))
            ge = int(r.get("genome_end", r.get("end", 0)))
            if gs > ge:
                gs, ge = ge, gs
            if (gs, ge) in exact or any(abs(ge - s) <= STOP_TOL for s in stops):
                count += 1
                break
    return count


FOCUS = [
    ("NC_002677.1", "M.leprae", 57.8, "problem"),
    ("NC_003155.5", "Streptomyces", 70.7, "problem"),
    ("NC_002929.2", "Bordetella", 67.7, "problem"),
    ("NC_003030.1", "Clostridium", 30.9, "clean"),
    ("NC_004350.2", "Streptococcus", 36.8, "clean"),
]

SEP = "=" * 95
print(f"\n{SEP}")
print("STAGE-BY-STAGE SENSITIVITY: where are real genes being lost?")
print(SEP)

all_rows = []

for acc, name, gc_pct, category in FOCUS:
    exact, stops, n_ref = load_ref(acc)
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    print(f"\n{SEP}")
    print(f"  {name} ({acc})  gc={gc*100:.1f}%  [{category}]  ref_genes={n_ref:,}")
    print(SEP)

    def pct(n):
        return f"{n}/{n_ref} ({n/n_ref*100:.1f}%)"

    # Stage 1: ORF finding
    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
    orfs_df = pd.DataFrame(orfs) if not isinstance(orfs, pd.DataFrame) else orfs
    orfs_list = orfs_df.to_dict("records")
    stage1 = covered_by(orfs_list, exact, stops)
    print(f"  Stage 1 — find_orfs_candidates:  {pct(stage1)}  ({len(orfs_list):,} ORFs total)")

    # Stage 2: Training + scoring
    with contextlib.redirect_stdout(io.StringIO()):
        training = create_training_set(sequence=seq, all_orfs=orfs)
        training = filter_training_adaptive(list(training), gc)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs_df, models)

    stage2_all = covered_by(orfs_list, exact, stops)  # same as stage 1 (scoring doesn't remove)
    scored_list = scored.to_dict("records") if hasattr(scored, "to_dict") else list(scored)

    # Stage 3: First filter
    with contextlib.redirect_stdout(io.StringIO()):
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
    filtered_list = filtered.to_dict("records") if hasattr(filtered, "to_dict") else list(filtered)
    stage3 = covered_by(filtered_list, exact, stops)
    print(
        f"  Stage 3 — first filter:          {pct(stage3)}  ({len(filtered_list):,} candidates remain)"
    )

    # Show score distribution of LOST genes at first filter
    if hasattr(scored, "columns") and "combined_score" in scored.columns:
        # Find reference genes that didn't survive the first filter
        surviving_stops = set()
        for c in filtered_list:
            ge = int(c.get("genome_end", c.get("end", 0)))
            surviving_stops.add(ge)
        lost_stops = [
            s for s in stops if not any(abs(s - ps) <= STOP_TOL for ps in surviving_stops)
        ]
        if lost_stops:
            # Find the highest-scoring ORF for each lost gene
            best_scores = []
            for s in lost_stops[:50]:  # sample 50
                mask = (
                    abs(scored["genome_end"].values - s) <= STOP_TOL
                    if "genome_end" in scored.columns
                    else np.zeros(len(scored), dtype=bool)
                )
                if mask.sum() > 0:
                    best_scores.append(float(scored.loc[mask, "combined_score"].max()))
            if best_scores:
                thr = FIRST_FILTER_THRESHOLD.get(
                    "min_score", FIRST_FILTER_THRESHOLD.get("score_threshold", 0)
                )
                print(
                    f"    Lost genes' best combined_score: min={min(best_scores):.3f}  "
                    f"median={np.median(best_scores):.3f}  max={max(best_scores):.3f}  "
                    f"first_filter_threshold~={thr}"
                )
                print(
                    f"    Fraction scoring BELOW threshold: {sum(s < thr for s in best_scores)}/{len(best_scores)}"
                )

    # Stage 4: Groups
    with contextlib.redirect_stdout(io.StringIO()):
        groups = organize_nested_orfs(filtered)
    stage4 = groups_with_tp(groups, exact, stops)
    print(f"  Stage 4 — organize_nested_orfs:  {pct(stage4)}  ({len(groups):,} groups)")

    # Stage 5: LGB filter
    with contextlib.redirect_stdout(io.StringIO()):
        groups_lgb = lgb.filter_groups(
            groups=groups,
            genome_id=acc,
            weights=START_SELECTION_WEIGHTS,
            threshold=0.07,
            genome_gc=gc,
        )
    stage5 = groups_with_tp(groups_lgb, exact, stops)
    print(f"  Stage 5 — LGB filter (t=0.07):   {pct(stage5)}  ({len(groups_lgb):,} groups remain)")

    # Show LGB score distribution for TP vs FP groups
    feat_df = lgb.extract_group_features(groups, acc, weights=START_SELECTION_WEIGHTS, genome_gc=gc)
    feat_df = feat_df.drop(columns=["group_id"], errors="ignore")
    mf = lgb.model.feature_name_
    if mf and mf[0].startswith("Column_"):
        mf = lgb.feature_names or mf
    probs = lgb.model.predict_proba(feat_df[mf].values, num_threads=1)[:, 1]
    gids = list(groups.keys())
    tp_probs = [
        probs[i]
        for i, gid in enumerate(gids)
        if groups_with_tp({gid: groups[gid]}, exact, stops) > 0
    ]
    fp_probs = [
        probs[i]
        for i, gid in enumerate(gids)
        if groups_with_tp({gid: groups[gid]}, exact, stops) == 0
    ]
    if tp_probs and fp_probs:
        print(
            f"    LGB prob — TP groups: median={np.median(tp_probs):.3f}  "
            f"pct_above_t={sum(p>=0.07 for p in tp_probs)/len(tp_probs)*100:.1f}%"
        )
        print(
            f"    LGB prob — FP groups: median={np.median(fp_probs):.3f}  "
            f"pct_above_t={sum(p>=0.07 for p in fp_probs)/len(fp_probs)*100:.1f}%"
        )

    # Stage 6: Start selection
    with contextlib.redirect_stdout(io.StringIO()):
        best_df = select_best_starts(groups_lgb, START_SELECTION_WEIGHTS)
    best_list = best_df.to_dict("records") if hasattr(best_df, "to_dict") else list(best_df)
    stage6 = covered_by(best_list, exact, stops)
    stage6_exact = covered_exact(best_list, exact)
    print(
        f"  Stage 6 — select_best_starts:    {pct(stage6)} (stop match)  "
        f"{stage6_exact}/{n_ref} ({stage6_exact/n_ref*100:.1f}%) exact"
    )

    # Stage 7: Second filter
    with contextlib.redirect_stdout(io.StringIO()):
        filtered2 = filter_candidates(best_df, **SECOND_FILTER_THRESHOLD)
    filtered2_list = (
        filtered2.to_dict("records") if hasattr(filtered2, "to_dict") else list(filtered2)
    )
    stage7 = covered_by(filtered2_list, exact, stops)
    print(f"  Stage 7 — second filter:         {pct(stage7)}  ({len(filtered2_list):,} remain)")

    # Stage 8: Hybrid filter
    cands = filtered2_list
    hf_feats = hf.extract_features(cands, genome_id=acc, genome_gc=gc)
    hf_preds = np.asarray(hf.predict(list(cands))).ravel()
    final = [c for c, p in zip(cands, hf_preds) if int(p) == 1]
    stage8 = covered_by(final, exact, stops)
    print(f"  Stage 8 — hybrid filter:         {pct(stage8)}  ({len(final):,} final predictions)")

    print(f"\n  LOSS SUMMARY:")
    stages = [stage1, stage3, stage4, stage5, stage6, stage7, stage8]
    names = ["find_orfs", "1st_filter", "grouping", "LGB", "start_sel", "2nd_filter", "hybrid"]
    for i in range(1, len(stages)):
        loss = stages[i - 1] - stages[i]
        if loss > 0:
            print(
                f"    {names[i-1]:<12} -> {names[i]:<12}: -{loss} genes lost ({loss/n_ref*100:.1f}%)"
            )

    all_rows.append(
        {
            "acc": acc,
            "name": name,
            "gc_pct": gc_pct,
            "category": category,
            "n_ref": n_ref,
            "stage1_orfs": stage1,
            "stage3_filt1": stage3,
            "stage4_groups": stage4,
            "stage5_lgb": stage5,
            "stage6_starts": stage6,
            "stage7_filt2": stage7,
            "stage8_hybrid": stage8,
            "final_sens": round(stage8 / n_ref * 100, 1),
        }
    )

df = pd.DataFrame(all_rows)
print(f"\n{SEP}")
print("FINAL SENSITIVITY BY STAGE (% of reference genes surviving)")
print(SEP)
print(
    f"  {'Genome':<15} {'Ref':>5} {'ORF':>6} {'Filt1':>6} {'Group':>6} "
    f"{'LGB':>6} {'Start':>6} {'Filt2':>6} {'Final':>6}"
)
print(f"  {'-'*15} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for _, r in df.iterrows():

    def p(n):
        return f"{n/r['n_ref']*100:.0f}%"

    print(
        f"  {r['name']:<15} {r['n_ref']:>5} {p(r['stage1_orfs']):>6} "
        f"{p(r['stage3_filt1']):>6} {p(r['stage4_groups']):>6} "
        f"{p(r['stage5_lgb']):>6} {p(r['stage6_starts']):>6} "
        f"{p(r['stage7_filt2']):>6} {p(r['stage8_hybrid']):>6}"
    )

df.to_csv(OUT_DIR / "mleprae_sensitivity_stages.csv", index=False)
print(f"\nSaved: {OUT_DIR}/mleprae_sensitivity_stages.csv")
print(SEP)
