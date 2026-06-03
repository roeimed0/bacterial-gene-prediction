# EXPERIMENT: Pipeline stage analysis -- where are TPs lost and FPs introduced?
# STATUS: active
# RESULT: pending
"""
Runs each pipeline step separately and evaluates against reference GFF to find:
  - TP coverage at each stage (what % of reference genes survive)
  - Sensitivity (TPs / total reference CDS)
  - Precision (TPs / total predictions)
  - Where the biggest TP losses occur (= biggest improvement opportunities)
  - Where FPs enter the pipeline (= where precision drops)

Pipeline steps evaluated:
  0. Reference: how many CDS are in reference GFF
  1. After find_orfs_candidates (min_length=100): TP coverage in candidate pool
  2. After first filter (filter_candidates): TP survival
  3. After organize_nested_orfs + LGB group filter: TP group survival
  4. After start selection (baseline weighted sum): TP with correct start
  5. After second filter: TP survival
  6. After HybridGeneFilter: final predictions

Genomes: 2 problem (B. pertussis, M. leprae), 2 reference (C. acetobutyl.,
         S. agalactiae), 1 medium (Rhodococcus), 1 Archaea (Haloarcula)

Run from repo root:
    python scripts/experiments/analyze_pipeline_stages.py
"""

import bisect
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, SECOND_FILTER_THRESHOLD, START_SELECTION_WEIGHTS
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier, StartSelectionClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    normalize_all_orf_scores,
    organize_nested_orfs,
    score_all_orfs,
    select_best_starts,
)

DATA_DIR = get_data_dir("full_dataset")
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
STOP_TOL = 3

FOCUS_GENOMES = [
    # (accession, label, phylum, expected_f1)
    ("NC_002929.2", "B. pertussis", "Proteobacteria", 60.1),  # problem
    ("NC_002677.1", "M. leprae", "Actinobacteria", 41.7),  # worst
    ("NC_008268.1", "Rhodococcus", "Actinobacteria", 65.3),  # medium
    ("NC_015948.1", "Haloarcula", "Archaea", 76.8),  # Archaea
    ("NC_003030.1", "C. acetobutyl.", "Firmicutes", 88.0),  # reference good
    ("NC_004350.2", "S. agalactiae", "Firmicutes", 87.1),  # reference good
]


def load_ref_cds(acc):
    """Load reference CDS as (start, end) set and sorted stop list."""
    gff = get_gff_path(acc)
    r = pd.read_csv(gff, sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = sorted(set(c["e"].astype(int).tolist()))
    return exact, stops, len(exact)


def is_tp(start, end, exact, stops):
    if start > end:
        start, end = end, start
    if (start, end) in exact:
        return True
    return any(abs(end - s) <= STOP_TOL for s in stops)


def orfs_to_coords(orfs):
    """Extract (genome_start, genome_end) from ORF dicts or DataFrame."""
    if isinstance(orfs, pd.DataFrame):
        rows = orfs.to_dict("records")
    elif isinstance(orfs, list) and orfs and isinstance(orfs[0], dict):
        rows = orfs
    else:
        return []
    coords = []
    for r in rows:
        gs = int(r.get("genome_start", r.get("start", 0)))
        ge = int(r.get("genome_end", r.get("end", 0)))
        coords.append((gs, ge))
    return coords


def groups_to_coords(groups):
    """Extract best ORF (highest combined_score or first) from each group."""
    coords = []
    for gid, gdf in groups.items():
        if isinstance(gdf, pd.DataFrame):
            if "combined_score" in gdf.columns:
                best = gdf.loc[gdf["combined_score"].idxmax()]
            else:
                best = gdf.iloc[0]
            gs = int(best.get("genome_start", best.get("start", 0)))
            ge = int(best.get("genome_end", best.get("end", 0)))
            coords.append((gs, ge))
        elif isinstance(gdf, list) and gdf:
            r = gdf[0]
            gs = int(r.get("genome_start", r.get("start", 0)))
            ge = int(r.get("genome_end", r.get("end", 0)))
            coords.append((gs, ge))
    return coords


def build_candidate_index(coords):
    """Build sorted index of candidate end positions for fast lookup."""
    ends = sorted(set(ge for gs, ge in coords))
    exact_set = set(coords)
    return exact_set, ends


def eval_coords(coords, exact, stops, n_ref):
    """
    Sensitivity = unique reference genes covered by at least one candidate.
    Iterate over reference genes to avoid double-counting.
    Precision   = predictions that match any reference gene / total predictions.
    """
    if not coords:
        return 0.0, 0.0, 0.0, 0, 0

    cand_exact, cand_ends = build_candidate_index(coords)

    # Coverage: for each reference gene, does any candidate cover it?
    n_covered = 0
    for ref_s, ref_e in exact:
        # Exact match?
        if (ref_s, ref_e) in cand_exact:
            n_covered += 1
            continue
        # Stop-codon match: any candidate with end within STOP_TOL of ref_e?
        i = bisect.bisect_left(cand_ends, ref_e - STOP_TOL)
        while i < len(cand_ends) and cand_ends[i] <= ref_e + STOP_TOL:
            n_covered += 1
            break
            i += 1

    # Precision: how many candidate predictions match any reference gene?
    n_pred = len(coords)
    tps_for_prec = sum(1 for gs, ge in coords if is_tp(gs, ge, exact, stops))

    sens = n_covered / max(n_ref, 1)
    prec = tps_for_prec / max(n_pred, 1)
    f1 = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else 0.0
    return sens, prec, f1, n_covered, n_pred - tps_for_prec


# Load ML models once
lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))
ss = StartSelectionClassifier()
ss.load(str(MODELS_DIR / "start_selector.pkl"))

SEP = "=" * 110
print(f"\n{SEP}")
print("PIPELINE STAGE ANALYSIS -- where are TPs lost and FPs introduced?")
print(f"  Evaluating sensitivity/precision/F1 at each step vs reference GFF")
print(SEP)

STEPS = ["1.ORFs", "2.Filter1", "3.LGB", "4.Starts", "5.Filter2", "6.Hybrid"]
all_rows = []

for acc, label, phylum, expected_f1 in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"  SKIP {acc}")
        continue
    try:
        exact, stops, n_ref = load_ref_cds(acc)
    except Exception:
        print(f"  SKIP {acc}: no GFF")
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    print(
        f"\n  {acc}  {label:<20}  [{phylum}]  gc={gc*100:.1f}%  expected_F1={expected_f1}%  n_ref={n_ref}"
    )
    print(
        f"  {'Step':<14} {'n_cands':>8} {'n_TP':>7} {'Sens':>7} {'Prec':>7} {'F1':>7}  Bottleneck?"
    )
    print(f"  {'-'*14} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*20}")

    with contextlib.redirect_stdout(io.StringIO()):
        # Step 1: ORF detection
        orfs = find_orfs_candidates(seq, min_length=100)
        coords1 = orfs_to_coords(orfs)
        s1, p1, f1s, t1, fp1 = eval_coords(coords1, exact, stops, n_ref)
        print(
            f"  {'1.ORFs':<14} {len(coords1):>8} {t1:>7} {s1*100:>7.1f}% {p1*100:>7.1f}% {f1s*100:>7.1f}%"
            f"  {'<-- SENS CEILING' if s1 < 0.90 else ''}"
        )

        # Step 2: training, models, score, first filter
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        filtered1 = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        coords2 = orfs_to_coords(filtered1)
        s2, p2, f2s, t2, fp2 = eval_coords(coords2, exact, stops, n_ref)
        loss2 = t1 - t2
        print(
            f"  {'2.Filter1':<14} {len(coords2):>8} {t2:>7} {s2*100:>7.1f}% {p2*100:>7.1f}% {f2s*100:>7.1f}%"
            f"  {'<-- loses ' + str(loss2) + ' TPs' if loss2 > n_ref*0.02 else ''}"
        )

        # Step 3: LGB group filter
        groups0 = organize_nested_orfs(filtered1)
        groups = lgb.filter_groups(
            groups=groups0, genome_id=acc, weights=START_SELECTION_WEIGHTS, threshold=0.07
        )
        coords3 = groups_to_coords(groups)
        s3, p3, f3s, t3, fp3 = eval_coords(coords3, exact, stops, n_ref)
        loss3 = t2 - t3
        print(
            f"  {'3.LGB':<14} {len(coords3):>8} {t3:>7} {s3*100:>7.1f}% {p3*100:>7.1f}% {f3s*100:>7.1f}%"
            f"  {'<-- loses ' + str(loss3) + ' TPs' if loss3 > n_ref*0.02 else ''}"
        )

        # Step 4: Start selection
        top = ss.select_best_starts(groups, seq, models, START_SELECTION_WEIGHTS)
        coords4 = orfs_to_coords(top)
        s4, p4, f4s, t4, fp4 = eval_coords(coords4, exact, stops, n_ref)
        loss4 = t3 - t4
        print(
            f"  {'4.Starts':<14} {len(coords4):>8} {t4:>7} {s4*100:>7.1f}% {p4*100:>7.1f}% {f4s*100:>7.1f}%"
            f"  {'<-- loses ' + str(loss4) + ' TPs' if loss4 > n_ref*0.02 else ''}"
        )

        # Step 5: Second filter
        filtered2 = filter_candidates(top, **SECOND_FILTER_THRESHOLD)
        coords5 = orfs_to_coords(filtered2)
        s5, p5, f5s, t5, fp5 = eval_coords(coords5, exact, stops, n_ref)
        loss5 = t4 - t5
        print(
            f"  {'5.Filter2':<14} {len(coords5):>8} {t5:>7} {s5*100:>7.1f}% {p5*100:>7.1f}% {f5s*100:>7.1f}%"
            f"  {'<-- loses ' + str(loss5) + ' TPs' if loss5 > n_ref*0.02 else ''}"
        )

        # Step 6: Hybrid filter
        final = hf.filter_candidates(
            candidates=filtered2, genome_id=acc, threshold=hf.threshold, batch_size=32
        )
        coords6 = orfs_to_coords(final)
        s6, p6, f6s, t6, fp6 = eval_coords(coords6, exact, stops, n_ref)
        loss6 = t5 - t6
        print(
            f"  {'6.Hybrid':<14} {len(coords6):>8} {t6:>7} {s6*100:>7.1f}% {p6*100:>7.1f}% {f6s*100:>7.1f}%"
            f"  {'<-- loses ' + str(loss6) + ' TPs' if loss6 > n_ref*0.02 else ''}"
        )

    # Summary: biggest drops
    sens_drops = [
        ("1.ORFs->2.Filter1", s1 - s2),
        ("2.Filter1->3.LGB", s2 - s3),
        ("3.LGB->4.Starts", s3 - s4),
        ("4.Starts->5.Filter2", s4 - s5),
        ("5.Filter2->6.Hybrid", s5 - s6),
    ]
    biggest_sens = max(sens_drops, key=lambda x: x[1])
    print(f"\n  Biggest sensitivity loss: {biggest_sens[0]}  ({biggest_sens[1]*100:+.1f}pp)")
    print(f"  Final vs expected:  F1={f6s*100:.1f}%  expected={expected_f1:.1f}%")

    for step, (sens, prec, f1v, n_tp, n_fp) in zip(
        STEPS,
        [
            (s1, p1, f1s, t1, fp1),
            (s2, p2, f2s, t2, fp2),
            (s3, p3, f3s, t3, fp3),
            (s4, p4, f4s, t4, fp4),
            (s5, p5, f5s, t5, fp5),
            (s6, p6, f6s, t6, fp6),
        ],
    ):
        all_rows.append(
            {
                "acc": acc,
                "label": label,
                "phylum": phylum,
                "step": step,
                "n_cands": n_tp + n_fp,
                "n_tp": n_tp,
                "sens": round(sens * 100, 1),
                "prec": round(prec * 100, 1),
                "f1": round(f1v * 100, 1),
            }
        )

df = pd.DataFrame(all_rows)
print(f"\n{SEP}")
print("CROSS-GENOME SUMMARY: Mean sensitivity at each step")
print(SEP)
print(f"  {'Step':<20} ", end="")
for _, label, _, _ in FOCUS_GENOMES:
    print(f"  {label[:12]:>12}", end="")
print(f"  {'MEAN':>8}")
print("  " + "-" * 110)
for step in STEPS:
    print(f"  {step:<20} ", end="")
    vals = []
    for acc, _, _, _ in FOCUS_GENOMES:
        row = df[(df["acc"] == acc) & (df["step"] == step)]
        if len(row):
            v = row.iloc[0]["sens"]
            print(f"  {v:>11.1f}%", end="")
            vals.append(v)
        else:
            print(f"  {'n/a':>12}", end="")
    print(f"  {np.mean(vals):>7.1f}%" if vals else "")

out = (
    Path(__file__).parent.parent.parent / "lgb_attribution_results" / "pipeline_stage_analysis.csv"
)
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
