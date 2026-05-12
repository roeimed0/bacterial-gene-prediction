"""
Stage-by-stage sensitivity analysis of the production pipeline.

For every holdout genome, counts surviving real genes (TP) at each stage:
  1. ORF detection
  2. First filter
  3. LGB group filter
  4. Start selection + second filter
  5. Hybrid filter (final predictions)

Reports per-genome and per-group breakdowns to identify where genes are
lost and which genome types are most affected.

Run from repo root:
    python scripts/pipeline_stage_analysis.py
"""

import contextlib
import io
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    FIRST_FILTER_THRESHOLD,
    SECOND_FILTER_THRESHOLD,
    START_SELECTION_WEIGHTS,
    TEST_GENOMES,
)
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    organize_nested_orfs,
    score_all_orfs,
    select_best_starts,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = get_data_dir("full_dataset")

_HOLDOUT_GROUPS = {
    "NC_002947.4": "Proteobacteria",
    "NC_002929.2": "Proteobacteria",
    "NC_003143.1": "Proteobacteria",
    "NC_003116.1": "Proteobacteria",
    "NC_004757.1": "Proteobacteria",
    "NC_008497.1": "Firmicutes",
    "NC_004350.2": "Firmicutes",
    "NC_006270.3": "Firmicutes",
    "NC_006274.1": "Firmicutes",
    "NC_003030.1": "Firmicutes",
    "NC_003155.5": "Actinobacteria",
    "NC_003450.3": "Actinobacteria",
    "NC_002677.1": "Actinobacteria",
    "NC_008268.1": "Actinobacteria",
    "NC_006958.1": "Actinobacteria",
    "NC_008818.1": "Archaea",
    "NC_015948.1": "Archaea",
    "NC_014408.1": "Archaea",
    "NC_019977.1": "Archaea",
    "NC_007644.1": "Archaea",
}

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))


def get_ref_set(acc):
    ref = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    return set(zip(ref[ref[2] == "CDS"][3].astype(int), ref[ref[2] == "CDS"][4].astype(int)))


def count_tp(df, rset):
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if df is None or len(df) == 0:
        return 0
    gs = df.get("genome_start", df.get("start", pd.Series(dtype=int))).astype(int)
    ge = df.get("genome_end", df.get("end", pd.Series(dtype=int))).astype(int)
    return sum(1 for s, e in zip(gs, ge) if (s, e) in rset)


rows = []
genomes = [a for a in TEST_GENOMES if (Path(DATA_DIR) / f"{a}.fasta").exists()]
total = len(genomes)

for idx, acc in enumerate(genomes, 1):
    grp = _HOLDOUT_GROUPS.get(acc, "?")
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    gc_pct = (seq.count("G") + seq.count("C")) / len(seq)
    rset = get_ref_set(acc)
    n_ref = len(rset)

    print(f"[{idx:02d}/{total}] {acc}  GC={gc_pct:.0%}  ref={n_ref}  [{grp}]", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        f1 = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups = organize_nested_orfs(f1)
        groups = lgb.filter_groups(
            groups, genome_id=acc, weights=START_SELECTION_WEIGHTS, threshold=0.07
        )
        top = select_best_starts(groups, START_SELECTION_WEIGHTS)
        f2 = filter_candidates(top, **SECOND_FILTER_THRESHOLD)
        final_raw = hf.filter_candidates(
            candidates=f2.to_dict("records"), genome_id=acc, threshold=hf.threshold, batch_size=64
        )
    final = pd.DataFrame(final_raw) if isinstance(final_raw, list) else final_raw

    tp1 = count_tp(f1, rset)
    tp2 = count_tp(top, rset)
    tp3 = count_tp(f2, rset)
    tp4 = count_tp(final, rset)
    tp0 = count_tp(scored, rset)

    rows.append(
        {
            "acc": acc,
            "group": grp,
            "gc_pct": gc_pct,
            "n_ref": n_ref,
            "tp_scored": tp0,
            "tot_scored": len(scored),
            "tp_f1": tp1,
            "tot_f1": len(f1),
            "tp_lgb": tp2,
            "tot_lgb": len(top),
            "tp_f2": tp3,
            "tot_f2": len(f2),
            "tp_final": tp4,
            "tot_final": len(final),
            # sensitivity at each stage
            "sens_f1": 100 * tp1 / max(n_ref, 1),
            "sens_lgb": 100 * tp2 / max(n_ref, 1),
            "sens_f2": 100 * tp3 / max(n_ref, 1),
            "sens_final": 100 * tp4 / max(n_ref, 1),
            # genes lost at each stage
            "lost_f1": tp0 - tp1,
            "lost_lgb": tp1 - tp2,
            "lost_f2": tp2 - tp3,
            "lost_hybrid": tp3 - tp4,
            "prec_final": 100 * tp4 / max(len(final), 1),
        }
    )

SEP = "=" * 80
print(f"\n{SEP}")
print("PER-GENOME STAGE BREAKDOWN")
print(SEP)
print(
    f"  {'Accession':<14} {'Grp':<16} {'GC':>4}  "
    f"{'Sens@F1':>8}  {'Sens@LGB':>9}  {'Sens@F2':>8}  {'Sens@HYB':>9}  "
    f"  Lost:  {'F1':>5}  {'LGB':>5}  {'F2':>5}  {'HYB':>5}"
)
print("  " + "-" * 105)

for r in rows:
    print(
        f"  {r['acc']:<14} {r['group']:<16} {r['gc_pct']:>4.0%}  "
        f"{r['sens_f1']:>7.1f}%  {r['sens_lgb']:>8.1f}%  {r['sens_f2']:>7.1f}%  {r['sens_final']:>8.1f}%"
        f"  {r['lost_f1']:>6}  {r['lost_lgb']:>5}  {r['lost_f2']:>5}  {r['lost_hybrid']:>5}"
    )

print(f"\n{SEP}")
print("GROUP SUMMARY — mean sensitivity and genes lost per stage")
print(SEP)
print(
    f"  {'Group':<18} {'Sens@F1':>8}  {'Sens@LGB':>9}  {'Sens@F2':>8}  {'Sens@HYB':>9}  "
    f"  Lost:  {'F1%':>6}  {'LGB%':>6}  {'F2%':>6}  {'HYB%':>6}"
)
print("  " + "-" * 100)

by_grp = defaultdict(list)
for r in rows:
    by_grp[r["group"]].append(r)

for grp in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
    g = by_grp[grp]

    def m(key):
        return mean(r[key] for r in g)

    # fraction of n_ref lost at each stage
    lf1 = mean(r["lost_f1"] / max(r["n_ref"], 1) * 100 for r in g)
    llgb = mean(r["lost_lgb"] / max(r["n_ref"], 1) * 100 for r in g)
    lf2 = mean(r["lost_f2"] / max(r["n_ref"], 1) * 100 for r in g)
    lhyb = mean(r["lost_hybrid"] / max(r["n_ref"], 1) * 100 for r in g)
    print(
        f"  {grp:<18} {m('sens_f1'):>7.1f}%  {m('sens_lgb'):>8.1f}%  {m('sens_f2'):>7.1f}%  "
        f"{m('sens_final'):>8.1f}%  {lf1:>7.1f}%  {llgb:>7.1f}%  {lf2:>7.1f}%  {lhyb:>7.1f}%"
    )

all_r = rows


def m(key):
    return mean(r[key] for r in all_r)


lf1 = mean(r["lost_f1"] / max(r["n_ref"], 1) * 100 for r in all_r)
llgb = mean(r["lost_lgb"] / max(r["n_ref"], 1) * 100 for r in all_r)
lf2 = mean(r["lost_f2"] / max(r["n_ref"], 1) * 100 for r in all_r)
lhyb = mean(r["lost_hybrid"] / max(r["n_ref"], 1) * 100 for r in all_r)
print(
    f"  {'OVERALL':<18} {m('sens_f1'):>7.1f}%  {m('sens_lgb'):>8.1f}%  {m('sens_f2'):>7.1f}%  "
    f"{m('sens_final'):>8.1f}%  {lf1:>7.1f}%  {llgb:>7.1f}%  {lf2:>7.1f}%  {lhyb:>7.1f}%"
)

print(f"\n{SEP}")
print("BIGGEST BOTTLENECKS — top 5 genomes by genes lost at each stage")
print(SEP)
for stage, lost_key, label in [
    ("1st filter", "lost_f1", "First filter (codon/IMM/length/combined threshold)"),
    ("LGB", "lost_lgb", "LGB group filter"),
    ("2nd filter", "lost_f2", "Second filter (stricter combined threshold)"),
    ("Hybrid", "lost_hybrid", "Hybrid filter (final CNN+Dense)"),
]:
    top5 = sorted(rows, key=lambda r: r[lost_key] / max(r["n_ref"], 1), reverse=True)[:5]
    print(f"\n  {label}:")
    for r in top5:
        pct = r[lost_key] / max(r["n_ref"], 1) * 100
        print(
            f"    {r['acc']}  {r['group']:<16}  GC={r['gc_pct']:.0%}  "
            f"lost {r[lost_key]:>4} genes ({pct:.1f}% of CDS)"
        )
