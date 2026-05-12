"""
Stage-by-stage diagnostic: where does GC3 lose candidates vs baseline?

For each target genome, counts surviving real genes (TP) and total
predictions at every pipeline stage with and without GC3.

Run from repo root:
    python scripts/gc3_pipeline_diagnostic.py
"""

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.traditional_methods as tm
from src.config import FIRST_FILTER_THRESHOLD, SECOND_FILTER_THRESHOLD, START_SELECTION_WEIGHTS
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
from src.traditional_methods import _gc3_content as _gc3_real
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

# Worst-hit genomes from the 3-way comparison
TARGETS = [
    ("NC_008818.1", "Hyperthermus butylicus", "Archaea"),  # worst Archaea drop
    ("NC_007644.1", "Methanoculleus marisnigri", "Archaea"),  # 2nd worst
    ("NC_003155.5", "Streptomyces avermitilis", "Actinobacteria"),  # was supposed to benefit
    ("NC_002677.1", "Mycobacterium leprae", "Actinobacteria"),  # our target
]

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))


def ref_set(acc):
    ref = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    return set(zip(ref[ref[2] == "CDS"][3].astype(int), ref[ref[2] == "CDS"][4].astype(int)))


def tp(df_or_list, rset):
    if isinstance(df_or_list, pd.DataFrame):
        return sum(
            1
            for _, r in df_or_list.iterrows()
            if (
                int(r.get("genome_start", r.get("start", 0))),
                int(r.get("genome_end", r.get("end", 0))),
            )
            in rset
        )
    return sum(
        1
        for r in df_or_list
        if (
            int(r.get("genome_start", r.get("start", 0))),
            int(r.get("genome_end", r.get("end", 0))),
        )
        in rset
    )


SEP = "=" * 70

for acc, name, grp in TARGETS:
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    rset = ref_set(acc)
    n_ref = len(rset)
    gc_pct = (seq.count("G") + seq.count("C")) / len(seq)

    print(f"\n{SEP}")
    print(f"{name}  [{grp}]  GC={gc_pct:.0%}  ref_CDS={n_ref:,}")
    print(SEP)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)

    header = f"  {'Stage':<30} {'w/o GC3':>12}  {'w/ GC3':>12}  {'delta':>8}"
    print(header)
    print("  " + "-" * 65)

    for gc3_on in [False, True]:
        tm._gc3_content = _gc3_real if gc3_on else (lambda s: 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            scored = score_all_orfs(orfs.copy(), models)

        f1 = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        with contextlib.redirect_stdout(io.StringIO()):
            groups = organize_nested_orfs(f1)
            groups = lgb.filter_groups(
                groups, genome_id=acc, weights=START_SELECTION_WEIGHTS, threshold=0.07
            )
        top = select_best_starts(groups, START_SELECTION_WEIGHTS)
        f2 = filter_candidates(top, **SECOND_FILTER_THRESHOLD)
        with contextlib.redirect_stdout(io.StringIO()):
            final = hf.filter_candidates(
                candidates=f2.to_dict("records"),
                genome_id=acc,
                threshold=hf.threshold,
                batch_size=64,
            )
        final_df = pd.DataFrame(final) if isinstance(final, list) else final

        tag = "GC3 ON " if gc3_on else "GC3 OFF"
        stages = [
            ("1. All ORFs", len(orfs), tp(orfs, rset)),
            ("2. After 1st filter", len(f1), tp(f1, rset)),
            ("3. After LGB", sum(len(v) for v in groups.values()), tp(top, rset)),
            ("4. After start select", len(top), tp(top, rset)),
            ("5. After 2nd filter", len(f2), tp(f2, rset)),
            ("6. After Hybrid", len(final_df), tp(final_df, rset)),
        ]
        if not gc3_on:
            _base = stages
        else:
            _gc3 = stages

    tm._gc3_content = _gc3_real

    for i, (stage, _, _) in enumerate(_base):
        b_tot, b_tp = _base[i][1], _base[i][2]
        g_tot, g_tp = _gc3[i][1], _gc3[i][2]
        b_sens = 100 * b_tp / n_ref
        g_sens = 100 * g_tp / n_ref
        d_tp = g_tp - b_tp
        d_tot = g_tot - b_tot
        flag = " <<" if abs(d_tp) >= 5 else ""
        print(
            f"  {stage:<30}  tot={b_tot:>6} tp={b_tp:>4} ({b_sens:.0f}%)"
            f"  tot={g_tot:>6} tp={g_tp:>4} ({g_sens:.0f}%)"
            f"  dTP={d_tp:>+4}{flag}"
        )

print(f"\n{SEP}")
print("Done.")
