"""GC3 direction-corrected before/after test on all 20 holdout genomes."""

import collections
import contextlib
import io
import sys

import numpy as np

sys.path.insert(0, ".")
import pandas as pd

import src.traditional_methods as tm
from src.config import FIRST_FILTER_THRESHOLD, TEST_GENOMES
from src.data_management import get_gff_path, load_genome_sequence
from src.traditional_methods import _gc3_content as _gc3_orig
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    score_all_orfs,
)

GROUPS = {
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

rows = []
for idx, acc in enumerate(TEST_GENOMES, 1):
    genome = load_genome_sequence(f"data/full_dataset/{acc}.fasta")
    seq = genome["sequence"]
    gc_pct = (seq.count("G") + seq.count("C")) / len(seq)
    grp = GROUPS.get(acc, "?")

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)

    ref = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    ref_set = set(zip(ref[ref[2] == "CDS"][3].astype(int), ref[ref[2] == "CDS"][4].astype(int)))
    n_ref = len(ref_set)

    gc3_all = [_gc3_orig(s) for s in orfs["sequence"].values]
    genome_gc3 = float(np.median(gc3_all))
    tsq = models.get("training_sequences", [])
    coding_gc3 = float(np.mean([_gc3_orig(s) for s in tsq])) if tsq else genome_gc3
    direction = "+" if coding_gc3 >= genome_gc3 else "-"

    tm._gc3_content = lambda s: 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        sb = score_all_orfs(orfs.copy(), models)
    tm._gc3_content = _gc3_orig
    with contextlib.redirect_stdout(io.StringIO()):
        sa = score_all_orfs(orfs.copy(), models)

    fb = filter_candidates(sb, **FIRST_FILTER_THRESHOLD)
    fa = filter_candidates(sa, **FIRST_FILTER_THRESHOLD)
    ids_b = set(zip(fb["genome_start"].astype(int), fb["genome_end"].astype(int)))
    ids_a = set(zip(fa["genome_start"].astype(int), fa["genome_end"].astype(int)))
    tp_b = sum(1 for r in ref_set if r in ids_b)
    tp_a = sum(1 for r in ref_set if r in ids_a)
    s_b = 100 * tp_b / max(n_ref, 1)
    s_a = 100 * tp_a / max(n_ref, 1)
    delta = s_a - s_b
    rows.append((acc, grp, gc_pct, s_b, s_a, delta, direction))
    print(
        f"[{idx:02d}/20] {acc}  {grp:<16} GC={gc_pct:.0%}  dir={direction}  "
        f"before={s_b:.1f}%  after={s_a:.1f}%  delta={delta:+.1f}pp",
        flush=True,
    )

print()
print("GROUP SUMMARY")
by_grp = collections.defaultdict(list)
for r in rows:
    by_grp[r[1]].append(r)
all_b, all_a = [], []
for grp in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
    mb = np.mean([r[3] for r in by_grp[grp]])
    ma = np.mean([r[4] for r in by_grp[grp]])
    all_b += [r[3] for r in by_grp[grp]]
    all_a += [r[4] for r in by_grp[grp]]
    print(f"  {grp:<18}  before={mb:.1f}%  after={ma:.1f}%  delta={ma-mb:+.1f}pp")
print(
    f'  {"OVERALL":<18}  before={np.mean(all_b):.1f}%  after={np.mean(all_a):.1f}%  '
    f"delta={np.mean(all_a)-np.mean(all_b):+.1f}pp"
)
