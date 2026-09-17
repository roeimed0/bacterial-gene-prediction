"""Quick check: TP purity with target=4000 + top-50% pass-2 filter."""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
STOP_TOL = 3


def load_ref(acc):
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    c = r[r[2] == "CDS"][[3, 4]].rename(columns={3: "s", 4: "e"}).drop_duplicates()
    exact = set(zip(c["s"].astype(int), c["e"].astype(int)))
    stops = set(c["e"].astype(int).tolist())
    return exact, stops


def is_tp(orf, exact, stops):
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if (gs, ge) in exact:
        return True
    return any(abs(ge - s) <= STOP_TOL for s in stops)


focus = [
    ("NC_003155.5", "Streptomyces"),
    ("NC_002929.2", "Bordetella"),
    ("NC_002677.1", "M.leprae"),
    ("NC_003030.1", "Clostridium"),
    ("NC_008268.1", "Rhodococcus"),
    ("NC_004350.2", "Streptococcus"),
]

print(
    f"\n  {'Genome':<15} {'n_4000':>7} {'TP%_before':>10} {'n_after50%':>10} {'TP%_after':>10} {'delta':>8}"
)
print(f"  {'-'*15} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

for acc, name in focus:
    exact, stops = load_ref(acc)
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    print(f"  {name:<15}  gc={gc*100:.1f}%", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(
            sequence=seq,
            all_orfs=orfs,
            glimmer_max_size=4000,
            flexible_target_size=4000,
        )
        training = filter_training_adaptive(list(training), gc)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        tdf = pd.DataFrame(training)
        scored = score_all_orfs(tdf, models)

    labels = np.array([int(is_tp(o, exact, stops)) for o in training], dtype=int)
    n_before = len(labels)
    tp_before = int(labels.sum())
    fp_before = n_before - tp_before

    thresh = float(np.percentile(scored["combined_score"].values, 50))
    keep = scored["combined_score"].values >= thresh
    labels_after = labels[keep]
    n_after = int(keep.sum())
    tp_after = int(labels_after.sum())
    fp_after = n_after - tp_after
    delta = (tp_after / n_after - tp_before / n_before) * 100

    print(
        f"    before: n={n_before}  TP={tp_before} ({tp_before/n_before*100:.1f}%)  FP={fp_before}"
    )
    print(
        f"    after:  n={n_after}  TP={tp_after} ({tp_after/n_after*100:.1f}%)  FP={fp_after}  delta={delta:+.1f}pp"
    )
