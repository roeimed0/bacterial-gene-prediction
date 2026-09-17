# EXPERIMENT: Timing start selection after batch vectorization
# STATUS: active
# RESULT: pending
import contextlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIRST_FILTER_THRESHOLD, START_SELECTION_WEIGHTS
from src.data_management import get_data_dir, load_genome_sequence
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

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
ss = StartSelectionClassifier()
ss.load(str(MODELS_DIR / "start_selector.pkl"))
thr = json.load(open(MODELS_DIR / "thresholds.json"))
LGB_T = thr["orf_classifier_lgb"]["threshold"]

TARGETS = [
    ("NC_000913.3", "E.coli 4.6Mbp"),
    ("NC_002677.1", "M.leprae 3.3Mbp"),
    ("NC_003155.5", "Streptomyces 9.0Mbp"),
]

print(f"{'Genome':<20} {'Groups':>7}  {'Contested':>9}  {'Time':>7}  {'ms/contest':>11}")
print("-" * 62)

for acc, label in TARGETS:
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        tr = create_training_set(sequence=seq, all_orfs=orfs)
        ig = create_intergenic_set(sequence=seq, all_orfs=orfs)
        mdl = build_all_scoring_models(tr, ig)
        scored = score_all_orfs(orfs, mdl)
        filt1 = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        grps0 = organize_nested_orfs(filt1)
        grps = lgb.filter_groups(
            groups=grps0,
            genome_id=acc,
            weights=START_SELECTION_WEIGHTS,
            threshold=LGB_T,
            genome_gc=gc,
        )
    n_groups = len(grps)
    t0 = time.perf_counter()
    top = ss.select_best_starts(grps, seq, mdl, START_SELECTION_WEIGHTS)
    elapsed = time.perf_counter() - t0
    # count contested
    n_contested = sum(1 for gdf in grps.values() if (hasattr(gdf, "__len__") and len(gdf) >= 2))
    ms_per = elapsed * 1000 / max(n_contested, 1)
    print(f"{label:<20} {n_groups:>7}  {n_contested:>9}  {elapsed:>6.2f}s  {ms_per:>10.3f}ms")

print("\nDone.")
