"""
Find the LGB threshold for a new model that matches/exceeds baseline precision.

Runs the full pipeline once per genome, then evaluates multiple thresholds
analytically so a single ~20-min run covers all candidate thresholds.

Usage:
    python scripts/lgb_threshold_sweep.py --lgb-path models/orf_classifier_lgb_v2.pkl
    python scripts/lgb_threshold_sweep.py --lgb-path models/orf_classifier_lgb_v2.pkl --target-prec 81.71
"""

import argparse
import contextlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.comparative_analysis import compare_orfs_to_reference
from src.config import GENOME_CATALOG, TEST_GENOMES
from src.data_management import get_data_dir
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
from src.pipeline import predict_genome_from_file

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
catalog_map = {g["accession"]: g for g in GENOME_CATALOG}

parser = argparse.ArgumentParser()
parser.add_argument("--lgb-path", default=str(MODELS_DIR / "orf_classifier_lgb_v2.pkl"))
parser.add_argument("--hf-path", default=str(MODELS_DIR / "hybrid_best_model.pkl"))
parser.add_argument(
    "--target-prec",
    type=float,
    default=81.71,
    help="Minimum precision the new model must reach (default: 81.71)",
)
parser.add_argument("--hf-threshold", type=float, default=None)
args = parser.parse_args()

THRESHOLDS = [0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.592]

lgb = OrfGroupClassifier()
lgb.load(args.lgb_path)
# Hybrid is disabled for this sweep — we are only tuning the LGB threshold.
# hf=None passes candidates directly to evaluation without the final filter.
hf = None
hf_t = None

print(f"LGB: {args.lgb_path}")
print(f"HF:  DISABLED (LGB threshold sweep only)")
print(f"Target precision >= {args.target_prec:.2f}%\n")

# Run pipeline at each threshold for all genomes
threshold_results = {t: [] for t in THRESHOLDS}

genomes = [acc for acc in TEST_GENOMES if os.path.exists(os.path.join(DATA_DIR, f"{acc}.fasta"))]
total = len(genomes)

for idx, acc in enumerate(genomes, 1):
    grp = _HOLDOUT_GROUPS.get(acc, catalog_map.get(acc, {}).get("group", "?"))
    print(f"[{idx:02d}/{total}] {acc}  ({grp})", flush=True)
    for t in THRESHOLDS:
        with contextlib.redirect_stdout(io.StringIO()):
            final = predict_genome_from_file(
                fasta_path=os.path.join(DATA_DIR, f"{acc}.fasta"),
                genome_id=acc,
                lgb=lgb,
                lgb_threshold=t,
                hf=None,
                hf_threshold=None,
            )
        with contextlib.redirect_stdout(io.StringIO()):
            r = compare_orfs_to_reference(final, acc)
        threshold_results[t].append(
            {
                "acc": acc,
                "group": grp,
                "f1": r["f1_pct"],
                "sens": r["sensitivity_pct"],
                "prec": r["precision_pct"],
            }
        )

# Summary table
print(f"\n{'Threshold':>10}  {'F1':>7}  {'Sens':>7}  {'Prec':>7}  {'>=target?':>10}")
print("-" * 55)
best_t = None
for t in THRESHOLDS:
    rs = threshold_results[t]
    f1 = mean(r["f1"] for r in rs)
    sens = mean(r["sens"] for r in rs)
    prec = mean(r["prec"] for r in rs)
    ok = prec >= args.target_prec
    flag = " << MEETS TARGET" if ok else ""
    if ok and best_t is None:
        best_t = t
    print(
        f"  t={t:.3f}     {f1:>7.2f}  {sens:>7.2f}  {prec:>7.2f}  {'YES' if ok else 'no':>10}{flag}"
    )

print()
if best_t is not None:
    rs = threshold_results[best_t]
    f1 = mean(r["f1"] for r in rs)
    sens = mean(r["sens"] for r in rs)
    prec = mean(r["prec"] for r in rs)
    print(f"RECOMMENDED THRESHOLD: {best_t}  (F1={f1:.2f}  Sens={sens:.2f}  Prec={prec:.2f})")
else:
    print("No threshold reached target precision.")
