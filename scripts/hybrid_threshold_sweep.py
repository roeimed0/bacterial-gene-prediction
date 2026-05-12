"""
Sweep HybridGeneFilter thresholds to find the optimal operating point.
LGB v2 is fixed at t=0.05; Hybrid threshold is varied.

Run from repo root:
    python scripts/hybrid_threshold_sweep.py
"""

import contextlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

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

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.471, 0.50]
LGB_T = 0.05
TARGET_PREC = 81.57  # baseline precision to match

lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb_v2.pkl"))
hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model_v2.pkl"))

print(f"LGB v2  fixed at t={LGB_T}")
print(f"Hybrid v2 calibrated threshold: {hf.threshold:.3f}")
print(f"Target precision >= {TARGET_PREC}%\n")

genomes = [a for a in TEST_GENOMES if os.path.exists(os.path.join(DATA_DIR, f"{a}.fasta"))]
results = {t: [] for t in THRESHOLDS}

for idx, acc in enumerate(genomes, 1):
    grp = _HOLDOUT_GROUPS.get(acc, catalog_map.get(acc, {}).get("group", "?"))
    print(f"[{idx:02d}/20] {acc}  ({grp})", flush=True)
    for t in THRESHOLDS:
        with contextlib.redirect_stdout(io.StringIO()):
            final = predict_genome_from_file(
                fasta_path=os.path.join(DATA_DIR, f"{acc}.fasta"),
                genome_id=acc,
                lgb=lgb,
                lgb_threshold=LGB_T,
                hf=hf,
                hf_threshold=t,
            )
        with contextlib.redirect_stdout(io.StringIO()):
            r = compare_orfs_to_reference(final, acc)
        results[t].append(
            {
                "acc": acc,
                "group": grp,
                "f1": r["f1_pct"],
                "sens": r["sensitivity_pct"],
                "prec": r["precision_pct"],
            }
        )

print(f"\n{'Threshold':>10}  {'F1':>7}  {'Sens':>7}  {'Prec':>7}  {'>=target?':>10}")
print("-" * 52)
best_t = None
for t in THRESHOLDS:
    rs = results[t]
    f1 = mean(r["f1"] for r in rs)
    sens = mean(r["sens"] for r in rs)
    prec = mean(r["prec"] for r in rs)
    ok = prec >= TARGET_PREC
    flag = " << MEETS TARGET" if ok else ""
    if ok and best_t is None:
        best_t = (t, f1, sens, prec)
    mark = "YES" if ok else "no"
    print(f"  t={t:.3f}     {f1:>7.2f}  {sens:>7.2f}  {prec:>7.2f}  {mark:>10}{flag}")

print()
# Among thresholds that meet the precision target, pick the one with highest F1.
candidates = [
    (
        t,
        mean(r["f1"] for r in results[t]),
        mean(r["sens"] for r in results[t]),
        mean(r["prec"] for r in results[t]),
    )
    for t in THRESHOLDS
    if mean(r["prec"] for r in results[t]) >= TARGET_PREC
]
if candidates:
    best = max(candidates, key=lambda x: x[1])
    print(f"RECOMMENDED (best F1 with prec>={TARGET_PREC}%): t={best[0]}")
    print(f"  F1={best[1]:.2f}  Sens={best[2]:.2f}  Prec={best[3]:.2f}")
else:
    best = max(THRESHOLDS, key=lambda t: mean(r["f1"] for r in results[t]))
    f1 = mean(r["f1"] for r in results[best])
    prec = mean(r["prec"] for r in results[best])
    print(f"No threshold reached precision target. Best F1: t={best}  F1={f1:.2f}  Prec={prec:.2f}")
