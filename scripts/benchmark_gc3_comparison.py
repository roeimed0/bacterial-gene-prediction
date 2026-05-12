"""
Three-way comparison benchmark for GC3 weight calibration.

Runs the full pipeline on all 20 holdout genomes under three configurations:
  A) Old weights, no GC3 score  (baseline behaviour)
  B) Old weights + GC3=1.0      (GC3 added, equal weight)
  C) Calibrated weights + GC3   (coordinate-search optimised)

Uses production LGB + Hybrid models throughout so results are end-to-end.
"""

import contextlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.traditional_methods as tm
from src.comparative_analysis import compare_orfs_to_reference
from src.config import (
    FIRST_FILTER_THRESHOLD,
    GENOME_CATALOG,
    START_SELECTION_WEIGHTS,
    TEST_GENOMES,
)
from src.data_management import get_data_dir
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
from src.pipeline import predict_genome_from_file
from src.traditional_methods import _gc3_content as _gc3_real

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

CONFIGS = {
    "A_baseline_no_gc3": {
        "desc": "Old weights, NO GC3",
        "weights": START_SELECTION_WEIGHTS,
        "gc3_on": False,
    },
    "B_old_weights_gc3": {
        "desc": "Old weights + GC3=1.0",
        "weights": START_SELECTION_WEIGHTS,
        "gc3_on": True,
    },
    "C_calibrated_gc3": {
        "desc": "Calibrated weights + GC3=1.25",
        "weights": {
            "codon": 4.8562,
            "imm": 0.7580,
            "rbs": 0.5745,
            "length": 7.4367,
            "start": 0.2066,
            "gc3": 1.2500,
        },
        "gc3_on": True,
    },
}

# Load models once
lgb = OrfGroupClassifier()
lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))

genomes = [acc for acc in TEST_GENOMES if os.path.exists(os.path.join(DATA_DIR, f"{acc}.fasta"))]

all_results = {cfg: {} for cfg in CONFIGS}

_orig_weights = tm.START_SELECTION_WEIGHTS  # save original

for idx, acc in enumerate(genomes, 1):
    grp = _HOLDOUT_GROUPS.get(acc, "?")
    print(f"[{idx:02d}/20] {acc}  ({grp})", flush=True)

    for cfg_key, cfg in CONFIGS.items():
        # Patch GC3 scoring
        tm._gc3_content = _gc3_real if cfg["gc3_on"] else (lambda s: 0.0)
        # Patch start-selection weights directly on the module
        tm.START_SELECTION_WEIGHTS = cfg["weights"]

        with contextlib.redirect_stdout(io.StringIO()):
            final = predict_genome_from_file(
                fasta_path=os.path.join(DATA_DIR, f"{acc}.fasta"),
                genome_id=acc,
                lgb=lgb,
                lgb_threshold=0.07,
                hf=hf,
                hf_threshold=hf.threshold,
            )
        with contextlib.redirect_stdout(io.StringIO()):
            r = compare_orfs_to_reference(final, acc)
        all_results[cfg_key][acc] = {
            "group": grp,
            "f1": r["f1_pct"],
            "sens": r["sensitivity_pct"],
            "prec": r["precision_pct"],
        }

# Restore originals
tm._gc3_content = _gc3_real
tm.START_SELECTION_WEIGHTS = _orig_weights

# ── Summary ─────────────────────────────────────────────────────────────────
SEP = "=" * 80
print(f"\n{SEP}")
print("RESULTS BY GROUP")
print(SEP)

for cfg_key, cfg in CONFIGS.items():
    res = all_results[cfg_key]
    by_grp = defaultdict(list)
    for acc, r in res.items():
        by_grp[r["group"]].append(r)
    print(f"\n  {cfg['desc']}")
    for grp in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
        g = by_grp.get(grp, [])
        if g:
            print(
                f"    {grp:<18}  F1={mean(r['f1'] for r in g):.2f}  "
                f"Sens={mean(r['sens'] for r in g):.2f}  "
                f"Prec={mean(r['prec'] for r in g):.2f}"
            )
    all_f1 = [r["f1"] for r in res.values()]
    all_s = [r["sens"] for r in res.values()]
    all_p = [r["prec"] for r in res.values()]
    print(
        f"    {'OVERALL':<18}  F1={mean(all_f1):.2f}  "
        f"Sens={mean(all_s):.2f}  Prec={mean(all_p):.2f}"
    )

# ── Delta table ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DELTA vs BASELINE (A)")
print(SEP)
base = all_results["A_baseline_no_gc3"]
for cfg_key in ["B_old_weights_gc3", "C_calibrated_gc3"]:
    cfg = CONFIGS[cfg_key]
    res = all_results[cfg_key]
    print(f"\n  vs {cfg['desc']}:")
    by_grp_b = defaultdict(list)
    by_grp_n = defaultdict(list)
    for acc in res:
        g = res[acc]["group"]
        by_grp_b[g].append(base[acc])
        by_grp_n[g].append(res[acc])
    for grp in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
        b = by_grp_b.get(grp, [])
        n = by_grp_n.get(grp, [])
        if b:
            df1 = mean(r["f1"] for r in n) - mean(r["f1"] for r in b)
            ds = mean(r["sens"] for r in n) - mean(r["sens"] for r in b)
            dp = mean(r["prec"] for r in n) - mean(r["prec"] for r in b)
            flag = " <<" if df1 > 0.5 else (" !!" if df1 < -0.5 else "")
            print(f"    {grp:<18}  dF1={df1:+.2f}  dSens={ds:+.2f}  dPrec={dp:+.2f}{flag}")
    df1_all = mean(res[a]["f1"] for a in res) - mean(base[a]["f1"] for a in base)
    ds_all = mean(res[a]["sens"] for a in res) - mean(base[a]["sens"] for a in base)
    dp_all = mean(res[a]["prec"] for a in res) - mean(base[a]["prec"] for a in base)
    print(f"    {'OVERALL':<18}  dF1={df1_all:+.2f}  dSens={ds_all:+.2f}  dPrec={dp_all:+.2f}")
