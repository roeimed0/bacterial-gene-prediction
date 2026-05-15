"""
Benchmark the full prediction pipeline on all TEST_GENOMES.

Reports F1 / Sensitivity / Precision per genome and per taxonomic group.
Optionally saves results to experiments/log.json for experiment tracking.

Run from repo root:
    python scripts/benchmark.py                        # print results only
    python scripts/benchmark.py --save "description"   # save to experiment log
    python scripts/benchmark.py --compare              # compare vs last saved run
    python scripts/benchmark.py --save "X" --set-baseline  # save + mark as baseline

Options:
    --save DESC       Save results to experiments/log.json with description DESC
    --compare         Show delta vs the previous logged run
    --set-baseline    Mark this run as the new baseline for future comparisons
    --group GROUP     Run only one taxonomy group (Proteobacteria / Firmicutes / ...)
    --limit N         Use only N genomes (quick smoke test)
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.comparative_analysis import compare_orfs_to_reference
from src.config import GENOME_CATALOG, TEST_GENOMES
from src.data_management import get_data_dir
from src.ml_models import HybridGeneFilter, OrfGroupClassifier, StartSelectionClassifier
from src.pipeline import predict_genome_from_file

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = get_data_dir("full_dataset")
LOG_FILE = Path(__file__).parent.parent.parent / "experiments" / "log.json"
THRESHOLDS_FILE = MODELS_DIR / "thresholds.json"
SEP = "=" * 85


def _load_thresholds() -> dict:
    """Load calibrated thresholds from models/thresholds.json."""
    if THRESHOLDS_FILE.exists():
        with open(THRESHOLDS_FILE) as f:
            return json.load(f)
    return {}


# ── Model loading ─────────────────────────────────────────────────────────────


def _model_hash(path: Path) -> str:
    """Short SHA256 of model file for provenance tracking."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def load_models(lgb_path: str = None, hf_path: str = None, ss_path: str = None):
    lgb = OrfGroupClassifier()
    lgb.load(lgb_path or str(MODELS_DIR / "orf_classifier_lgb.pkl"))
    hf = HybridGeneFilter()
    with contextlib.redirect_stdout(io.StringIO()):
        hf.load(hf_path or str(MODELS_DIR / "hybrid_best_model.pkl"))
    ss = None
    ss_file = Path(ss_path) if ss_path else MODELS_DIR / "start_selector.pkl"
    if ss_file.exists():
        ss = StartSelectionClassifier()
        ss.load(str(ss_file))
    return lgb, hf, ss


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_genome(
    accession: str,
    lgb: OrfGroupClassifier,
    hf: HybridGeneFilter,
    lgb_t: float,
    hf_t: float,
    ss: StartSelectionClassifier = None,
) -> dict:
    fasta = os.path.join(DATA_DIR, f"{accession}.fasta")
    if not os.path.exists(fasta):
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        final = predict_genome_from_file(
            fasta_path=fasta,
            genome_id=accession,
            lgb=lgb,
            lgb_threshold=lgb_t,
            hf=hf,
            hf_threshold=hf_t,
            ss=ss,
        )

    with contextlib.redirect_stdout(io.StringIO()):
        r = compare_orfs_to_reference(final, accession)
    return r


# ── Experiment log ────────────────────────────────────────────────────────────


def load_log() -> list:
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def save_log(entries: list) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Benchmark pipeline on TEST_GENOMES.")
parser.add_argument(
    "--save", metavar="DESC", help="Save results to experiment log with this description"
)
parser.add_argument("--compare", action="store_true", help="Show delta vs previous logged run")
parser.add_argument("--set-baseline", action="store_true", help="Mark this run as the new baseline")
parser.add_argument("--group", help="Run only this taxonomy group")
parser.add_argument("--limit", type=int, default=0, help="Limit to N genomes (0=all)")
parser.add_argument(
    "--lgb-path", help="Override LGB model path (default: models/orf_classifier_lgb.pkl)"
)
parser.add_argument(
    "--hf-path", help="Override Hybrid model path (default: models/hybrid_best_model.pkl)"
)
parser.add_argument(
    "--start-selector",
    help="Override start selector model path (default: models/start_selector.pkl)",
)
parser.add_argument(
    "--lgb-threshold", type=float, default=None, help="Override LGB threshold (default: 0.07)"
)
parser.add_argument(
    "--hf-threshold",
    type=float,
    default=None,
    help="Override Hybrid threshold (default: model's saved value)",
)
args = parser.parse_args()

# Taxonomy group mapping for the clean holdout genomes.
# These accessions are intentionally absent from GENOME_CATALOG (training pool),
# so their group must be declared here rather than looked up from the catalog.
_HOLDOUT_GROUPS = {
    # Proteobacteria
    "NC_002947.4": ("Pseudomonas putida KT2440", "Proteobacteria"),
    "NC_002929.2": ("Bordetella pertussis Tohama I", "Proteobacteria"),
    "NC_003143.1": ("Yersinia pestis CO92", "Proteobacteria"),
    "NC_003116.1": ("Neisseria meningitidis Z2491", "Proteobacteria"),
    "NC_004757.1": ("Nitrosomonas europaea ATCC 19718", "Proteobacteria"),
    # Firmicutes
    "NC_008497.1": ("Lactobacillus brevis ATCC 367", "Firmicutes"),
    "NC_004350.2": ("Streptococcus agalactiae A909", "Firmicutes"),
    "NC_006270.3": ("Bacillus licheniformis DSM 13", "Firmicutes"),
    "NC_006274.1": ("Bacillus cereus E33L", "Firmicutes"),
    "NC_003030.1": ("Clostridium acetobutylicum ATCC 824", "Firmicutes"),
    # Actinobacteria
    "NC_003155.5": ("Streptomyces avermitilis MA-4680", "Actinobacteria"),
    "NC_003450.3": ("Corynebacterium glutamicum ATCC 13032", "Actinobacteria"),
    "NC_002677.1": ("Mycobacterium leprae TN", "Actinobacteria"),
    "NC_008268.1": ("Nocardia farcinica IFM 10152", "Actinobacteria"),
    "NC_006958.1": ("Corynebacterium glutamicum R", "Actinobacteria"),
    # Archaea
    "NC_008818.1": ("Hyperthermus butylicus DSM 5456", "Archaea"),
    "NC_015948.1": ("Haloarcula hispanica ATCC 33960", "Archaea"),
    "NC_014408.1": ("Methanobrevibacter ruminantium M1", "Archaea"),
    "NC_019977.1": ("Methanosaeta harundinacea 6Ac", "Archaea"),
    "NC_007644.1": ("Methanoculleus marisnigri JR1", "Archaea"),
}

# Build genome list: look up group from GENOME_CATALOG first, then holdout map
catalog_map = {g["accession"]: g for g in GENOME_CATALOG}
genomes = []
for acc in TEST_GENOMES:
    if acc in catalog_map:
        info = catalog_map[acc]
    elif acc in _HOLDOUT_GROUPS:
        name, group = _HOLDOUT_GROUPS[acc]
        info = {"accession": acc, "name": name, "group": group}
    else:
        info = {"accession": acc, "name": acc, "group": "Unknown"}
    if args.group and info["group"] != args.group:
        continue
    fasta = os.path.join(DATA_DIR, f"{acc}.fasta")
    if os.path.exists(fasta):
        genomes.append(info)

if args.limit:
    genomes = genomes[: args.limit]

print(f"\n{SEP}")
print(f"BENCHMARK — {len(genomes)} genomes from TEST_GENOMES")
print(SEP)

_lgb_path = args.lgb_path or str(MODELS_DIR / "orf_classifier_lgb.pkl")
_hf_path = args.hf_path or str(MODELS_DIR / "hybrid_best_model.pkl")
lgb, hf, ss = load_models(_lgb_path, _hf_path, ss_path=args.start_selector)
_thresh = _load_thresholds()
_lgb_stem = Path(_lgb_path).stem
_hf_stem = Path(_hf_path).stem
lgb_t = args.lgb_threshold or _thresh.get(_lgb_stem, {}).get("threshold") or 0.07
hf_t = args.hf_threshold or _thresh.get(_hf_stem, {}).get("threshold") or hf.threshold

print(f"LGB threshold: {lgb_t}  |  Hybrid threshold: {hf_t}")
print(f"LGB model: {_lgb_path}  [{_model_hash(Path(_lgb_path))}]")
print(f"Hybrid model: {_hf_path}  [{_model_hash(Path(_hf_path))}]\n")

print(f"  {'Accession':<16} {'Group':<18} {'F1':>7} {'Sens':>7} {'Prec':>7} {'Time':>7}")
print(f"  {'-'*16} {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

results = []
group_data = defaultdict(list)
_bench_start = time.perf_counter()

for info in genomes:
    acc = info["accession"]
    group = info.get("group", "Unknown")
    print(f"  {acc}...", end=" ", flush=True)
    _t0 = time.perf_counter()
    r = run_genome(acc, lgb, hf, lgb_t, hf_t, ss=ss)
    _elapsed = time.perf_counter() - _t0
    if r is None:
        print("SKIP")
        continue
    f1 = r["f1_pct"]
    sens = r["sensitivity_pct"]
    prec = r["precision_pct"]
    print(f"\r  {acc:<16} {group:<18} {f1:>7.2f} {sens:>7.2f} {prec:>7.2f} {_elapsed:>6.1f}s")
    results.append(
        {
            "accession": acc,
            "group": group,
            "f1": f1,
            "sensitivity": sens,
            "precision": prec,
            "runtime_s": round(_elapsed, 2),
        }
    )
    group_data[group].append((f1, sens, prec))

# Group summary
print(f"\n{SEP}")
print("GROUP SUMMARY")
print(SEP)
print(f"  {'Group':<22} {'F1':>7} {'Sens':>7} {'Prec':>7}  n")
print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7}  -")
for group in sorted(group_data):
    vals = group_data[group]
    mf1 = mean(v[0] for v in vals)
    ms = mean(v[1] for v in vals)
    mp = mean(v[2] for v in vals)
    print(f"  {group:<22} {mf1:>7.2f} {ms:>7.2f} {mp:>7.2f}  {len(vals)}")

all_f1 = [r["f1"] for r in results]
all_s = [r["sensitivity"] for r in results]
all_p = [r["precision"] for r in results]
overall = {"f1": mean(all_f1), "sensitivity": mean(all_s), "precision": mean(all_p)}
_total_elapsed = time.perf_counter() - _bench_start
print(
    f"\n  {'OVERALL':<22} {overall['f1']:>7.2f}"
    f" {overall['sensitivity']:>7.2f} {overall['precision']:>7.2f}  {len(results)}"
    f"   total: {_total_elapsed:.1f}s"
)

# Compare vs previous run
if args.compare:
    log = load_log()
    if len(log) >= 1:
        prev = log[-1]
        prev_overall = prev["overall"]
        print(f"\n{SEP}")
        print(f"DELTA vs previous run: \"{prev['description']}\" ({prev['timestamp'][:10]})")
        print(SEP)
        print(f"  F1:          {overall['f1'] - prev_overall['f1']:+.2f}pp")
        print(f"  Sensitivity: {overall['sensitivity'] - prev_overall['sensitivity']:+.2f}pp")
        print(f"  Precision:   {overall['precision'] - prev_overall['precision']:+.2f}pp")
        prev_by_acc = {r["accession"]: r for r in prev.get("results", [])}
        regressions = [
            r
            for r in results
            if r["accession"] in prev_by_acc and r["f1"] < prev_by_acc[r["accession"]]["f1"] - 0.05
        ]
        if regressions:
            print("\n  Regressions (>0.05pp drop):")
            for r in regressions:
                delta = r["f1"] - prev_by_acc[r["accession"]]["f1"]
                print(f"    {r['accession']:<16} {delta:+.2f}pp F1")
        else:
            print("  No regressions.")
    else:
        print("\n  No previous run to compare against.")

# Save to log
if args.save:
    log = load_log()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "description": args.save,
        "is_baseline": args.set_baseline,
        "models": {
            "lgb": {
                "path": _lgb_path,
                "hash": _model_hash(Path(_lgb_path)),
                "threshold": lgb_t,
            },
            "hybrid": {
                "path": _hf_path,
                "hash": _model_hash(Path(_hf_path)),
                "threshold": hf_t,
            },
        },
        "n_genomes": len(results),
        "runtime_s": round(_total_elapsed, 1),
        "overall": overall,
        "by_group": {
            g: {
                "f1": mean(v[0] for v in vals),
                "sensitivity": mean(v[1] for v in vals),
                "precision": mean(v[2] for v in vals),
                "n": len(vals),
            }
            for g, vals in group_data.items()
        },
        "results": results,
    }
    log.append(entry)
    save_log(log)
    tag = " [BASELINE]" if args.set_baseline else ""
    print(f"\n  Saved to {LOG_FILE}{tag}  (entry #{len(log)})")

print(SEP)
