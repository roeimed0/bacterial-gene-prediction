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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.comparative_analysis import compare_orfs_to_reference
from src.config import (
    FIRST_FILTER_THRESHOLD,
    GENOME_CATALOG,
    SECOND_FILTER_THRESHOLD,
    START_SELECTION_WEIGHTS,
    TEST_GENOMES,
)
from src.data_management import get_data_dir, load_genome_sequence
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
LOG_FILE = Path(__file__).parent.parent / "experiments" / "log.json"
SEP = "=" * 85


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


def load_models():
    lgb = OrfGroupClassifier()
    lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
    hf = HybridGeneFilter()
    with contextlib.redirect_stdout(io.StringIO()):
        hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))
    return lgb, hf


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_genome(
    accession: str, lgb: OrfGroupClassifier, hf: HybridGeneFilter, lgb_t: float, hf_t: float
) -> dict:
    fasta = os.path.join(DATA_DIR, f"{accession}.fasta")
    if not os.path.exists(fasta):
        return None
    genome = load_genome_sequence(fasta)
    if not genome:
        return None
    seq = genome["sequence"]

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups = organize_nested_orfs(filtered)
        groups = lgb.filter_groups(
            groups=groups, genome_id=accession, weights=START_SELECTION_WEIGHTS, threshold=lgb_t
        )
        top = select_best_starts(groups, START_SELECTION_WEIGHTS)
        final = filter_candidates(top, **SECOND_FILTER_THRESHOLD)
        final = hf.filter_candidates(
            candidates=final, genome_id=accession, threshold=hf_t, batch_size=32
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
args = parser.parse_args()

# Build genome list from TEST_GENOMES + GENOME_CATALOG for group lookup
catalog_map = {g["accession"]: g for g in GENOME_CATALOG}
genomes = []
for acc in TEST_GENOMES:
    info = catalog_map.get(acc, {"accession": acc, "name": acc, "group": "Unknown"})
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

lgb, hf = load_models()
lgb_t = 0.07
hf_t = hf.threshold

print(f"LGB threshold: {lgb_t}  |  Hybrid threshold: {hf_t}")
print(f"LGB hash: {_model_hash(MODELS_DIR / 'orf_classifier_lgb.pkl')}")
print(f"Hybrid hash: {_model_hash(MODELS_DIR / 'hybrid_best_model.pkl')}\n")

print(f"  {'Accession':<16} {'Group':<18} {'F1':>7} {'Sens':>7} {'Prec':>7}")
print(f"  {'-'*16} {'-'*18} {'-'*7} {'-'*7} {'-'*7}")

results = []
group_data = defaultdict(list)

for info in genomes:
    acc = info["accession"]
    group = info.get("group", "Unknown")
    print(f"  {acc}...", end=" ", flush=True)
    r = run_genome(acc, lgb, hf, lgb_t, hf_t)
    if r is None:
        print("SKIP")
        continue
    f1 = r["f1_score"]
    sens = r["sensitivity"]
    prec = r["precision"]
    print(f"\r  {acc:<16} {group:<18} {f1:>7.2f} {sens:>7.2f} {prec:>7.2f}")
    results.append(
        {"accession": acc, "group": group, "f1": f1, "sensitivity": sens, "precision": prec}
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
print(
    f"\n  {'OVERALL':<22} {overall['f1']:>7.2f} {overall['sensitivity']:>7.2f} {overall['precision']:>7.2f}  {len(results)}"
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
            print(f"\n  Regressions (>0.05pp drop):")
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
                "path": str(MODELS_DIR / "orf_classifier_lgb.pkl"),
                "hash": _model_hash(MODELS_DIR / "orf_classifier_lgb.pkl"),
                "threshold": lgb_t,
            },
            "hybrid": {
                "path": str(MODELS_DIR / "hybrid_best_model.pkl"),
                "hash": _model_hash(MODELS_DIR / "hybrid_best_model.pkl"),
                "threshold": hf_t,
            },
        },
        "n_genomes": len(results),
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
