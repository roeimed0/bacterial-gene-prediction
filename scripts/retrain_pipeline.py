"""
Unified retraining pipeline — train → sweep → benchmark → promote (#166, #176).

Atomic ML retraining cycle:
  1. train_lgb.py                   → orf_classifier_lgb_v2.pkl
  2. lgb_threshold_sweep.py         → best LGB threshold
  3. train_hybrid.py --lgb-path v2  → hybrid_best_model_v2.pkl
  4. hybrid_threshold_sweep.py      → best Hybrid threshold
  5. benchmark.py --compare         → confirm full-pipeline improvement
  6. train_start_classifier.py      → start_selector_retrain.pkl
  7. benchmark_start_classifier.py  → confirm start-selection improvement
  8. If confirmed: promote all → production, update thresholds.json

Run from repo root:
    python scripts/retrain_pipeline.py [--seed 42] [--epochs 50]
                                       [--skip-hybrid] [--skip-start-classifier]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
MODELS = REPO / "models"
SCRIPTS = REPO / "scripts"
THRESHOLDS_FILE = MODELS / "thresholds.json"
PYTHON = sys.executable
SEP = "=" * 70


def run(cmd: list, step: str) -> None:
    print(f"\n{SEP}\nSTEP: {step}\nCMD:  {' '.join(str(c) for c in cmd)}\n{SEP}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        print(f"\n[ERROR] '{step}' failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def read_threshold(output_file: Path) -> float | None:
    """Parse recommended threshold from a sweep output file."""
    import re

    if not output_file.exists():
        return None
    for line in output_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "RECOMMENDED" in line:
            m = re.search(r"t=([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None


def load_thresholds() -> dict:
    return json.load(open(THRESHOLDS_FILE)) if THRESHOLDS_FILE.exists() else {}


def save_thresholds(data: dict) -> None:
    json.dump(data, open(THRESHOLDS_FILE, "w"), indent=2)
    print(f"  Updated {THRESHOLDS_FILE.name}")


parser = argparse.ArgumentParser(description="Atomic ML retraining pipeline.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--limit", type=int, default=0, help="Genomes per group (0=all)")
parser.add_argument("--skip-hybrid", action="store_true")
parser.add_argument("--skip-start-classifier", action="store_true")
parser.add_argument("--target-prec", type=float, default=None)
args = parser.parse_args()

lgb_v2 = MODELS / "orf_classifier_lgb_v2.pkl"
hf_v2 = MODELS / "hybrid_best_model_v2.pkl"
ss_retrain = MODELS / "start_selector_retrain.pkl"
lgb_sweep_out = REPO / "lgb_sweep_v2.txt"
hf_sweep_out = REPO / "hybrid_sweep_v2.txt"

print(f"\n{'='*70}\nRETRAIN PIPELINE  seed={args.seed}  epochs={args.epochs}\n{'='*70}")

# 1. Train LGB
lgb_cmd = [
    PYTHON,
    str(SCRIPTS / "training" / "train_lgb.py"),
    "--seed",
    str(args.seed),
    "--no-val-compare",
]
if args.limit:
    lgb_cmd += ["--limit", str(args.limit)]
run(lgb_cmd, "Train LGB v2")

# 2. Sweep LGB threshold
sweep_cmd = [PYTHON, str(SCRIPTS / "lgb_threshold_sweep.py"), "--lgb-path", str(lgb_v2)]
if args.target_prec:
    sweep_cmd += ["--target-prec", str(args.target_prec)]
with open(lgb_sweep_out, "w") as f:
    subprocess.run(sweep_cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
lgb_t = read_threshold(lgb_sweep_out) or 0.07
print(f"\n  LGB recommended threshold: {lgb_t}")

if not args.skip_hybrid:
    # 3. Train Hybrid
    hf_cmd = [
        PYTHON,
        str(SCRIPTS / "training" / "train_hybrid.py"),
        "--lgb-path",
        str(lgb_v2),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--no-compare",
    ]
    if args.limit:
        hf_cmd += ["--limit", str(args.limit)]
    run(hf_cmd, "Train Hybrid v2")

    # 4. Sweep Hybrid threshold
    hf_cmd2 = [PYTHON, str(SCRIPTS / "hybrid_threshold_sweep.py")]
    if args.target_prec:
        hf_cmd2 += ["--target-prec", str(args.target_prec)]
    with open(hf_sweep_out, "w") as f:
        subprocess.run(hf_cmd2, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    hf_t = read_threshold(hf_sweep_out) or 0.25
    print(f"  Hybrid recommended threshold: {hf_t}")
else:
    hf_v2 = MODELS / "hybrid_best_model.pkl"
    hf_t = load_thresholds().get("hybrid_best_model", {}).get("threshold", 0.25)
    print(f"\n  Hybrid skipped — using production model (t={hf_t})")

# 5. Benchmark
bm_cmd = [
    PYTHON,
    str(SCRIPTS / "evaluation" / "benchmark.py"),
    "--lgb-path",
    str(lgb_v2),
    "--lgb-threshold",
    str(lgb_t),
    "--hf-path",
    str(hf_v2),
    "--hf-threshold",
    str(hf_t),
    "--save",
    f"retrain seed={args.seed} epochs={args.epochs}",
    "--compare",
]
run(bm_cmd, "Benchmark new pipeline")
print(
    "  TIP: run benchmark.py --auto-save at any time to log a run "
    "without a manual --save description."
)

# 6. Train Start Classifier
if not args.skip_start_classifier:
    ss_cmd = [
        PYTHON,
        str(SCRIPTS / "training" / "train_start_classifier.py"),
        "--seed",
        str(args.seed),
        "--out-model",
        str(ss_retrain),
    ]
    run(ss_cmd, "Train Start Classifier (step 6)")

    # 7. Benchmark Start Classifier
    run(
        [PYTHON, str(SCRIPTS / "evaluation" / "benchmark_start_classifier.py")],
        "Benchmark Start Classifier (step 7)",
    )
    print(
        f"\n  NOTE: benchmark_start_classifier.py used production start_selector.pkl.\n"
        f"  To test the new model, manually: copy {ss_retrain.name} → start_selector.pkl\n"
        f"  and re-run benchmark_start_classifier.py before deciding to promote."
    )
else:
    print("\n  Start classifier skipped (--skip-start-classifier).")

# 8. Promote
print(f"\n{SEP}\nCheck benchmark output above.\n{SEP}")
answer = input("\nPromote v2 models to production? [y/N] ").strip().lower()
if answer != "y":
    print("Cancelled.  v2 models saved as *_v2.pkl / start_selector_retrain.pkl.")
    sys.exit(0)

shutil.copy(MODELS / "orf_classifier_lgb.pkl", MODELS / "orf_classifier_lgb_v1_backup.pkl")
shutil.copy(MODELS / "hybrid_best_model.pkl", MODELS / "hybrid_best_model_v1_backup.pkl")
shutil.copy(lgb_v2, MODELS / "orf_classifier_lgb.pkl")
if not args.skip_hybrid:
    shutil.copy(hf_v2, MODELS / "hybrid_best_model.pkl")
if not args.skip_start_classifier and ss_retrain.exists():
    shutil.copy(MODELS / "start_selector.pkl", MODELS / "start_selector_prev_backup.pkl")
    shutil.copy(ss_retrain, MODELS / "start_selector.pkl")
    print("  Promoted start_selector_retrain.pkl → start_selector.pkl")

thresh = load_thresholds()
thresh["orf_classifier_lgb"]["threshold"] = lgb_t
if not args.skip_hybrid:
    thresh["hybrid_best_model"]["threshold"] = hf_t
save_thresholds(thresh)

promoted = f"LGB t={lgb_t}"
if not args.skip_hybrid:
    promoted += f"  Hybrid t={hf_t}"
if not args.skip_start_classifier:
    promoted += "  StartClassifier=retrained"
print(f"\n  Promoted.  {promoted}")
print("  Update models/MODEL_LOG.md with provenance.")
print("  Run: python scripts/evaluation/benchmark.py --save 'retrain ...' --compare")
