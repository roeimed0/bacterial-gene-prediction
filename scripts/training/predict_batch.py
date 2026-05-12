"""
Batch gene prediction — run the full pipeline on multiple FASTA files.

ML models are loaded once and reused across all genomes, so the per-genome
overhead is just the pipeline itself (~10-30 s each) rather than also paying
model-load cost on every invocation.

Input sources (mutually exclusive):
  --input-list FILE    Plain-text file with one FASTA path per line
  --input GLOB         Glob pattern (e.g. "genomes/*.fasta") — quote in shell
  FASTA [FASTA ...]    Positional arguments (one or more paths)

Output:
  --output-dir DIR     Directory for {genome_stem}_predictions.gff files
                       (default: results/)

Run from repo root:
    python scripts/predict_batch.py genome1.fasta genome2.fasta
    python scripts/predict_batch.py --input-list batch.txt --output-dir out/
    python scripts/predict_batch.py --input "genomes/*.fasta"

Options:
    --no-group-ml        Skip LightGBM group filter
    --no-final-ml        Skip HybridGeneFilter
    --group-threshold T  LGB threshold (default: 0.07)
    --final-threshold T  Hybrid threshold (default: 0.25)
    --min-length N       Minimum ORF length in bp (default: 100)
"""

import argparse
import contextlib
import glob
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management import load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
from src.pipeline import predict_genome
from src.pipeline import write_gff as _write_gff_fn

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
SEP = "=" * 70


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Batch gene prediction on multiple FASTA files.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("fastas", nargs="*", help="FASTA file paths")
parser.add_argument("--input-list", metavar="FILE", help="Text file with one FASTA path per line")
parser.add_argument("--input", metavar="GLOB", help="Glob pattern for FASTA files")
parser.add_argument("--output-dir", default="results", help="Output directory (default: results/)")
parser.add_argument("--no-group-ml", action="store_true", help="Skip LGB group filter")
parser.add_argument("--no-final-ml", action="store_true", help="Skip HybridGeneFilter")
parser.add_argument("--group-threshold", type=float, default=0.07)
parser.add_argument("--final-threshold", type=float, default=None)
parser.add_argument("--min-length", type=int, default=100, help="Minimum ORF length bp")
args = parser.parse_args()

# ── Collect input files ───────────────────────────────────────────────────────

input_files: list[str] = []

if args.fastas:
    input_files.extend(args.fastas)
if args.input_list:
    with open(args.input_list) as f:
        input_files.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
if args.input:
    input_files.extend(sorted(glob.glob(args.input)))

if not input_files:
    parser.error("No input files specified. Use positional args, --input-list, or --input.")

# Deduplicate while preserving order
seen: set[str] = set()
unique_files: list[str] = []
for fp in input_files:
    abs_fp = os.path.abspath(fp)
    if abs_fp not in seen:
        seen.add(abs_fp)
        unique_files.append(fp)
input_files = unique_files

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ── Load ML models once ───────────────────────────────────────────────────────

lgb = hf = None

if not args.no_group_ml:
    lgb_path = MODELS_DIR / "orf_classifier_lgb.pkl"
    if lgb_path.exists():
        lgb = OrfGroupClassifier()
        lgb.load(str(lgb_path))
    else:
        print(f"[!] LGB model not found at {lgb_path} — skipping group ML")

if not args.no_final_ml:
    hf_path = MODELS_DIR / "hybrid_best_model.pkl"
    if hf_path.exists():
        hf = HybridGeneFilter()
        with contextlib.redirect_stdout(io.StringIO()):
            hf.load(str(hf_path))
        if args.final_threshold is not None:
            hf.threshold = args.final_threshold
    else:
        print(f"[!] Hybrid model not found at {hf_path} — skipping final ML")

final_t = (
    args.final_threshold if args.final_threshold is not None else (hf.threshold if hf else 0.25)
)

print(f"\n{SEP}")
print(f"BATCH PREDICTION — {len(input_files)} genome(s)")
print(SEP)
print(f"  Output dir:       {output_dir.resolve()}")
print(f"  Group ML:         {'yes (t={:.2f})'.format(args.group_threshold) if lgb else 'disabled'}")
print(f"  Hybrid ML:        {'yes (t={:.2f})'.format(final_t) if hf else 'disabled'}")
print(f"  Min ORF length:   {args.min_length} bp\n")


# ── Per-genome prediction ─────────────────────────────────────────────────────


def _write_gff(predictions, genome_id: str, out_path: Path) -> None:
    _write_gff_fn(predictions, str(out_path), sequence_id=genome_id)


succeeded, failed, skipped = 0, [], 0

for i, fasta_path in enumerate(input_files, 1):
    stem = Path(fasta_path).stem
    out_path = output_dir / f"{stem}_predictions.gff"
    print(f"  [{i:>3}/{len(input_files)}] {Path(fasta_path).name}...", end=" ", flush=True)

    if not os.path.exists(fasta_path):
        print("SKIP (file not found)")
        skipped += 1
        continue

    t0 = time.time()
    try:
        genome = load_genome_sequence(fasta_path)
        if not genome:
            print("SKIP (could not read FASTA)")
            skipped += 1
            continue
        seq = genome["sequence"]
        genome_id = genome.get("id", stem)

        with contextlib.redirect_stdout(io.StringIO()):
            predictions = predict_genome(
                sequence=seq,
                genome_id=genome_id,
                lgb=lgb,
                lgb_threshold=args.group_threshold,
                hf=hf,
                hf_threshold=final_t,
                min_orf_length=args.min_length,
            )

        n_genes = len(predictions) if hasattr(predictions, "__len__") else "?"
        _write_gff(predictions, genome_id, out_path)
        elapsed = time.time() - t0
        print(f"{n_genes} genes  ({elapsed:.1f}s)  -> {out_path.name}")
        succeeded += 1

    except Exception as exc:
        print(f"ERROR: {exc}")
        failed.append((fasta_path, str(exc)))

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print(f"DONE  {succeeded} succeeded  {skipped} skipped  {len(failed)} failed")
if failed:
    print("\nFailed genomes:")
    for fp, err in failed:
        print(f"  {fp}: {err}")
print(SEP)
