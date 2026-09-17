# EXPERIMENT: Training set TP/FP quality at different min_length thresholds
# STATUS: active
# RESULT: pending
"""
For each candidate min_length threshold (300, 250, 200, 150 bp), measures:
  - n_training: total training ORFs selected
  - tp_frac: fraction that match a reference CDS (True Positives)
  - fp_frac: fraction that do NOT match (False Positives)
  - tp_count / fp_count

Focus genomes: the small-genome cases where n_training < 700 and F1 is low.
Also includes two high-F1 reference genomes so we can see what "good" looks like.

TP matching rule: exact (genome_start, genome_end) match to a reference CDS,
OR end coordinate matches within ±3bp (stop codon is reliable; start codon may
differ from reference due to alternative start site annotations).

Decision criteria:
  - A threshold is acceptable if tp_frac >= 0.85 (same TP rate as the current
    300bp threshold on high-performing genomes)
  - Any threshold where tp_frac < 0.80 introduces too much noise into the models

Run from repo root:
    python scripts/experiments/analyze_training_quality.py
"""

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import START_SELECTION_WEIGHTS
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    find_orfs_candidates,
    select_training_flexible,
    select_training_glimmer,
)

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

# Genomes to test: small-n_training problem cases + two reference genomes
# (accession, label, phylum, current_f1)
FOCUS_GENOMES = [
    # Problem cases — small training set
    ("NC_002677.1", "M. leprae", "Actinobacteria", 41.74),
    ("NC_002929.2", "B. pertussis", "Proteobacteria", 60.06),
    ("NC_008818.1", "Hyperthermus", "Archaea", 60.88),
    ("NC_008268.1", "Rhodococcus", "Actinobacteria", 65.34),
    ("NC_003155.5", "S. avermitilis", "Actinobacteria", 66.34),
    # Reference cases — large training set, high F1
    ("NC_003030.1", "C. acetobutyl.", "Firmicutes", 88.01),
    ("NC_004350.2", "S. agalactiae", "Firmicutes", 87.08),
]

MIN_LENGTHS = [300, 250, 200, 150]
STOP_TOLERANCE = 3  # bp tolerance for stop codon match


def load_reference_cds(gff_path: str) -> set:
    """Load reference CDS as set of (genome_start, genome_end) pairs."""
    ref = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
    cds = ref[ref[2] == "CDS"][[3, 4]].rename(columns={3: "start", 4: "end"})
    cds = cds.drop_duplicates()
    return set(zip(cds["start"].astype(int), cds["end"].astype(int)))


def is_tp(orf: dict, ref_coords: set, stop_tol: int = STOP_TOLERANCE) -> bool:
    """TP if exact match OR stop coordinate matches within tolerance."""
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    # Exact match
    if (gs, ge) in ref_coords:
        return True
    # Stop codon match (end within tolerance)
    for rs, re in ref_coords:
        if abs(ge - re) <= stop_tol:
            return True
    return False


def evaluate_training_config(
    orfs_list, ref_coords, min_length, glimmer_max=2000, flex_target=2000, flex_overlap=0.3
):
    """Run Glimmer+Flexible intersection at given params; return TP/FP stats."""
    with contextlib.redirect_stdout(io.StringIO()):
        glimmer = select_training_glimmer(
            orfs_list, min_length=min_length, max_training_size=glimmer_max
        )
        flexible = select_training_flexible(
            orfs_list,
            target_size=flex_target,
            min_length=min_length,
            max_length=20000,
            max_overlap_fraction=flex_overlap,
        )

    g_coords = {(o.get("genome_start", o["start"]), o.get("genome_end", o["end"])) for o in glimmer}
    f_coords = {
        (o.get("genome_start", o["start"]), o.get("genome_end", o["end"])) for o in flexible
    }
    inter_coords = g_coords & f_coords

    intersection = [
        o
        for o in orfs_list
        if (o.get("genome_start", o["start"]), o.get("genome_end", o["end"])) in inter_coords
    ]

    if not intersection:
        return 0, 0, 0, 0.0, 0.0

    tps = sum(1 for o in intersection if is_tp(o, ref_coords))
    fps = len(intersection) - tps
    tp_frac = tps / len(intersection)
    fp_frac = fps / len(intersection)
    return len(intersection), tps, fps, tp_frac, fp_frac


# ── Main ──────────────────────────────────────────────────────────────────────

SEP = "=" * 100
print(f"\n{SEP}")
print("DIAGNOSTIC: Training set TP/FP quality at different min_length thresholds")
print(f"  Focus: small-training-set genomes (n_train < 700) vs reference genomes")
print(f"  TP rule: exact coordinate match OR stop codon within {STOP_TOLERANCE}bp")
print(f"  Acceptable threshold: tp_frac >= 0.85")
print(SEP)

all_rows = []

for acc, label, phylum, f1 in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"  SKIP {acc}: no fasta")
        continue

    try:
        gff_path = get_gff_path(acc)
        ref_coords = load_reference_cds(gff_path)
    except Exception:
        print(f"  SKIP {acc}: no reference GFF")
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)

    orfs_list = orfs.to_dict("records") if isinstance(orfs, pd.DataFrame) else orfs
    n_ref = len(ref_coords)

    print(f"\n  {acc}  {label:<20}  [{phylum}]  F1={f1:.1f}%  ref_cds={n_ref:,}")
    print(
        f"  {'min_len':>8} {'n_train':>8} {'TP':>6} {'FP':>6} {'tp%':>6} " f"{'fp%':>6}  Assessment"
    )
    print(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*25}")

    for ml in MIN_LENGTHS:
        n, tp, fp, tp_f, fp_f = evaluate_training_config(orfs_list, ref_coords, min_length=ml)
        ok = "OK" if tp_f >= 0.85 else ("BORDERLINE" if tp_f >= 0.75 else "POOR")
        print(f"  {ml:>8} {n:>8} {tp:>6} {fp:>6} {tp_f:>6.1%} {fp_f:>6.1%}  {ok}")
        all_rows.append(
            {
                "accession": acc,
                "label": label,
                "phylum": phylum,
                "f1_pct": f1,
                "n_ref_cds": n_ref,
                "min_length": ml,
                "n_train": n,
                "tp": tp,
                "fp": fp,
                "tp_frac": round(tp_f, 4),
                "fp_frac": round(fp_f, 4),
            }
        )

# Summary across all genomes
print(f"\n{SEP}")
print("SUMMARY: Mean TP fraction by min_length threshold")
print(
    f"  {'min_length':>10}  {'mean_tp%':>10}  {'min_tp%':>9}  " f"{'n_train mean':>13}  Assessment"
)
print(f"  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*13}  {'-'*20}")
df = pd.DataFrame(all_rows)
for ml in MIN_LENGTHS:
    g = df[df["min_length"] == ml]
    mean_tp = g["tp_frac"].mean()
    min_tp = g["tp_frac"].min()
    mean_n = g["n_train"].mean()
    ok = "OK" if min_tp >= 0.85 else ("BORDERLINE" if min_tp >= 0.75 else "POOR")
    print(f"  {ml:>10}  {mean_tp:>10.1%}  {min_tp:>9.1%}  {mean_n:>13.0f}  {ok}")

print(f"\n{SEP}")
print("TRAINING SIZE GAIN vs TP COST")
print(f"  (How many extra ORFs do we gain per 1pp TP fraction dropped?)")
print(SEP)
baseline = df[df["min_length"] == 300][["accession", "n_train", "tp_frac"]].set_index("accession")
for ml in [250, 200, 150]:
    g = df[df["min_length"] == ml][["accession", "n_train", "tp_frac"]].set_index("accession")
    delta_n = (g["n_train"] - baseline["n_train"]).mean()
    delta_tp = (g["tp_frac"] - baseline["tp_frac"]).mean() * 100
    ratio = delta_n / max(abs(delta_tp), 0.01)
    print(
        f"  min_length={ml}: +{delta_n:.0f} ORFs avg, "
        f"{delta_tp:+.2f}pp TP fraction  "
        f"({ratio:.1f} extra ORFs per 1pp TP drop)"
    )

out_path = OUT_DIR / "training_quality_diagnostic.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(SEP)
print("DECISION GUIDE:")
print("  min_length=X is acceptable if: tp_frac >= 0.85 on ALL focus genomes")
print("  Preferred: highest min_length where ALL genomes still reach n_train >= 700")
print("  If no single threshold works for all: consider adaptive (genome-specific) threshold")
print(SEP)
