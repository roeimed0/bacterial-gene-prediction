# EXPERIMENT: Training set TP/FP breakdown — adaptive multi-feature filter
# STATUS: active
# RESULT: pending
"""Shows n, TP, FP for all 20 holdout genomes: baseline vs adaptive multi-feature filter."""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import TEST_GENOMES
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    create_training_set,
    filter_training_adaptive,
    find_orfs_candidates,
)

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
STOP_TOL = 3

META = {
    "NC_002947.4": ("P. putida", "Proteobacteria"),
    "NC_002929.2": ("B. pertussis", "Proteobacteria"),
    "NC_003143.1": ("Y. pestis", "Proteobacteria"),
    "NC_003116.1": ("N. meningitidis", "Proteobacteria"),
    "NC_004757.1": ("N. europaea", "Proteobacteria"),
    "NC_008497.1": ("L. brevis", "Firmicutes"),
    "NC_004350.2": ("S. agalactiae", "Firmicutes"),
    "NC_006270.3": ("B. licheniformis", "Firmicutes"),
    "NC_006274.1": ("B. cereus", "Firmicutes"),
    "NC_003030.1": ("C. acetobutyl.", "Firmicutes"),
    "NC_003155.5": ("S. avermitilis", "Actinobacteria"),
    "NC_003450.3": ("C. glutamicum A", "Actinobacteria"),
    "NC_002677.1": ("M. leprae", "Actinobacteria"),
    "NC_008268.1": ("Rhodococcus", "Actinobacteria"),
    "NC_006958.1": ("C. glutamicum R", "Actinobacteria"),
    "NC_008818.1": ("Hyperthermus", "Archaea"),
    "NC_015948.1": ("Haloarcula", "Archaea"),
    "NC_014408.1": ("Methanobrevibacter", "Archaea"),
    "NC_019977.1": ("Methanosaeta", "Archaea"),
    "NC_007644.1": ("Moorella", "Archaea"),
}


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


SEP = "=" * 115
print(f"\n{SEP}")
print("TRAINING SET QUALITY: All 20 holdout genomes — BASELINE vs ADAPTIVE MULTI-FEATURE FILTER")
print("  Filter: self-activating when median |GC3-GC12| < 0.10 AND gc >= 0.55")
print("  Rule:   keep if (abs_bias >= 0.12 AND nc <= 3.5) OR length >= 700")
print(SEP)
print(
    f"  {'Genome':<22} {'Phylum':<16} {'GC%':>5} {'Act?':>5}  "
    f"{'n_base':>7} {'TP':>5} {'FP':>5} {'TP%':>5}  "
    f"{'n_filt':>7} {'TP':>5} {'FP':>5} {'TP%':>5} {'dTP%':>6} {'dN':>5}"
)
print("  " + "-" * 113)

rows = []
for acc in TEST_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
    label, phylum = META.get(acc, (acc, "?"))
    try:
        exact, stops = load_ref(acc)
    except Exception:
        continue

    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        orfs_l = orfs.to_dict("records") if hasattr(orfs, "to_dict") else list(orfs)
        base = create_training_set(sequence=seq, all_orfs=orfs_l)

    filt = filter_training_adaptive(base, gc)
    activated = len(filt) < len(base)

    b_tp = sum(1 for o in base if is_tp(o, exact, stops))
    b_fp = len(base) - b_tp
    b_tp_pct = 100 * b_tp / max(len(base), 1)

    f_tp = sum(1 for o in filt if is_tp(o, exact, stops))
    f_fp = len(filt) - f_tp
    f_tp_pct = 100 * f_tp / max(len(filt), 1)
    d_tp = f_tp_pct - b_tp_pct
    d_n = len(filt) - len(base)

    act_str = "YES" if activated else "no"
    d_str = f"{d_tp:+.1f}" if activated else "  ---"
    dn_str = f"{d_n:+d}" if activated else "  ---"

    print(
        f"  {label:<22} {phylum:<16} {gc*100:>5.1f} {act_str:>5}  "
        f"{len(base):>7} {b_tp:>5} {b_fp:>5} {b_tp_pct:>5.1f}%  "
        f"{len(filt):>7} {f_tp:>5} {f_fp:>5} {f_tp_pct:>5.1f}% {d_str:>6} {dn_str:>5}"
    )

    rows.append(
        dict(
            acc=acc,
            label=label,
            phylum=phylum,
            gc_pct=round(gc * 100, 1),
            activated=activated,
            base_n=len(base),
            base_tp=b_tp,
            base_fp=b_fp,
            base_tp_pct=round(b_tp_pct, 1),
            filt_n=len(filt),
            filt_tp=f_tp,
            filt_fp=f_fp,
            filt_tp_pct=round(f_tp_pct, 1),
            d_tp_pct=round(d_tp, 1),
            d_n=d_n,
        )
    )

df = pd.DataFrame(rows)
print(SEP)
print("\nPHYLUM SUMMARY:")
print(
    f"  {'Phylum':<16}  {'n':>3}  {'Base TP% mean':>14}  {'Filt TP% mean':>14}  {'Gain':>6}  {'Activated':>10}"
)
print(f"  {'-'*16}  {'-'*3}  {'-'*14}  {'-'*14}  {'-'*6}  {'-'*10}")
for ph in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
    g = df[df["phylum"] == ph]
    n_act = g["activated"].sum()
    print(
        f"  {ph:<16}  {len(g):>3}  {g['base_tp_pct'].mean():>13.1f}%  "
        f"{g['filt_tp_pct'].mean():>13.1f}%  "
        f"{g['filt_tp_pct'].mean()-g['base_tp_pct'].mean():>+5.1f}pp  "
        f"{n_act}/{len(g)} filtered"
    )

print(
    f"\nOVERALL: base={df['base_tp_pct'].mean():.1f}% -> filtered={df['filt_tp_pct'].mean():.1f}% "
    f"(+{df['filt_tp_pct'].mean()-df['base_tp_pct'].mean():.1f}pp)"
)
print(f"Activated for {df['activated'].sum()}/{len(df)} genomes")
out = OUT_DIR / "multifilter_training_sets.csv"
df.to_csv(out, index=False)
print(f"Saved: {out}")
print(SEP)
