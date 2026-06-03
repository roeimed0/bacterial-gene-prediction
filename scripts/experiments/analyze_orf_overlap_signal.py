# EXPERIMENT: ORF overlap density as training set FP signal
# STATUS: concluded
# RESULT: REJECTED. TPs have MORE overlap than FPs (real gene loci attract more
#   alternative start candidates). Signal is backwards — cannot be used as filter.
#   grp_size (same stop group) also shows TPs in larger groups. "Take longest"
#   produces 0% change since Glimmer already selects longest at each stop.
#   FP categories: 0% wrong-start (already TP), 14-16% near-stop, 21-36%
#   gene-fragment, 33-39% neighbor, 14-31% intergenic.
"""
Hypothesis: FP training ORFs (pseudogenes, IS elements) sit in genomic regions
with more competing candidate ORFs than TP training ORFs.

De novo signal: for each training ORF, count how many OTHER candidate ORFs
(from all_orfs, ANY strand/frame) overlap it. High overlap density = ambiguous
region = potentially non-coding.

Also tests:
  - Same-stop-codon group size: TPs in groups with fewer alternatives?
  - Upstream ORF density: is the training ORF flanked by many ORFs?
  - Is the training ORF the LONGEST in its stop-codon group?

All signals are de novo (require only genome sequence, no trained model).
"""

import bisect
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import (
    create_training_set,
    find_orfs_candidates,
    organize_nested_orfs,
)

DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)
STOP_TOL = 3

FOCUS_GENOMES = [
    ("NC_002677.1", "M. leprae", "Actinobacteria", 41.74),
    ("NC_002929.2", "B. pertussis", "Proteobacteria", 60.06),
    ("NC_008818.1", "Hyperthermus", "Archaea", 60.88),
    ("NC_008268.1", "Rhodococcus", "Actinobacteria", 65.34),
    ("NC_003155.5", "S. avermitilis", "Actinobacteria", 66.34),
    ("NC_003030.1", "C. acetobutyl.", "Firmicutes", 88.01),
    ("NC_004350.2", "S. agalactiae", "Firmicutes", 87.08),
]

THRESHOLDS = [2, 3, 5, 8, 10, 15, 20]  # max competing ORFs to keep


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


def count_overlapping_orfs(orf, all_orf_intervals):
    """Count how many ORFs from all_orfs overlap this training ORF."""
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    # all_orf_intervals is sorted list of (start, end)
    # count overlaps: NOT (end < gs OR start > ge)
    count = 0
    i = bisect.bisect_left(all_orf_intervals, (gs,))
    # check backward
    j = i - 1
    while j >= 0:
        s, e = all_orf_intervals[j]
        if e < gs:
            break
        if s != gs or e != ge:  # exclude self
            count += 1
        j -= 1
    # check forward
    j = i
    while j < len(all_orf_intervals):
        s, e = all_orf_intervals[j]
        if s > ge:
            break
        if s != gs or e != ge:
            count += 1
        j += 1
    return count


def group_size(orf, groups):
    """How many ORFs share this ORF's stop codon group?"""
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    if gs > ge:
        gs, ge = ge, gs
    strand = orf.get("strand", "forward")
    key = (strand, ge)
    g = groups.get(key)
    if g is None:
        return 1
    if isinstance(g, pd.DataFrame):
        return len(g)
    return len(g)


SEP = "=" * 90
print(f"\n{SEP}")
print("DIAGNOSTIC: ORF Overlap Density as Training Set FP Signal")
print("  Hypothesis: FPs in regions with more competing ORF candidates than TPs")
print(SEP)

all_rows = []

for acc, label, phylum, f1 in FOCUS_GENOMES:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        continue
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
        training = create_training_set(sequence=seq, all_orfs=orfs_l)
        # Group raw ORFs by stop codon — need DataFrame
        orfs_df = orfs if hasattr(orfs, "groupby") else pd.DataFrame(orfs_l)
        groups = organize_nested_orfs(orfs_df)

    # Build sorted interval list from ALL ORFs (for overlap counting)
    all_intervals = sorted(
        set(
            (
                int(o.get("genome_start", o.get("start", 0))),
                int(o.get("genome_end", o.get("end", 0))),
            )
            for o in orfs_l
            if int(o.get("genome_start", o.get("start", 0)))
            <= int(o.get("genome_end", o.get("end", 0)))
        )
    )

    # Also build stop-codon groups dict
    stop_groups = {}
    for gid, gdf in groups.items():
        if isinstance(gdf, pd.DataFrame):
            ge_key = gdf.iloc[0].get("genome_end", gdf.iloc[0].get("end", 0))
            st_key = gdf.iloc[0].get("strand", "forward")
            stop_groups[(st_key, int(ge_key))] = gdf

    print(f"\n  {acc}  {label:<20}  [{phylum}]  F1={f1:.1f}%  gc={gc*100:.1f}%")

    tp_overlap, fp_overlap = [], []
    tp_grp_size, fp_grp_size = [], []
    tp_is_longest, fp_is_longest = [], []

    for orf in training:
        tp = is_tp(orf, exact, stops)
        n_overlap = count_overlapping_orfs(orf, all_intervals)
        gs = int(orf.get("genome_start", orf.get("start", 0)))
        ge = int(orf.get("genome_end", orf.get("end", 0)))
        if gs > ge:
            gs, ge = ge, gs
        length = ge - gs
        strand = orf.get("strand", "forward")
        grp = stop_groups.get((strand, ge))
        if grp is not None and isinstance(grp, pd.DataFrame):
            grp_size = len(grp)
            is_longest = int(length == grp["length"].max())
        else:
            grp_size = 1
            is_longest = 1

        if tp:
            tp_overlap.append(n_overlap)
            tp_grp_size.append(grp_size)
            tp_is_longest.append(is_longest)
        else:
            fp_overlap.append(n_overlap)
            fp_grp_size.append(grp_size)
            fp_is_longest.append(is_longest)

        all_rows.append(
            {
                "acc": acc,
                "label": label,
                "phylum": phylum,
                "f1": f1,
                "cls": "TP" if tp else "FP",
                "n_overlap": n_overlap,
                "grp_size": grp_size,
                "is_longest": is_longest,
                "length": length,
            }
        )

    n_tp, n_fp = len(tp_overlap), len(fp_overlap)

    print(f"    n_train={n_tp+n_fp}  TP={n_tp} ({100*n_tp/(n_tp+n_fp):.0f}%)  FP={n_fp}")

    # Signal 1: overlap count
    if tp_overlap and fp_overlap:
        _, p1 = mannwhitneyu(tp_overlap, fp_overlap, alternative="two-sided")
        sep1 = np.median(fp_overlap) - np.median(tp_overlap)
        print(
            f"    OVERLAP:   TP median={np.median(tp_overlap):.1f}  FP median={np.median(fp_overlap):.1f}  "
            f"sep={sep1:+.1f}  p={p1:.4f}  "
            f"{'SIGNAL' if abs(sep1)>1 and p1<0.05 else 'weak'}"
        )

    # Signal 2: group size (same stop codon)
    if tp_grp_size and fp_grp_size:
        _, p2 = mannwhitneyu(tp_grp_size, fp_grp_size, alternative="two-sided")
        sep2 = np.median(fp_grp_size) - np.median(tp_grp_size)
        print(
            f"    GRP_SIZE:  TP median={np.median(tp_grp_size):.1f}  FP median={np.median(fp_grp_size):.1f}  "
            f"sep={sep2:+.1f}  p={p2:.4f}  "
            f"{'SIGNAL' if abs(sep2)>0.5 and p2<0.05 else 'weak'}"
        )

    # Signal 3: is_longest in group
    tp_longest_frac = sum(tp_is_longest) / max(len(tp_is_longest), 1)
    fp_longest_frac = sum(fp_is_longest) / max(len(fp_is_longest), 1)
    print(
        f"    IS_LONGEST: TP frac={tp_longest_frac:.2f}  FP frac={fp_longest_frac:.2f}  "
        f"sep={tp_longest_frac-fp_longest_frac:+.2f}  "
        f"{'SIGNAL' if abs(tp_longest_frac-fp_longest_frac)>0.1 else 'weak'}"
    )

    # Threshold sweep: filter training by n_overlap <= T
    print(f"\n    Threshold sweep (keep training ORFs with n_overlap <= T):")
    print(f"    {'T':>4} {'n_keep':>7} {'tp_ret':>8} {'fp_rem':>8}  Decision")
    for T in THRESHOLDS:
        tp_k = sum(1 for v in tp_overlap if v <= T)
        fp_k = sum(1 for v in fp_overlap if v <= T)
        n_k = tp_k + fp_k
        tr = 100 * tp_k / max(n_tp, 1)
        fr = 100 * (n_fp - fp_k) / max(n_fp, 1)
        ok = (tr >= 80) and (fr >= 40) and (n_k >= 350)
        print(
            f"    {T:>4} {n_k:>7} {tr:>8.1f}% {fr:>8.1f}%  "
            f"{'PASS' if ok else 'PARTIAL' if tr>=80 or fr>=40 else 'FAIL'}"
        )

df = pd.DataFrame(all_rows)
print(f"\n{SEP}")
print("CROSS-GENOME: Which signal is most consistent?")
print(SEP)

for col, direction in [
    ("n_overlap", "FP > TP"),
    ("grp_size", "FP > TP"),
    ("is_longest", "TP > FP"),
]:
    n_sig = 0
    for acc, label, _, _ in FOCUS_GENOMES:
        g = df[df["acc"] == acc]
        tp = g[g["cls"] == "TP"][col].values
        fp = g[g["cls"] == "FP"][col].values
        if len(tp) < 2 or len(fp) < 2:
            continue
        _, p = mannwhitneyu(tp, fp, alternative="two-sided")
        sep = (
            np.median(fp) - np.median(tp)
            if direction == "FP > TP"
            else np.median(tp) - np.median(fp)
        )
        if sep > 0.5 and p < 0.05:
            n_sig += 1
    print(f"  {col:<15} significant in {n_sig}/{len(FOCUS_GENOMES)} genomes")

out = OUT_DIR / "orf_overlap_diagnostic.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
