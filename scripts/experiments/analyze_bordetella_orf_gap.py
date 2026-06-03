# EXPERIMENT: Bordetella ORF finding gap -- why are 23% of genes never found?
# STATUS: active
# RESULT: pending
"""
ML-NEW3: Diagnose why 23% of Bordetella pertussis reference genes never appear
as ORF candidates after find_orfs_candidates().

Hypotheses:
  H1 - Short genes: gene length < min_length=100bp cutoff
  H2 - Non-standard start codons: start codon not in {ATG, GTG, TTG}
  H3 - Sequencing gaps / N-runs: gene region contains ambiguous bases
  H4 - Nested overlap: gene is fully nested inside a larger gene (same stop, subset start)
  H5 - Annotation mismatch: reference CDS coordinates don't align with ORF boundaries

For each missed gene, classify it into one of these categories.
Also compares to M.leprae and clean genomes to see if this is Bordetella-specific.

Run from repo root:
    python scripts/experiments/analyze_bordetella_orf_gap.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.traditional_methods import find_orfs_candidates

DATA_DIR = get_data_dir("full_dataset")
STOPS = {"TAA", "TAG", "TGA"}
STARTS = {"ATG", "GTG", "TTG"}
MIN_LEN = 100  # production minimum ORF length

FOCUS = [
    ("NC_002929.2", "Bordetella", "problem"),
    ("NC_002677.1", "M.leprae", "problem"),
    ("NC_003155.5", "Streptomyces", "problem"),
    ("NC_003030.1", "Clostridium", "clean"),
    ("NC_004350.2", "Streptococcus", "clean"),
]

SEP = "=" * 95


def load_ref_cds(acc):
    """Load non-pseudogene CDS from GFF, return list of dicts with coords + strand."""
    r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
    cds = r[r[2] == "CDS"]
    if 8 in cds.columns:
        is_pseudo = cds[8].str.contains(
            "pseudo=true|pseudogene|frameshifted|internal.stop|disrupted", case=False, na=False
        )
        cds = cds[~is_pseudo]
    genes = []
    for _, row in cds.iterrows():
        genes.append(
            {
                "start": int(row[3]),
                "end": int(row[4]),
                "strand": "forward" if row[6] == "+" else "reverse",
                "length": int(row[4]) - int(row[3]) + 1,
            }
        )
    return genes


def get_codon(seq, pos, strand, length):
    """Extract the start codon of a gene from genome sequence (0-based pos)."""
    if strand == "forward":
        return seq[pos - 1 : pos + 2].upper() if pos >= 1 else "???"
    else:
        # Reverse strand: start codon is at the 3' end of the coordinate range
        end = pos + length - 1  # pos here is 1-based start, end is 1-based end
        codon = seq[end - 3 : end].upper()
        return codon.translate(str.maketrans("ACGT", "TGCA"))[::-1]


print(f"\n{SEP}")
print("BORDETELLA ORF FINDING GAP DIAGNOSIS")
print(f"  min_length={MIN_LEN}  start_codons={STARTS}")
print(SEP)

all_rows = []

for acc, name, category in FOCUS:
    fasta = f"{DATA_DIR}/{acc}.fasta"
    if not Path(fasta).exists():
        print(f"  SKIP {acc}")
        continue

    ref_genes = load_ref_cds(acc)
    genome = load_genome_sequence(fasta)
    seq = genome["sequence"]
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)

    print(f"\n{SEP}")
    print(f"  {name} ({acc})  GC={gc*100:.1f}%  [{category}]  ref_genes={len(ref_genes):,}")

    # Run ORF detection
    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=MIN_LEN)

    # Build lookup using genome_start/genome_end (actual genome coords for all strands).
    # The ORF finder also has 'start'/'end' which are local/reverse-complement coords
    # for reverse strand ORFs — those are wrong for GFF comparison.
    if hasattr(orfs, "to_dict"):
        gs_col = "genome_start" if "genome_start" in orfs.columns else "start"
        ge_col = "genome_end" if "genome_end" in orfs.columns else "end"
        orf_coords = {
            (min(int(s), int(e)), max(int(s), int(e))) for s, e in zip(orfs[gs_col], orfs[ge_col])
        }
    else:
        orf_coords = {
            (
                min(int(o.get("genome_start", o["start"])), int(o.get("genome_end", o["end"]))),
                max(int(o.get("genome_start", o["start"])), int(o.get("genome_end", o["end"]))),
            )
            for o in orfs
        }

    # Classify each reference gene
    found = missed_short = missed_nonstart = missed_n = missed_nested = missed_other = 0
    missed_genes = []

    for g in ref_genes:
        gs, ge = g["start"], g["end"]
        length = ge - gs + 1

        if (min(gs, ge), max(gs, ge)) in orf_coords:
            found += 1
            continue

        # MISSED — classify why
        # H1: too short
        if length < MIN_LEN:
            missed_short += 1
            missed_genes.append({**g, "reason": "H1_short", "gene_length": length})
            continue

        # H2: non-standard start codon
        codon = get_codon(seq, gs if g["strand"] == "forward" else gs, g["strand"], length)
        if codon not in STARTS:
            missed_nonstart += 1
            missed_genes.append(
                {**g, "reason": "H2_nonstart", "start_codon": codon, "gene_length": length}
            )
            continue

        # H3: N-runs in the gene body
        if g["strand"] == "forward":
            gene_seq = seq[gs - 1 : ge].upper()
        else:
            gene_seq = seq[gs - 1 : ge].upper()
        n_count = gene_seq.count("N")
        if n_count > 0:
            missed_n += 1
            missed_genes.append(
                {**g, "reason": "H3_Nrun", "n_count": n_count, "gene_length": length}
            )
            continue

        # H4: nested — check if this gene's stop is represented but different start
        gene_stop = ge if g["strand"] == "forward" else gs
        # Is any ORF ending at this stop in the candidate pool?
        gs_n, ge_n = min(gs, ge), max(gs, ge)
        same_stop = [(s, e) for s, e in orf_coords if abs(e - ge_n) <= 3 or abs(s - gs_n) <= 3]
        if same_stop:
            missed_nested += 1
            missed_genes.append(
                {
                    **g,
                    "reason": "H4_nested_or_alt_start",
                    "n_similar": len(same_stop),
                    "gene_length": length,
                }
            )
            continue

        # H5: other (annotation mismatch, unusual feature, etc.)
        missed_other += 1
        missed_genes.append({**g, "reason": "H5_other", "gene_length": length})

    n_ref = len(ref_genes)
    n_missed = n_ref - found
    print(
        f"\n  ORF detection:  {found:,}/{n_ref:,} found ({found/n_ref*100:.1f}%)  "
        f"missed: {n_missed:,} ({n_missed/n_ref*100:.1f}%)"
    )
    print(f"\n  Missed gene breakdown:")
    print(
        f"    H1 Too short (<{MIN_LEN}bp):           {missed_short:>5}  ({missed_short/n_ref*100:>5.1f}%)"
    )
    print(
        f"    H2 Non-standard start codon:   {missed_nonstart:>5}  ({missed_nonstart/n_ref*100:>5.1f}%)"
    )
    print(f"    H3 N-runs in gene body:        {missed_n:>5}  ({missed_n/n_ref*100:>5.1f}%)")
    print(
        f"    H4 Alt-start / nested:         {missed_nested:>5}  ({missed_nested/n_ref*100:>5.1f}%)"
    )
    print(
        f"    H5 Other:                      {missed_other:>5}  ({missed_other/n_ref*100:>5.1f}%)"
    )

    # Detail for H1 (short genes)
    short_genes = [g for g in missed_genes if g["reason"] == "H1_short"]
    if short_genes:
        lengths = [g["gene_length"] for g in short_genes]
        print(
            f"\n    H1 short gene lengths: min={min(lengths)} median={int(np.median(lengths))} max={max(lengths)}"
        )
        for cutoff in [50, 75, 100]:
            n_above = sum(1 for l in lengths if l >= cutoff)
            print(f"      >= {cutoff}bp: {n_above} ({n_above/n_ref*100:.1f}% of all ref)")

    # Detail for H2 (non-standard start)
    nonstart_genes = [g for g in missed_genes if g["reason"] == "H2_nonstart"]
    if nonstart_genes:
        from collections import Counter

        codon_counts = Counter(g.get("start_codon", "?") for g in nonstart_genes)
        print(f"\n    H2 non-standard start codons: {dict(codon_counts.most_common(8))}")

    # Detail for H4 (alt-start / nested)
    nested_genes = [g for g in missed_genes if g["reason"] == "H4_nested_or_alt_start"]
    if nested_genes:
        lengths = [g["gene_length"] for g in nested_genes]
        print(
            f"\n    H4 gene lengths: min={min(lengths)} median={int(np.median(lengths))} max={max(lengths)}"
        )

    all_rows.extend([{"acc": acc, "name": name, "category": category, **g} for g in missed_genes])

# Cross-genome summary
df = pd.DataFrame(all_rows)
print(f"\n{SEP}")
print("CROSS-GENOME SUMMARY: missed gene categories (%% of reference genes)")
print(
    f"  {'Genome':<15} {'Total%':>7} {'H1_short':>9} {'H2_nostart':>11} {'H3_N':>6} {'H4_alt':>7} {'H5_other':>9}"
)
print("  " + "-" * 68)
for acc, name, _ in FOCUS:
    sub = df[df["acc"] == acc]
    ref = load_ref_cds(acc)
    n = len(ref)
    if n == 0 or sub.empty:
        continue
    tot = len(sub) / n * 100
    h = {
        r: len(sub[sub["reason"] == r]) / n * 100
        for r in ["H1_short", "H2_nonstart", "H3_Nrun", "H4_nested_or_alt_start", "H5_other"]
    }
    print(
        f"  {name:<15} {tot:>7.1f}% {h['H1_short']:>9.1f}% {h['H2_nonstart']:>11.1f}% "
        f"{h['H3_Nrun']:>6.1f}% {h['H4_nested_or_alt_start']:>7.1f}% {h['H5_other']:>9.1f}%"
    )

out = Path("lgb_attribution_results") / "bordetella_orf_gap.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(SEP)
