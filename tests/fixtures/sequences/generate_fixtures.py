"""
Generate deterministic synthetic FASTA fixture files for the test suite.

Run from the repo root:
    python tests/fixtures/sequences/generate_fixtures.py

Design rules
------------
* Filler regions : "C" repeated  — contains no start codons (ATG/GTG/TTG) and
  no stop codons (TAA/TAG/TGA), so only deliberately placed ORFs are detected.
* ORF body codons: "GGG" repeated — not a start or stop codon.
* All positions are 0-indexed in this script; the coordinate table below
  records 1-based (start) and exclusive-end values as reported by
  find_orfs_candidates().
* The minimum detectable ORF length is 100 bp (the default min_length).

Coordinate reference (matches conftest.py constants)
-----------------------------------------------------
synthetic_single_orf.fasta   300 bp
  ORF1 fwd : start=31 end=180 length=150 frame=0

synthetic_multi_orf.fasta   2000 bp
  ORF1 fwd : start=1   end=150  length=150  frame=0
  ORF2 fwd : start=301 end=480  length=180  frame=0
  ORF3 fwd : start=601 end=750  length=150  frame=0
  ORF4 fwd : start=901 end=1098 length=198  frame=0
  ORF5 fwd : start=1201 end=1350 length=150 frame=0

synthetic_no_orf.fasta       500 bp
  0 ORFs (pure poly-C)

synthetic_reverse_strand.fasta  400 bp
  RC ORF  : rc_start=61 rc_end=339 length=279 frame=0
  genome_start=62 genome_end=340

synthetic_large.fasta        5000 bp
  8 fwd ORFs each 300 bp, spaced every 600 bp starting at position 0
  ORF k (k=1..8): start=600*(k-1)+1  end=600*(k-1)+300  length=300  frame=0
"""

from pathlib import Path
from Bio.Seq import Seq

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def filler(n: int) -> str:
    """n bp of poly-C (no start/stop codons)."""
    return "C" * n


def orf_body(n_internal_codons: int) -> str:
    """ATG + n_internal_codons * GGG + TAA."""
    return "ATG" + "GGG" * n_internal_codons + "TAA"


def write_fasta(path: Path, header: str, seq: str, line_width: int = 60) -> None:
    lines = [f">{header}"]
    for i in range(0, len(seq), line_width):
        lines.append(seq[i : i + line_width])
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path.name}  ({len(seq):,} bp)")


# ---------------------------------------------------------------------------
# synthetic_single_orf.fasta  — 300 bp, 1 forward ORF
# ---------------------------------------------------------------------------
# ATG at 0-indexed position 30 (frame 0: 30%3=0).
# ORF body: ATG + GGG*48 + TAA = 150 bp.
# 1-based: start=31, end=180, length=150.

single_orf_seq = filler(30) + orf_body(48) + filler(120)
assert len(single_orf_seq) == 300
write_fasta(HERE / "synthetic_single_orf.fasta", "SYNTHETIC_SINGLE_ORF length=300", single_orf_seq)


# ---------------------------------------------------------------------------
# synthetic_no_orf.fasta  — 500 bp, 0 ORFs
# ---------------------------------------------------------------------------
# Pure poly-C: no ATG/GTG/TTG anywhere → zero ORFs detected.

no_orf_seq = filler(500)
assert len(no_orf_seq) == 500
write_fasta(HERE / "synthetic_no_orf.fasta", "SYNTHETIC_NO_ORF length=500", no_orf_seq)


# ---------------------------------------------------------------------------
# synthetic_reverse_strand.fasta  — 400 bp, 1 ORF on the reverse strand only
# ---------------------------------------------------------------------------
# Design the reverse-complement sequence, then take its RC for the stored fwd
# strand.  The stored fwd strand is all G/C — no start codons on + strand.
#
# RC sequence design (read 5'→3'):
#   filler(60) + orf_body(91) + filler(61)  =  60+279+61 = 400 bp
#   ATG at rc position 60 (frame 0), stop ends at rc position 339.
#
# Genome coordinates (seq_len=400):
#   genome_start = 400 - 339 + 1 = 62
#   genome_end   = 400 - 60     = 340
#   length       = 279

rc_sequence = filler(60) + orf_body(91) + filler(61)
assert len(rc_sequence) == 400
rev_strand_seq = str(Seq(rc_sequence).reverse_complement())
write_fasta(HERE / "synthetic_reverse_strand.fasta", "SYNTHETIC_REVERSE_STRAND length=400", rev_strand_seq)


# ---------------------------------------------------------------------------
# synthetic_multi_orf.fasta  — 2000 bp, 5 forward ORFs
# ---------------------------------------------------------------------------
# All ORFs on forward strand, frame 0 (all start positions divisible by 3).
#
#  pos   0 – 149 : ORF1  ATG+GGG*48+TAA  150 bp  start=1   end=150
#  pos 150 – 299 : filler 150 bp
#  pos 300 – 479 : ORF2  ATG+GGG*58+TAA  180 bp  start=301 end=480
#  pos 480 – 599 : filler 120 bp
#  pos 600 – 749 : ORF3  ATG+GGG*48+TAA  150 bp  start=601 end=750
#  pos 750 – 899 : filler 150 bp
#  pos 900 – 1097: ORF4  ATG+GGG*64+TAA  198 bp  start=901 end=1098
#  pos 1098–1199 : filler 102 bp
#  pos 1200–1349 : ORF5  ATG+GGG*48+TAA  150 bp  start=1201 end=1350
#  pos 1350–1999 : filler 650 bp

multi_orf_seq = (
    orf_body(48)          # ORF1 150 bp  pos 0-149
    + filler(150)
    + orf_body(58)        # ORF2 180 bp  pos 300-479
    + filler(120)
    + orf_body(48)        # ORF3 150 bp  pos 600-749
    + filler(150)
    + orf_body(64)        # ORF4 198 bp  pos 900-1097
    + filler(102)
    + orf_body(48)        # ORF5 150 bp  pos 1200-1349
    + filler(650)
)
assert len(multi_orf_seq) == 2000
write_fasta(HERE / "synthetic_multi_orf.fasta", "SYNTHETIC_MULTI_ORF length=2000", multi_orf_seq)


# ---------------------------------------------------------------------------
# synthetic_large.fasta  — 5000 bp, 8 forward ORFs
# ---------------------------------------------------------------------------
# 8 ORFs of 300 bp each, one every 600 bp (300 bp ORF + 300 bp filler),
# followed by a 500 bp trailing filler.
#
# ORF k (k=1..8):
#   0-indexed start : 600*(k-1)
#   1-based start   : 600*(k-1) + 1
#   end             : 600*(k-1) + 300
#   length          : 300  (ATG + GGG*98 + TAA)
#   frame           : 0  (600*(k-1) % 3 == 0)

large_parts = []
for _ in range(8):
    large_parts.append(orf_body(98))   # 300 bp
    large_parts.append(filler(300))    # 300 bp gap
large_seq = "".join(large_parts[:-1]) + filler(500)
# Remove the trailing filler(300) added by the last iteration and replace
# with filler(500) to reach exactly 5000 bp.
# Recalculate: 8 ORFs * 300 + 7 fillers * 300 + 500 trailing = 2400+2100+500 = 5000

large_seq = ""
for k in range(8):
    large_seq += orf_body(98)   # 300 bp ORF
    if k < 7:
        large_seq += filler(300)  # 300 bp inter-ORF filler
large_seq += filler(500)           # trailing filler
assert len(large_seq) == 5000, f"Expected 5000, got {len(large_seq)}"
write_fasta(HERE / "synthetic_large.fasta", "SYNTHETIC_LARGE length=5000", large_seq)

print("Done.")
