"""
Shared fixtures for the bacterial gene prediction test suite.

Ground-truth constants for synthetic fixtures are defined here as module-level
constants so that test files can import them directly instead of embedding
magic numbers inline.

Coordinate conventions (match find_orfs_candidates() output)
-------------------------------------------------------------
* start  — 1-based nucleotide position of the ATG (= 0-based index + 1)
* end    — 0-based exclusive position after the stop codon (= i + 3)
* length — end - (start - 1)  ==  orf_body length in bp
* strand — "forward" or "reverse" (as returned by the detector)
"""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sequences"

# ---------------------------------------------------------------------------
# synthetic_single_orf.fasta — 300 bp, one ORF on the forward strand
# ATG at 0-indexed position 30 (frame 0); body = ATG + GGG*48 + TAA = 150 bp
# ---------------------------------------------------------------------------
SINGLE_ORF_START = 31       # 1-based
SINGLE_ORF_END = 180        # exclusive end (i + 3)
SINGLE_ORF_LENGTH = 150
SINGLE_ORF_STRAND = "forward"

# ---------------------------------------------------------------------------
# synthetic_multi_orf.fasta — 2000 bp, five forward-strand ORFs (all frame 0)
# ---------------------------------------------------------------------------
MULTI_ORF_COORDS = [
    {"start": 1,    "end": 150,  "length": 150, "strand": "forward"},
    {"start": 301,  "end": 480,  "length": 180, "strand": "forward"},
    {"start": 601,  "end": 750,  "length": 150, "strand": "forward"},
    {"start": 901,  "end": 1098, "length": 198, "strand": "forward"},
    {"start": 1201, "end": 1350, "length": 150, "strand": "forward"},
]

# ---------------------------------------------------------------------------
# synthetic_reverse_strand.fasta — 400 bp, one ORF on the reverse strand only
# RC-sequence ORF: rc_start=61 rc_end=339 length=279 frame=0
# Genome coords: genome_start=62 genome_end=340
# ---------------------------------------------------------------------------
REVERSE_ORF_GENOME_START = 62
REVERSE_ORF_GENOME_END = 340
REVERSE_ORF_LENGTH = 279
REVERSE_ORF_STRAND = "reverse"

# ---------------------------------------------------------------------------
# synthetic_large.fasta — 5000 bp, 8 forward-strand ORFs each 300 bp
# ORF k (k=1..8): start=600*(k-1)+1  end=600*(k-1)+300  length=300  frame=0
# ---------------------------------------------------------------------------
LARGE_ORF_COUNT = 8
LARGE_ORF_LENGTH = 300
LARGE_ORF_COORDS = [
    {"start": 600 * k + 1, "end": 600 * k + 300, "length": 300, "strand": "forward"}
    for k in range(8)
]

# ---------------------------------------------------------------------------
# Minimal ORF dict — used to build group fixtures for OrfGroupClassifier tests
# ---------------------------------------------------------------------------

_BASE_ORF = {
    "combined_score":    0.8,
    "rbs_score":         2.0,
    "codon_score":       0.5,
    "start_score":       0.7,
    "imm_score":         0.3,
    "strand":            "+",
    # normalized scores used by start-selection weighting
    "codon_score_norm":  0.6,
    "imm_score_norm":    0.4,
    "rbs_score_norm":    0.7,
    "length_score_norm": 0.5,
    "start_score_norm":  0.6,
}

# ---------------------------------------------------------------------------
# Minimal candidate dict — used for HybridGeneFilter tests
# The sequence is 27 bp: ATG start, TAA stop, in-frame.
# ---------------------------------------------------------------------------

_BASE_CANDIDATE = {
    "sequence":          "ATGAAACCCGGGTTTTTTGGGAAATAA",
    "length":            27,
    "start_codon":       "ATG",
    "codon_score_norm":  0.6,
    "imm_score_norm":    0.4,
    "rbs_score_norm":    0.7,
    "length_score_norm": 0.5,
    "start_score_norm":  0.6,
    "combined_score":    0.8,
    "rbs_score":         2.0,
    "gene_id":           "gene_1",
}


@pytest.fixture
def synthetic_orf():
    """Return a copy of the base ORF dict so tests can mutate it freely."""
    return dict(_BASE_ORF)


@pytest.fixture
def synthetic_candidate():
    """Return a copy of the base candidate dict so tests can mutate it freely."""
    return dict(_BASE_CANDIDATE)


@pytest.fixture
def single_orf_group(synthetic_orf):
    """
    ORF group containing exactly one ORF.

    This is the edge case for ``combined_margin_top2``, which requires at
    least two ORFs to compute a margin.  The fixture is used to verify that
    ``extract_group_features()`` handles single-member groups without error.
    """
    return {"group_1": [synthetic_orf]}


@pytest.fixture
def two_orf_group(synthetic_orf):
    """ORF group with two members — the normal case for group-level features."""
    orf_low = dict(synthetic_orf)
    orf_low["combined_score"] = 0.4
    return {"group_1": [synthetic_orf, orf_low]}
