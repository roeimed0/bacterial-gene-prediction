"""
Shared fixtures for the bacterial gene prediction test suite.

Ground-truth constants for synthetic fixtures are defined here as module-level
constants so that test files can import them directly instead of embedding
magic numbers inline.
"""

import pytest

# ---------------------------------------------------------------------------
# Ground-truth constants for synthetic sequence fixtures
# ---------------------------------------------------------------------------

# synthetic_single_orf.fasta — 300 bp, one ORF on the + strand
SINGLE_ORF_START = 30
SINGLE_ORF_STOP = 180
SINGLE_ORF_STRAND = "+"

# synthetic_multi_orf.fasta — 2000 bp, five ORFs with known coordinates
MULTI_ORF_COORDS = [
    {"start": 10,   "stop": 160,  "strand": "+"},
    {"start": 200,  "stop": 500,  "strand": "+"},
    {"start": 600,  "stop": 900,  "strand": "-"},
    {"start": 1000, "stop": 1300, "strand": "+"},
    {"start": 1500, "stop": 1800, "strand": "-"},
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
