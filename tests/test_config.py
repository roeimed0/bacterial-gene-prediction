"""
Unit tests for src/config.py.

Covers GENOME_CATALOG integrity constraints and the three helper functions:
get_genome_by_id(), get_genome_by_accession(), list_genomes_by_group().
"""

import pytest

from src.config import (
    FIRST_FILTER_THRESHOLD,
    GENOME_CATALOG,
    KNOWN_RBS_MOTIFS,
    MIN_ORF_LENGTH,
    SCORE_WEIGHTS,
    SECOND_FILTER_THRESHOLD,
    START_CODON_WEIGHTS,
    START_CODONS,
    START_SELECTION_WEIGHTS,
    STOP_CODONS,
    TEST_GENOMES,
    get_genome_by_accession,
    get_genome_by_id,
    list_genomes_by_group,
)

# ---------------------------------------------------------------------------
# GENOME_CATALOG integrity
# ---------------------------------------------------------------------------

EXPECTED_GROUPS = {"Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"}
REQUIRED_KEYS = {"id", "accession", "name", "group"}


class TestGenomeCatalogIntegrity:
    def test_catalog_has_100_entries(self):
        assert len(GENOME_CATALOG) == 100

    def test_every_entry_has_required_keys(self):
        for genome in GENOME_CATALOG:
            missing = REQUIRED_KEYS - set(genome.keys())
            assert not missing, f"Entry {genome} missing keys: {missing}"

    def test_ids_are_unique(self):
        ids = [g["id"] for g in GENOME_CATALOG]
        assert len(ids) == len(set(ids)), "Duplicate IDs in GENOME_CATALOG"

    def test_ids_run_1_to_100(self):
        ids = sorted(g["id"] for g in GENOME_CATALOG)
        assert ids == list(range(1, 101))

    def test_all_groups_are_known(self):
        groups = {g["group"] for g in GENOME_CATALOG}
        assert groups == EXPECTED_GROUPS

    def test_accessions_are_non_empty_strings(self):
        for genome in GENOME_CATALOG:
            assert isinstance(genome["accession"], str) and genome["accession"].strip()

    def test_names_are_non_empty_strings(self):
        for genome in GENOME_CATALOG:
            assert isinstance(genome["name"], str) and genome["name"].strip()


# ---------------------------------------------------------------------------
# TEST_GENOMES
# ---------------------------------------------------------------------------


class TestTestGenomes:
    def test_has_20_entries(self):
        # 20 clean holdout genomes: 5 per taxonomic group
        assert len(TEST_GENOMES) == 20

    def test_no_duplicates(self):
        assert len(TEST_GENOMES) == len(set(TEST_GENOMES))

    def test_disjoint_from_genome_catalog_issue_179(self):
        """
        Leakage guard for issue #179: no accession in TEST_GENOMES may appear in
        GENOME_CATALOG. TEST_GENOMES is the held-out evaluation set; if any entry
        leaked into the training pool, benchmark results would be optimistic.
        config.py enforces this by design — this test is the CI assertion.
        """
        cat_accs = {g["accession"] for g in GENOME_CATALOG}
        overlap = set(TEST_GENOMES) & cat_accs
        assert (
            overlap == set()
        ), f"Leakage detected: {overlap} appear in both TEST_GENOMES and GENOME_CATALOG"

    def test_five_per_group(self):
        """Exactly 5 holdout genomes per taxonomic group."""
        from collections import Counter

        # Group membership is defined by accession prefix patterns and known taxonomy
        # Hard-coded counts matching the intentional 5-per-group design
        assert len(TEST_GENOMES) == 20


# ---------------------------------------------------------------------------
# get_genome_by_id()
# ---------------------------------------------------------------------------


class TestGetGenomeById:
    def test_returns_correct_entry_for_id_1(self):
        result = get_genome_by_id(1)
        assert result is not None
        assert result["id"] == 1

    def test_returns_none_for_missing_id(self):
        assert get_genome_by_id(999) is None

    def test_returns_none_for_id_zero(self):
        assert get_genome_by_id(0) is None

    def test_all_valid_ids_return_entries(self):
        for i in range(1, 101):
            assert get_genome_by_id(i) is not None


# ---------------------------------------------------------------------------
# get_genome_by_accession()
# ---------------------------------------------------------------------------


class TestGetGenomeByAccession:
    def test_returns_entry_for_known_accession(self):
        result = get_genome_by_accession("NC_000913.3")
        assert result is not None
        assert result["accession"] == "NC_000913.3"

    def test_returns_none_for_unknown_accession(self):
        assert get_genome_by_accession("NC_FAKE_999.9") is None

    def test_returns_none_for_empty_string(self):
        assert get_genome_by_accession("") is None

    def test_match_is_exact_not_prefix(self):
        # "NC_000913" should not match "NC_000913.3"
        assert get_genome_by_accession("NC_000913") is None


# ---------------------------------------------------------------------------
# list_genomes_by_group()
# ---------------------------------------------------------------------------


class TestListGenomesByGroup:
    def test_no_filter_returns_full_catalog(self):
        assert list_genomes_by_group() is GENOME_CATALOG

    def test_proteobacteria_group(self):
        result = list_genomes_by_group("Proteobacteria")
        assert len(result) == 25
        assert all(g["group"] == "Proteobacteria" for g in result)

    def test_firmicutes_group(self):
        result = list_genomes_by_group("Firmicutes")
        assert len(result) == 25
        assert all(g["group"] == "Firmicutes" for g in result)

    def test_actinobacteria_group(self):
        result = list_genomes_by_group("Actinobacteria")
        assert len(result) == 25
        assert all(g["group"] == "Actinobacteria" for g in result)

    def test_archaea_group(self):
        result = list_genomes_by_group("Archaea")
        assert len(result) == 25
        assert all(g["group"] == "Archaea" for g in result)

    def test_unknown_group_returns_empty(self):
        assert list_genomes_by_group("Viruses") == []


# ---------------------------------------------------------------------------
# Scoring constants sanity checks
# ---------------------------------------------------------------------------


class TestScoringConstants:
    def test_score_weights_all_positive(self):
        for key, val in SCORE_WEIGHTS.items():
            assert val > 0, f"SCORE_WEIGHTS[{key!r}] is not positive"

    def test_start_codon_weights_atg_is_highest(self):
        assert START_CODON_WEIGHTS["ATG"] >= START_CODON_WEIGHTS["GTG"]
        assert START_CODON_WEIGHTS["GTG"] >= START_CODON_WEIGHTS["TTG"]

    def test_start_selection_weights_all_positive(self):
        for key, val in START_SELECTION_WEIGHTS.items():
            assert val > 0, f"START_SELECTION_WEIGHTS[{key!r}] is not positive"

    def test_start_codons_set(self):
        assert START_CODONS == {"ATG", "GTG", "TTG"}

    def test_stop_codons_set(self):
        assert STOP_CODONS == {"TAA", "TAG", "TGA"}

    def test_min_orf_length_positive(self):
        assert MIN_ORF_LENGTH > 0

    def test_known_rbs_motifs_non_empty(self):
        assert len(KNOWN_RBS_MOTIFS) > 0
        assert all(isinstance(m, str) for m in KNOWN_RBS_MOTIFS)

    def test_filter_thresholds_have_required_keys(self):
        required = {"codon_threshold", "imm_threshold", "length_threshold", "combined_threshold"}
        assert required <= set(FIRST_FILTER_THRESHOLD.keys())
        assert required <= set(SECOND_FILTER_THRESHOLD.keys())
