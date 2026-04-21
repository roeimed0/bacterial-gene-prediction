"""
Unit tests for src/data_management.py.

All NCBI (Entrez) calls are mocked.  File-reading tests use temporary FASTA
and GFF files written by the tests themselves.
"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_management import (
    get_data_dir,
    get_fasta_path,
    get_gff_path,
    get_project_root,
    get_reference_orfs_from_gff,
    load_genome_sequence,
    load_reference_genes_from_gff,
)

# ---------------------------------------------------------------------------
# Synthetic file content
# ---------------------------------------------------------------------------

_FASTA_CONTENT = textwrap.dedent("""\
    >NC_TEST.1 Test genome
    ATGCGTAAATTTGGGCCCATG
""")

_GFF_CDS = textwrap.dedent("""\
    ##gff-version 3
    NC_TEST\t.\tCDS\t100\t300\t.\t+\t0\t.
    NC_TEST\t.\tCDS\t500\t700\t.\t-\t0\t.
    NC_TEST\t.\tgene\t50\t350\t.\t+\t0\t.
""")

_GFF_GENE_ONLY = textwrap.dedent("""\
    ##gff-version 3
    NC_TEST\t.\tgene\t100\t300\t.\t+\t0\t.
    NC_TEST\t.\tgene\t500\t700\t.\t-\t0\t.
""")

_GFF_DUPLICATES = textwrap.dedent("""\
    ##gff-version 3
    NC_TEST\t.\tCDS\t100\t300\t.\t+\t0\t.
    NC_TEST\t.\tCDS\t100\t300\t.\t+\t0\t.
    NC_TEST\t.\tCDS\t500\t700\t.\t-\t0\t.
""")


# ---------------------------------------------------------------------------
# get_project_root()
# ---------------------------------------------------------------------------


class TestGetProjectRoot:
    def test_returns_path_object(self):
        assert isinstance(get_project_root(), Path)

    def test_contains_src_directory(self):
        root = get_project_root()
        assert (root / "src").is_dir()


# ---------------------------------------------------------------------------
# get_data_dir()
# ---------------------------------------------------------------------------


class TestGetDataDir:
    def test_returns_string(self):
        result = get_data_dir("processed")
        assert isinstance(result, str)

    def test_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.data_management.get_project_root", lambda: tmp_path
        )
        result = get_data_dir("test_subdir")
        assert Path(result).is_dir()

    def test_subdir_is_in_path(self):
        result = get_data_dir("full_dataset")
        assert "full_dataset" in result


# ---------------------------------------------------------------------------
# get_gff_path() / get_fasta_path()
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_get_gff_path_ends_with_gff(self):
        path = get_gff_path("NC_000913.3")
        assert path.endswith("NC_000913.3.gff")

    def test_get_fasta_path_ends_with_fasta(self):
        path = get_fasta_path("NC_000913.3")
        assert path.endswith("NC_000913.3.fasta")

    def test_get_gff_path_contains_accession(self):
        assert "NC_000913.3" in get_gff_path("NC_000913.3")

    def test_get_fasta_path_contains_accession(self):
        assert "NC_000913.3" in get_fasta_path("NC_000913.3")


# ---------------------------------------------------------------------------
# load_genome_sequence()
# ---------------------------------------------------------------------------


class TestLoadGenomeSequence:
    def test_returns_dict_with_required_keys(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(_FASTA_CONTENT)

        result = load_genome_sequence(str(fasta))

        assert result is not None
        for key in ("accession", "description", "length", "sequence"):
            assert key in result, f"Missing key: {key}"

    def test_sequence_is_uppercase(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">TEST\natgcgt\n")

        result = load_genome_sequence(str(fasta))
        assert result["sequence"] == result["sequence"].upper()

    def test_length_matches_sequence(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(_FASTA_CONTENT)

        result = load_genome_sequence(str(fasta))
        assert result["length"] == len(result["sequence"])

    def test_returns_none_for_missing_file(self):
        result = load_genome_sequence("/nonexistent/path/genome.fasta")
        assert result is None

    def test_accession_matches_fasta_id(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">NC_TEST.1 Some description\nATGCGT\n")

        result = load_genome_sequence(str(fasta))
        assert result["accession"] == "NC_TEST.1"


# ---------------------------------------------------------------------------
# load_reference_genes_from_gff()
# ---------------------------------------------------------------------------


class TestLoadReferenceGenesFromGff:
    def test_returns_set_of_tuples(self, tmp_path):
        gff = tmp_path / "ref.gff"
        gff.write_text(_GFF_CDS)

        result = load_reference_genes_from_gff(str(gff))
        assert isinstance(result, set)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_cds_features_preferred_over_gene(self, tmp_path):
        gff = tmp_path / "ref.gff"
        gff.write_text(_GFF_CDS)

        result = load_reference_genes_from_gff(str(gff))
        # GFF has 2 CDS and 1 gene; should return only 2 (the CDS)
        assert (100, 300) in result
        assert (500, 700) in result
        assert len(result) == 2

    def test_falls_back_to_gene_when_no_cds(self, tmp_path):
        gff = tmp_path / "ref.gff"
        gff.write_text(_GFF_GENE_ONLY)

        result = load_reference_genes_from_gff(str(gff))
        assert (100, 300) in result
        assert (500, 700) in result

    def test_duplicates_removed(self, tmp_path):
        gff = tmp_path / "ref.gff"
        gff.write_text(_GFF_DUPLICATES)

        result = load_reference_genes_from_gff(str(gff))
        # (100, 300) appears twice in the file; should appear once in result
        assert len(result) == 2

    def test_returns_empty_set_on_missing_file(self):
        result = load_reference_genes_from_gff("/nonexistent/ref.gff")
        assert result == set()


# ---------------------------------------------------------------------------
# get_reference_orfs_from_gff()
# ---------------------------------------------------------------------------


class TestGetReferenceOrfsFromGff:
    def test_returns_list_of_dicts(self, tmp_path):
        gff = tmp_path / "ref.gff"
        gff.write_text(_GFF_CDS)

        result = get_reference_orfs_from_gff(str(gff))
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_each_entry_has_required_keys(self, tmp_path):
        gff = tmp_path / "ref.gff"
        gff.write_text(_GFF_CDS)

        result = get_reference_orfs_from_gff(str(gff))
        for entry in result:
            for key in ("start", "end", "strand", "length"):
                assert key in entry, f"Missing key: {key}"

    def test_length_is_end_minus_start_plus_one(self, tmp_path):
        gff = tmp_path / "ref.gff"
        gff.write_text(_GFF_CDS)

        result = get_reference_orfs_from_gff(str(gff))
        for entry in result:
            assert entry["length"] == entry["end"] - entry["start"] + 1

    def test_extracts_only_cds_features(self, tmp_path):
        gff = tmp_path / "ref.gff"
        gff.write_text(_GFF_CDS)

        result = get_reference_orfs_from_gff(str(gff))
        # _GFF_CDS has 2 CDS and 1 gene; should return only the 2 CDS
        assert len(result) == 2

    def test_returns_empty_list_on_missing_file(self):
        result = get_reference_orfs_from_gff("/nonexistent/ref.gff")
        assert result == []
