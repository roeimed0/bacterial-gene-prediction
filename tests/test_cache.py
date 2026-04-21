"""
Unit tests for src/cache.py.

All tests use a temporary directory and mock out network calls and ORF
detection so the cache layer can be exercised in isolation — no NCBI
downloads, no genome files needed.
"""

import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.cache import (
    add_genome_to_cache,
    cache_stats,
    clear_cache,
    get_cache_file,
    get_cached_genome,
    load_cache,
    precompute_genomes,
    save_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache_entry(n_orfs: int = 3) -> dict:
    """Return a minimal cached genome entry."""
    return {
        "sequence": "ATGCGTAA" * 10,
        "orfs": [{"start": i * 10, "end": i * 10 + 9} for i in range(n_orfs)],
    }


# ---------------------------------------------------------------------------
# get_cache_file()
# ---------------------------------------------------------------------------


class TestGetCacheFile:
    def test_returns_string(self):
        result = get_cache_file()
        assert isinstance(result, str)

    def test_ends_with_cache_filename(self):
        from src.config import CACHE_FILENAME
        assert get_cache_file().endswith(CACHE_FILENAME)


# ---------------------------------------------------------------------------
# save_cache() / load_cache()
# ---------------------------------------------------------------------------


class TestSaveAndLoadCache:
    def test_round_trip(self, tmp_path):
        cache_file = str(tmp_path / "test_cache.pkl")
        data = {"genome_A": _make_cache_entry(5)}

        with patch("src.cache.get_cache_file", return_value=cache_file):
            save_cache(data)
            loaded = load_cache()

        assert loaded == data

    def test_load_returns_empty_dict_when_no_file(self, tmp_path):
        missing_path = str(tmp_path / "nonexistent.pkl")
        with patch("src.cache.get_cache_file", return_value=missing_path):
            result = load_cache()
        assert result == {}

    def test_save_creates_parent_directories(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "cache.pkl")
        with patch("src.cache.get_cache_file", return_value=nested):
            save_cache({"g": _make_cache_entry()})
        assert os.path.exists(nested)

    def test_save_overwrites_existing(self, tmp_path):
        cache_file = str(tmp_path / "cache.pkl")
        with patch("src.cache.get_cache_file", return_value=cache_file):
            save_cache({"g1": _make_cache_entry()})
            save_cache({"g2": _make_cache_entry(10)})
            loaded = load_cache()
        assert "g1" not in loaded
        assert "g2" in loaded


# ---------------------------------------------------------------------------
# get_cached_genome()
# ---------------------------------------------------------------------------


class TestGetCachedGenome:
    def test_returns_entry_for_known_genome(self):
        entry = _make_cache_entry(4)
        result = get_cached_genome("NC_TEST", cached_data={"NC_TEST": entry})
        assert result == entry

    def test_raises_key_error_for_unknown_genome(self):
        with pytest.raises(KeyError, match="NC_MISSING"):
            get_cached_genome("NC_MISSING", cached_data={})

    def test_key_error_message_contains_genome_id(self):
        genome_id = "NC_000913.3"
        with pytest.raises(KeyError) as exc_info:
            get_cached_genome(genome_id, cached_data={})
        assert genome_id in str(exc_info.value)


# ---------------------------------------------------------------------------
# clear_cache()
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_deletes_existing_cache_file(self, tmp_path):
        cache_file = str(tmp_path / "cache.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump({}, f)
        assert os.path.exists(cache_file)

        with patch("src.cache.get_cache_file", return_value=cache_file):
            clear_cache()

        assert not os.path.exists(cache_file)

    def test_no_error_when_cache_does_not_exist(self, tmp_path):
        missing = str(tmp_path / "no_cache.pkl")
        with patch("src.cache.get_cache_file", return_value=missing):
            clear_cache()  # should not raise


# ---------------------------------------------------------------------------
# add_genome_to_cache()
# ---------------------------------------------------------------------------


class TestAddGenomeToCache:
    def test_skips_already_cached_genome(self):
        cached = {"NC_001": _make_cache_entry()}
        was_added = add_genome_to_cache(
            "NC_001", email="test@test.com", cached_data=cached
        )
        assert was_added is False
        assert len(cached) == 1

    def test_adds_new_genome(self):
        mock_orfs = [{"start": 0, "end": 99}]
        cached = {}

        with (
            patch("src.cache.download_genome_and_reference", return_value=("f.fasta", "f.gff")),
            patch("src.cache.load_genome_sequence", return_value={"sequence": "ATGCGTAA"}),
            patch("src.cache.find_orfs_candidates", return_value=mock_orfs),
        ):
            was_added = add_genome_to_cache("NC_NEW", email="t@t.com", cached_data=cached)

        assert was_added is True
        assert "NC_NEW" in cached
        assert cached["NC_NEW"]["orfs"] == mock_orfs

    def test_uses_min_orf_length_from_config_when_none_given(self):
        from src.config import MIN_ORF_LENGTH

        captured = {}

        def fake_find_orfs(seq, min_length=100):
            captured["min_length"] = min_length
            return []

        with (
            patch("src.cache.download_genome_and_reference", return_value=("f.fasta", "f.gff")),
            patch("src.cache.load_genome_sequence", return_value={"sequence": "ATGCGTAA"}),
            patch("src.cache.find_orfs_candidates", side_effect=fake_find_orfs),
        ):
            add_genome_to_cache("NC_X", email="t@t.com", cached_data={}, min_length=None)

        assert captured["min_length"] == MIN_ORF_LENGTH


# ---------------------------------------------------------------------------
# precompute_genomes()
# ---------------------------------------------------------------------------


class TestPrecomputeGenomes:
    def test_returns_populated_cache(self):
        mock_orfs = [{"start": 0, "end": 99}]

        with (
            patch("src.cache.download_genome_and_reference", return_value=("f.fasta", "f.gff")),
            patch("src.cache.load_genome_sequence", return_value={"sequence": "ATGCGTAA"}),
            patch("src.cache.find_orfs_candidates", return_value=mock_orfs),
            patch("src.cache.save_cache"),
        ):
            result = precompute_genomes(["NC_A", "NC_B"], email="t@t.com", cached_data={})

        assert "NC_A" in result
        assert "NC_B" in result

    def test_skips_already_cached_genomes(self):
        existing = {"NC_A": _make_cache_entry()}

        with (
            patch("src.cache.download_genome_and_reference") as mock_dl,
            patch("src.cache.save_cache"),
        ):
            precompute_genomes(["NC_A"], email="t@t.com", cached_data=existing)

        mock_dl.assert_not_called()

    def test_empty_genome_list_returns_existing_cache(self):
        existing = {"NC_A": _make_cache_entry()}
        with patch("src.cache.save_cache"):
            result = precompute_genomes([], email="t@t.com", cached_data=existing)
        assert result == existing


# ---------------------------------------------------------------------------
# cache_stats()
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_does_not_raise_on_empty_cache(self, capsys):
        cache_stats(cached_data={})

    def test_does_not_raise_on_populated_cache(self, capsys):
        data = {
            "NC_A": _make_cache_entry(10),
            "NC_B": _make_cache_entry(5),
        }
        cache_stats(cached_data=data)
