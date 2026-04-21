"""
Unit tests for src/comparative_analysis.py.

Focus: metric formulas in compare_orfs_to_reference() and the GFF-parsing
logic in compare_results_file_to_reference().  All file I/O and NCBI calls
are exercised through temporary files — no network access required.
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from src.comparative_analysis import (
    compare_orfs_to_reference,
    compare_results_file_to_reference,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_GFF_HEADER = "##gff-version 3\n"

_GFF_LINES = textwrap.dedent("""\
    ##gff-version 3
    NC_TEST\t.\tCDS\t100\t300\t.\t+\t0\t.
    NC_TEST\t.\tCDS\t500\t700\t.\t+\t0\t.
    NC_TEST\t.\tCDS\t900\t1100\t.\t-\t0\t.
""")

# Three reference genes: (100,300), (500,700), (900,1100)
_REF_COORDS = [(100, 300), (500, 700), (900, 1100)]


def _make_gff_file(tmp_path: Path, content: str = _GFF_LINES) -> str:
    """Write a minimal GFF3 file and return its path."""
    gff = tmp_path / "test.gff"
    gff.write_text(content)
    return str(gff)


def _orfs_from_coords(coords):
    """Convert (start, end) pairs to minimal ORF dicts."""
    return [{"start": s, "end": e, "strand": "+"} for s, e in coords]


# ---------------------------------------------------------------------------
# compare_orfs_to_reference — metric formula verification
# ---------------------------------------------------------------------------


class TestCompareOrfsToReference:
    def test_perfect_match_gives_100_sensitivity_and_precision(self, tmp_path):
        gff = _make_gff_file(tmp_path)
        orfs = _orfs_from_coords(_REF_COORDS)

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            result = compare_orfs_to_reference(orfs, "NC_TEST")

        assert result["true_positives"] == 3
        assert result["false_negatives"] == 0
        assert result["false_positives"] == 0
        assert result["sensitivity"] == pytest.approx(100.0)
        assert result["precision"] == pytest.approx(100.0)
        assert result["f1_score"] == pytest.approx(100.0)

    def test_no_overlap_gives_zero_sensitivity_and_precision(self, tmp_path):
        gff = _make_gff_file(tmp_path)
        orfs = _orfs_from_coords([(1, 50), (2000, 2100)])  # none match

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            result = compare_orfs_to_reference(orfs, "NC_TEST")

        assert result["true_positives"] == 0
        assert result["sensitivity"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(0.0)
        assert result["f1_score"] == pytest.approx(0.0)

    def test_partial_match_correct_tp_fp_fn(self, tmp_path):
        gff = _make_gff_file(tmp_path)
        # predict 2 of 3 reference genes + 1 wrong prediction
        orfs = _orfs_from_coords([(100, 300), (500, 700), (9999, 10000)])

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            result = compare_orfs_to_reference(orfs, "NC_TEST")

        assert result["true_positives"] == 2
        assert result["false_positives"] == 1   # (9999, 10000) is spurious
        assert result["false_negatives"] == 1   # (900, 1100) was missed

    def test_sensitivity_formula(self, tmp_path):
        """Sensitivity = TP / reference_count * 100."""
        gff = _make_gff_file(tmp_path)
        orfs = _orfs_from_coords([(100, 300)])  # 1 of 3 reference genes

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            result = compare_orfs_to_reference(orfs, "NC_TEST")

        expected = 1 / 3 * 100
        assert result["sensitivity"] == pytest.approx(expected)

    def test_precision_formula(self, tmp_path):
        """Precision = TP / predicted_count * 100."""
        gff = _make_gff_file(tmp_path)
        # predict 1 correct + 1 wrong out of 2 predictions
        orfs = _orfs_from_coords([(100, 300), (9999, 10000)])

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            result = compare_orfs_to_reference(orfs, "NC_TEST")

        expected = 1 / 2 * 100
        assert result["precision"] == pytest.approx(expected)

    def test_f1_formula(self, tmp_path):
        """F1 = 2 * P * S / (P + S)."""
        gff = _make_gff_file(tmp_path)
        orfs = _orfs_from_coords([(100, 300)])  # 1/3 sensitivity, 1/1 precision

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            result = compare_orfs_to_reference(orfs, "NC_TEST")

        s = result["sensitivity"]
        p = result["precision"]
        expected_f1 = 2 * p * s / (p + s)
        assert result["f1_score"] == pytest.approx(expected_f1)

    def test_empty_orf_list_raises_value_error(self, tmp_path):
        # Current behaviour: pd.DataFrame([]) has no columns, triggering the
        # coord-column check.  Ideal behaviour would be to return 0.0 precision
        # (see issue #29 for the planned fix).
        gff = _make_gff_file(tmp_path)
        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            with pytest.raises(ValueError):
                compare_orfs_to_reference([], "NC_TEST")

    def test_empty_reference_gff_raises_empty_data_error(self, tmp_path):
        # Current behaviour: pandas raises EmptyDataError on a header-only GFF.
        # Ideal behaviour would be to return sensitivity=0.0 (see issue #29).
        import pandas as pd
        empty_gff = tmp_path / "empty.gff"
        empty_gff.write_text(_GFF_HEADER)
        orfs = _orfs_from_coords([(100, 300)])

        with patch("src.comparative_analysis.get_gff_path", return_value=str(empty_gff)):
            with pytest.raises(pd.errors.EmptyDataError):
                compare_orfs_to_reference(orfs, "NC_TEST")

    def test_accepts_genome_start_genome_end_columns(self, tmp_path):
        """ORF dicts using genome_start/genome_end keys should also work."""
        gff = _make_gff_file(tmp_path)
        orfs = [
            {"genome_start": 100, "genome_end": 300, "strand": "+"},
            {"genome_start": 500, "genome_end": 700, "strand": "+"},
        ]

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            result = compare_orfs_to_reference(orfs, "NC_TEST")

        assert result["true_positives"] == 2

    def test_raises_value_error_for_missing_coord_columns(self, tmp_path):
        gff = _make_gff_file(tmp_path)
        orfs = [{"combined_score": 0.9}]  # no start/end keys

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            with pytest.raises(ValueError, match="start"):
                compare_orfs_to_reference(orfs, "NC_TEST")

    def test_returned_dict_has_all_required_keys(self, tmp_path):
        gff = _make_gff_file(tmp_path)
        orfs = _orfs_from_coords(_REF_COORDS)

        with patch("src.comparative_analysis.get_gff_path", return_value=gff):
            result = compare_orfs_to_reference(orfs, "NC_TEST")

        required = {
            "predicted", "reference", "true_positives",
            "false_negatives", "false_positives",
            "sensitivity", "precision", "f1_score",
        }
        assert required <= set(result.keys())


# ---------------------------------------------------------------------------
# compare_results_file_to_reference — GFF file parsing
# ---------------------------------------------------------------------------


class TestCompareResultsFileToReference:
    def _write_predictions_gff(self, path: Path, coords) -> None:
        """Write a minimal GFF predictions file."""
        lines = ["##gff-version 3\n"]
        for start, end in coords:
            lines.append(f"NC_TEST\t.\tCDS\t{start}\t{end}\t.\t+\t0\t.\n")
        path.write_text("".join(lines))

    def test_perfect_match(self, tmp_path, monkeypatch):
        pred_path = tmp_path / "NC_TEST_predictions.gff"
        self._write_predictions_gff(pred_path, _REF_COORDS)
        ref_gff = _make_gff_file(tmp_path)

        monkeypatch.chdir(tmp_path)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "NC_TEST_predictions.gff").write_text(pred_path.read_text())

        with patch("src.comparative_analysis.get_gff_path", return_value=ref_gff):
            with patch(
                "src.comparative_analysis.load_reference_genes_from_gff",
                return_value=set(_REF_COORDS),
            ):
                result = compare_results_file_to_reference("NC_TEST")

        assert result["true_positives"] == 3
        # Note: compare_results_file_to_reference returns sensitivity/precision
        # as fractions (0.0–1.0), unlike compare_orfs_to_reference which uses
        # percentages (0–100).  This inconsistency is tracked in issue #29.
        assert result["sensitivity"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)

    def test_raises_file_not_found_when_results_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results").mkdir()

        with pytest.raises(FileNotFoundError, match="NC_MISSING"):
            compare_results_file_to_reference("NC_MISSING")

    def test_comment_and_blank_lines_skipped(self, tmp_path, monkeypatch):
        """Lines starting with '#' and blank lines must not be counted as predictions."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        pred_file = results_dir / "NC_TEST_predictions.gff"
        pred_file.write_text(
            "##gff-version 3\n"
            "# comment line\n"
            "\n"
            "NC_TEST\t.\tCDS\t100\t300\t.\t+\t0\t.\n"
        )

        monkeypatch.chdir(tmp_path)
        ref_gff = _make_gff_file(tmp_path)

        with patch("src.comparative_analysis.get_gff_path", return_value=ref_gff):
            with patch(
                "src.comparative_analysis.load_reference_genes_from_gff",
                return_value={(100, 300)},
            ):
                result = compare_results_file_to_reference("NC_TEST")

        assert result["predicted_count"] == 1
