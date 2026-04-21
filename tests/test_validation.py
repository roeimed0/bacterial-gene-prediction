"""
Unit tests for src/validation.py.

validate_predictions() and validate_from_results_directory() both delegate
to compare_results_file_to_reference(), which is tested in
test_comparative_analysis.py.  These tests verify the wrapper logic:
genome_id extraction, path construction, and error propagation.
print_validation_report() is tested as a smoke test (no crash, correct
formatting keys).
"""

from unittest.mock import patch

import pytest

from src.validation import (
    print_validation_report,
    validate_from_results_directory,
    validate_predictions,
)

# ---------------------------------------------------------------------------
# A minimal metrics dict as returned by compare_results_file_to_reference()
# ---------------------------------------------------------------------------

_METRICS = {
    "genome_id": "NC_TEST",
    "results_file": "results/NC_TEST_predictions.gff",
    "reference_file": "data/full_dataset/NC_TEST.gff",
    "reference_count": 10,
    "predicted_count": 8,
    "true_positives": 7,
    "false_positives": 1,
    "false_negatives": 3,
    "sensitivity": 0.7,
    "recall": 0.7,
    "precision": 0.875,
    "f1_score": 0.7778,
}


# ---------------------------------------------------------------------------
# validate_predictions()
# ---------------------------------------------------------------------------


class TestValidatePredictions:
    def test_delegates_to_compare_results(self):
        # validate_predictions does a local import, so patch the source module
        with patch(
            "src.comparative_analysis.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            result = validate_predictions(
                pred_path="results/NC_TEST_predictions.gff",
                genome_id="NC_TEST",
            )

        mock_crf.assert_called_once_with("NC_TEST")
        assert result == _METRICS

    def test_extracts_genome_id_from_filename_when_not_provided(self):
        with patch(
            "src.comparative_analysis.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_predictions(pred_path="results/NC_000913.3_predictions.gff")

        mock_crf.assert_called_once_with("NC_000913.3")

    def test_explicit_genome_id_overrides_filename(self):
        with patch(
            "src.comparative_analysis.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_predictions(
                pred_path="results/anything.gff",
                genome_id="NC_EXPLICIT",
            )

        mock_crf.assert_called_once_with("NC_EXPLICIT")

    def test_propagates_file_not_found_error(self):
        with patch(
            "src.comparative_analysis.compare_results_file_to_reference",
            side_effect=FileNotFoundError("Results file not found"),
        ):
            with pytest.raises(FileNotFoundError):
                validate_predictions(pred_path="results/NC_MISSING_predictions.gff")

    def test_returns_metrics_dict(self):
        with patch(
            "src.comparative_analysis.compare_results_file_to_reference",
            return_value=_METRICS,
        ):
            result = validate_predictions("results/NC_TEST_predictions.gff", genome_id="NC_TEST")

        assert isinstance(result, dict)
        assert "sensitivity" in result
        assert "precision" in result
        assert "f1_score" in result


# ---------------------------------------------------------------------------
# print_validation_report()
# ---------------------------------------------------------------------------


class TestPrintValidationReport:
    def test_does_not_raise(self, capsys):
        print_validation_report(_METRICS)

    def test_outputs_sensitivity_value(self, capsys):
        print_validation_report(_METRICS)
        output = capsys.readouterr().out
        assert "70.00" in output or "0.70" in output or "70%" in output

    def test_outputs_f1_score(self, capsys):
        print_validation_report(_METRICS)
        output = capsys.readouterr().out
        assert "0.7778" in output or "F1" in output.upper()

    def test_outputs_tp_fp_fn_counts(self, capsys):
        print_validation_report(_METRICS)
        output = capsys.readouterr().out
        assert "7" in output   # TP
        assert "1" in output   # FP
        assert "3" in output   # FN


# ---------------------------------------------------------------------------
# validate_from_results_directory()
# ---------------------------------------------------------------------------


class TestValidateFromResultsDirectory:
    def test_constructs_correct_pred_path(self):
        with patch(
            "src.comparative_analysis.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_from_results_directory("NC_TEST")

        mock_crf.assert_called_once_with("NC_TEST")

    def test_returns_metrics_dict(self):
        with patch(
            "src.comparative_analysis.compare_results_file_to_reference",
            return_value=_METRICS,
        ):
            result = validate_from_results_directory("NC_TEST")

        assert result == _METRICS
