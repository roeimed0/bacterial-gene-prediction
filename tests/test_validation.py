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
            "src.validation.compare_results_file_to_reference",
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
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_predictions(pred_path="results/NC_000913.3_predictions.gff")

        mock_crf.assert_called_once_with("NC_000913.3")

    def test_explicit_genome_id_overrides_filename(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_predictions(
                pred_path="results/anything.gff",
                genome_id="NC_EXPLICIT",
            )

        mock_crf.assert_called_once_with("NC_EXPLICIT")

    def test_propagates_file_not_found_error(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            side_effect=FileNotFoundError("Results file not found"),
        ):
            with pytest.raises(FileNotFoundError):
                validate_predictions(pred_path="results/NC_MISSING_predictions.gff")

    def test_returns_metrics_dict(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ):
            result = validate_predictions("results/NC_TEST_predictions.gff", genome_id="NC_TEST")

        assert isinstance(result, dict)
        assert "sensitivity" in result
        assert "precision" in result
        assert "f1_score" in result

    def test_genome_id_stripped_of_predictions_suffix(self):
        """Stem 'NC_000913.3_predictions' → genome_id 'NC_000913.3'."""
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_predictions("results/NC_000913.3_predictions.gff")

        mock_crf.assert_called_once_with("NC_000913.3")

    def test_propagates_value_error(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            side_effect=ValueError("bad coords"),
        ):
            with pytest.raises(ValueError):
                validate_predictions("results/NC_TEST_predictions.gff", genome_id="NC_TEST")

    def test_called_exactly_once(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_predictions("results/NC_TEST_predictions.gff", genome_id="NC_TEST")

        assert mock_crf.call_count == 1


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
        assert "7" in output  # TP
        assert "1" in output  # FP
        assert "3" in output  # FN

    def test_outputs_genome_id(self, capsys):
        print_validation_report(_METRICS)
        output = capsys.readouterr().out
        assert "NC_TEST" in output

    def test_outputs_precision_value(self, capsys):
        print_validation_report(_METRICS)
        output = capsys.readouterr().out
        assert "87.50" in output or "0.875" in output or "87%" in output or "Precision" in output

    def test_outputs_reference_count(self, capsys):
        print_validation_report(_METRICS)
        output = capsys.readouterr().out
        assert "10" in output  # reference_count

    def test_outputs_predicted_count(self, capsys):
        print_validation_report(_METRICS)
        output = capsys.readouterr().out
        assert "8" in output  # predicted_count


# ---------------------------------------------------------------------------
# validate_from_results_directory()
# ---------------------------------------------------------------------------


class TestValidateFromResultsDirectory:
    def test_calls_compare_with_correct_genome_id(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_from_results_directory("NC_TEST")

        mock_crf.assert_called_once_with("NC_TEST")

    def test_returns_metrics_dict(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ):
            result = validate_from_results_directory("NC_TEST")

        assert result == _METRICS

    def test_returns_dict_with_sensitivity_key(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ):
            result = validate_from_results_directory("NC_TEST")

        assert "sensitivity" in result

    def test_returns_dict_with_f1_key(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ):
            result = validate_from_results_directory("NC_TEST")

        assert "f1_score" in result

    def test_propagates_file_not_found(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            side_effect=FileNotFoundError("missing"),
        ):
            with pytest.raises(FileNotFoundError):
                validate_from_results_directory("NC_MISSING")

    def test_called_exactly_once(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_from_results_directory("NC_TEST")

        assert mock_crf.call_count == 1

    def test_different_genome_id_passes_correctly(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ) as mock_crf:
            validate_from_results_directory("NC_000913.3")

        mock_crf.assert_called_once_with("NC_000913.3")

    def test_result_has_no_missing_core_keys(self):
        with patch(
            "src.validation.compare_results_file_to_reference",
            return_value=_METRICS,
        ):
            result = validate_from_results_directory("NC_TEST")

        for key in ("sensitivity", "precision", "f1_score", "true_positives"):
            assert key in result, f"Missing key: {key}"


# ===========================================================================
# Tests: validate_batch()
# ===========================================================================


class TestValidateBatch:
    """Tests for the per-taxonomy batch validation function.

    Uses a fixture of 6 synthetic genomes across 3 groups so every
    aggregation path (single-genome group, multi-genome group, overall,
    flag threshold) is exercised.
    """

    # ── Six synthetic metric dicts ────────────────────────────────────────
    # Proteobacteria: NC_A (good), NC_B (good)
    # Archaea:        NC_C (poor — should trigger the flag)
    # Firmicutes:     NC_D, NC_E, NC_F (mixed)

    _MAKE = staticmethod(
        lambda acc, sens, prec, f1: {
            "genome_id": acc,
            "results_file": f"results/{acc}_predictions.gff",
            "reference_file": f"ref/{acc}.gff",
            "reference_count": 100,
            "predicted_count": 90,
            "true_positives": int(sens * 100),
            "false_positives": 90 - int(sens * 100),
            "false_negatives": 100 - int(sens * 100),
            "sensitivity": sens,
            "precision": prec,
            "f1_score": f1,
        }
    )

    @classmethod
    def _all_metrics(cls):
        return {
            "NC_A": cls._MAKE("NC_A", 0.85, 0.90, 0.874),
            "NC_B": cls._MAKE("NC_B", 0.80, 0.88, 0.838),
            "NC_C": cls._MAKE("NC_C", 0.35, 0.60, 0.441),  # poor — Archaea
            "NC_D": cls._MAKE("NC_D", 0.75, 0.82, 0.784),
            "NC_E": cls._MAKE("NC_E", 0.78, 0.85, 0.814),
            "NC_F": cls._MAKE("NC_F", 0.72, 0.80, 0.758),
        }

    _GROUP_MAP = {
        "NC_A": "Proteobacteria",
        "NC_B": "Proteobacteria",
        "NC_C": "Archaea",
        "NC_D": "Firmicutes",
        "NC_E": "Firmicutes",
        "NC_F": "Firmicutes",
    }

    _ACCESSIONS = list(_GROUP_MAP.keys())

    # ── Structure ─────────────────────────────────────────────────────────

    def test_returns_required_top_level_keys(self):
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        assert set(r.keys()) == {"per_genome", "per_group", "overall"}

    def test_per_genome_has_all_accessions(self):
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        assert set(r["per_genome"].keys()) == set(self._ACCESSIONS)

    def test_per_genome_contains_group_field(self):
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        for acc, grp in self._GROUP_MAP.items():
            assert r["per_genome"][acc]["group"] == grp

    # ── Per-group counts and averages ─────────────────────────────────────

    def test_per_group_n_counts_correct(self):
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        assert r["per_group"]["Proteobacteria"]["n"] == 2
        assert r["per_group"]["Archaea"]["n"] == 1
        assert r["per_group"]["Firmicutes"]["n"] == 3

    def test_per_group_sensitivity_is_mean_of_members(self):
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        # Proteobacteria: mean(85, 80) = 82.5
        assert r["per_group"]["Proteobacteria"]["sensitivity"] == pytest.approx(82.5, abs=0.1)
        # Firmicutes: mean(75, 78, 72) = 75
        assert r["per_group"]["Firmicutes"]["sensitivity"] == pytest.approx(75.0, abs=0.1)

    def test_per_group_f1_correct(self):
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        prot_f1 = r["per_group"]["Proteobacteria"]["f1_score"]
        assert 80.0 < prot_f1 < 90.0  # sanity range for good genomes

    # ── Overall aggregation ───────────────────────────────────────────────

    def test_overall_n_equals_successful_genome_count(self):
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        assert r["overall"]["n"] == len(self._ACCESSIONS)

    def test_overall_sensitivity_is_macro_average(self):
        """Overall sensitivity = mean of all per-genome sensitivities."""
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        expected = (85 + 80 + 35 + 75 + 78 + 72) / 6
        assert r["overall"]["sensitivity"] == pytest.approx(expected, abs=0.1)

    # ── Robustness ───────────────────────────────────────────────────────

    def test_failed_genome_skipped_others_still_reported(self):
        from src.validation import validate_batch

        m = self._all_metrics()

        def _side(acc):
            if acc == "NC_C":
                raise FileNotFoundError("results not found")
            return m[acc]

        with patch("src.validation.compare_results_file_to_reference", side_effect=_side):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        assert "NC_C" not in r["per_genome"]
        assert "Archaea" not in r["per_group"]  # only genome in group skipped
        assert r["overall"]["n"] == len(self._ACCESSIONS) - 1

    def test_all_failed_returns_zero_overall(self):
        from src.validation import validate_batch

        with patch(
            "src.validation.compare_results_file_to_reference",
            side_effect=FileNotFoundError("not found"),
        ):
            r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        assert r["overall"]["n"] == 0
        assert r["per_genome"] == {}

    def test_empty_accession_list_returns_empty(self):
        from src.validation import validate_batch

        with patch("src.validation.compare_results_file_to_reference"):
            r = validate_batch([], group_map={})

        assert r["overall"]["n"] == 0
        assert r["per_group"] == {}

    def test_unknown_accession_falls_back_to_unknown_group(self):
        """Accession not in group_map → assigned group 'Unknown'."""
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(["NC_A"], group_map={})  # empty map → Unknown

        assert r["per_genome"]["NC_A"]["group"] == "Unknown"
        assert "Unknown" in r["per_group"]

    def test_single_genome_overall_equals_that_genome(self):
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            r = validate_batch(["NC_A"], group_map={"NC_A": "Proteobacteria"})

        assert r["overall"]["sensitivity"] == pytest.approx(85.0, abs=0.1)
        assert r["overall"]["n"] == 1

    def test_flagging_threshold_marks_poor_group(self):
        """Archaea (35% sens) should be flagged as >5 pp below overall."""
        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        overall_f1 = r["overall"]["f1_score"]
        archaea_f1 = r["per_group"]["Archaea"]["f1_score"]
        assert archaea_f1 < overall_f1 - 5.0  # would trigger the flag

    def test_low_n_warning_printed_when_group_below_10(self):
        """Groups with n < 10 must emit a statistically-motivated warning."""
        import contextlib
        import io

        from src.validation import validate_batch

        m = self._all_metrics()
        with patch("src.validation.compare_results_file_to_reference", side_effect=lambda a: m[a]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                validate_batch(self._ACCESSIONS, group_map=self._GROUP_MAP)

        output = buf.getvalue()
        # All groups in the fixture have n < 10 (max is Firmicutes n=3)
        assert "Low-n" in output or "n < 10" in output

    def test_no_low_n_warning_when_all_groups_above_threshold(self):
        """No warning when every group has n >= 10."""
        import contextlib
        import io

        from src.validation import validate_batch

        # Build 20 genomes, all Proteobacteria
        big_map = {f"NC_{i}": "Proteobacteria" for i in range(20)}
        base = self._all_metrics()["NC_A"]

        with patch("src.validation.compare_results_file_to_reference", return_value=base):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                validate_batch(list(big_map.keys()), group_map=big_map)

        output = buf.getvalue()
        assert "Low-n" not in output
