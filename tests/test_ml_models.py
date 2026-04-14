"""
Tests for src/ml_models.py — OrfGroupClassifier and HybridGeneFilter.

Coverage in this file:
  - OrfGroupClassifier: feature extraction, feature count, edge cases, error handling
  - HybridGeneFilter:   feature extraction, feature count, edge cases, error handling

Regression tests follow the naming convention:
    test_<behavior>_issue_<N>
and include a docstring explaining the original bug.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml_models import HybridGeneFilter, OrfGroupClassifier

# ---------------------------------------------------------------------------
# Expected constants (must stay in sync with models/feature_names.pkl and
# HybridGeneFilter.feature_names — see role_ml_engineer.md for the full list)
# ---------------------------------------------------------------------------

EXPECTED_LGB_FEATURE_COUNT = 31
EXPECTED_HYBRID_FEATURE_COUNT = 25


# ===========================================================================
# OrfGroupClassifier
# ===========================================================================


class TestOrfGroupClassifierFeatureExtraction:
    """Unit tests for OrfGroupClassifier.extract_group_features()."""

    def test_returns_correct_number_of_feature_columns(self, two_orf_group):
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        # group_id is metadata, not a model feature
        feature_cols = [c for c in df.columns if c != "group_id"]
        assert len(feature_cols) == EXPECTED_LGB_FEATURE_COUNT, (
            f"Expected {EXPECTED_LGB_FEATURE_COUNT} features, got {len(feature_cols)}: "
            f"{sorted(feature_cols)}"
        )

    def test_returns_one_row_per_group(self, two_orf_group):
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        assert len(df) == 1

    def test_group_id_column_present(self, two_orf_group):
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        assert "group_id" in df.columns

    def test_single_orf_group_does_not_crash(self, single_orf_group):
        """
        Edge case: a group with only one ORF must not crash when computing
        ``combined_margin_top2``, which requires sorting at least two scores.
        The implementation uses the single score directly when len == 1.
        """
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(single_orf_group, genome_id="test")
        assert len(df) == 1

    def test_combined_margin_top2_single_orf_equals_score(self, synthetic_orf):
        """When there is only one ORF, combined_margin_top2 equals that ORF's score."""
        clf = OrfGroupClassifier()
        group = {"g": [synthetic_orf]}
        df = clf.extract_group_features(group, genome_id="test")
        assert df.loc[0, "combined_margin_top2"] == pytest.approx(
            synthetic_orf["combined_score"], abs=1e-6
        )

    def test_num_orfs_matches_group_size(self, two_orf_group):
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        assert df.loc[0, "num_orfs"] == 2

    def test_strand_fractions_sum_to_one(self, two_orf_group):
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        total = df.loc[0, "strand_plus_frac"] + df.loc[0, "strand_minus_frac"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_combined_mean_within_score_range(self, two_orf_group):
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        row = df.iloc[0]
        assert row["combined_mean"] <= row["combined_max"]
        assert row["combined_mean"] >= 0.0

    def test_empty_group_dict_returns_empty_dataframe(self):
        clf = OrfGroupClassifier()
        df = clf.extract_group_features({}, genome_id="test")
        assert len(df) == 0


class TestOrfGroupClassifierPredictGroups:
    """Tests for OrfGroupClassifier.predict_groups() — error handling."""

    def test_raises_value_error_on_missing_features_issue_47(self, two_orf_group):
        """
        Regression for issue #47: predict_groups() silently used fewer features
        when the extracted DataFrame was missing a feature the model expected.
        It printed a warning and continued with a truncated feature matrix,
        which either produced wrong predictions or caused a cryptic LightGBM
        shape error downstream.

        After the fix, predict_groups() must raise ValueError immediately,
        listing the missing feature names so the caller can diagnose the root cause.
        """
        clf = OrfGroupClassifier()
        clf.model = MagicMock()
        # Inject a feature name that extract_group_features() never produces
        clf.model.feature_name_ = ["combined_max", "__nonexistent_feature__"]

        with pytest.raises(ValueError, match="__nonexistent_feature__"):
            clf.predict_groups(two_orf_group, genome_id="test")

    def test_raises_runtime_error_when_model_not_loaded(self, two_orf_group):
        """
        Calling predict_groups() without first calling load() should fail with
        a clear AttributeError (self.model is None → None has no feature_name_).
        This documents the expected failure mode; a future ENH may make it an
        explicit RuntimeError.
        """
        clf = OrfGroupClassifier()
        with pytest.raises((AttributeError, RuntimeError)):
            clf.predict_groups(two_orf_group, genome_id="test")

    def test_value_error_message_contains_missing_feature_name_issue_47(
        self, two_orf_group
    ):
        """The ValueError message must name the missing feature(s) explicitly."""
        clf = OrfGroupClassifier()
        clf.model = MagicMock()
        clf.model.feature_name_ = ["combined_max", "__sentinel_feature__"]

        with pytest.raises(ValueError) as exc_info:
            clf.predict_groups(two_orf_group, genome_id="test")

        assert "__sentinel_feature__" in str(exc_info.value)


# ===========================================================================
# HybridGeneFilter
# ===========================================================================


class TestHybridGeneFilterFeatureExtraction:
    """Unit tests for HybridGeneFilter.extract_features()."""

    def test_returns_correct_number_of_feature_columns(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        assert len(df.columns) == EXPECTED_HYBRID_FEATURE_COUNT, (
            f"Expected {EXPECTED_HYBRID_FEATURE_COUNT} features, "
            f"got {len(df.columns)}: {sorted(df.columns.tolist())}"
        )

    def test_feature_names_match_instance_list(self, synthetic_candidate):
        """Column names must exactly match self.feature_names (order matters for tensor construction)."""
        hgf = HybridGeneFilter()
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        assert list(df.columns) == hgf.feature_names

    def test_returns_one_row_per_candidate(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        df = hgf.extract_features([synthetic_candidate, synthetic_candidate], genome_id="test")
        assert len(df) == 2

    def test_gc_content_in_valid_range(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        gc = df.loc[0, "gc_content"]
        assert 0.0 <= gc <= 1.0

    def test_length_bp_matches_sequence_length(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        assert df.loc[0, "length_bp"] == pytest.approx(
            len(synthetic_candidate["sequence"]), abs=1e-6
        )

    def test_length_log_equals_log_of_length(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        expected = math.log(len(synthetic_candidate["sequence"]))
        assert df.loc[0, "length_log"] == pytest.approx(expected, abs=1e-6)

    def test_atg_start_codon_type_is_zero(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        synthetic_candidate["start_codon"] = "ATG"
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        assert df.loc[0, "start_codon_type"] == 0.0

    def test_gtg_start_codon_type_is_one(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        synthetic_candidate["start_codon"] = "GTG"
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        assert df.loc[0, "start_codon_type"] == 1.0

    def test_ttg_start_codon_type_is_two(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        synthetic_candidate["start_codon"] = "TTG"
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        assert df.loc[0, "start_codon_type"] == 2.0

    def test_no_crash_on_empty_sequence(self):
        """A candidate with no sequence should not raise — all computed features default to 0."""
        hgf = HybridGeneFilter()
        candidate = {"sequence": "", "length": 0, "gene_id": "empty"}
        df = hgf.extract_features([candidate], genome_id="test")
        assert len(df) == 1
        assert df.loc[0, "gc_content"] == 0.0


class TestHybridGeneFilterPredict:
    """Tests for HybridGeneFilter.predict() — error handling and edge cases."""

    def test_raises_runtime_error_when_model_not_loaded(self, synthetic_candidate):
        hgf = HybridGeneFilter()
        with pytest.raises(RuntimeError, match="load()"):
            hgf.predict([synthetic_candidate], genome_id="test")

    def test_returns_empty_arrays_on_empty_candidate_list(self):
        hgf = HybridGeneFilter()
        hgf.model = MagicMock()
        preds, probs, gene_ids = hgf.predict([], genome_id="test")
        assert len(preds) == 0
        assert len(probs) == 0
        assert len(gene_ids) == 0

    def test_raises_value_error_on_missing_features_issue_47(
        self, synthetic_candidate
    ):
        """
        Regression for issue #47: predict() silently constructed a feature
        tensor with fewer columns than the DenseBranch(input_dim=25) expects
        when extract_features() returned a DataFrame missing one or more
        feature columns.  The Dense branch would then fail with a cryptic
        PyTorch shape error rather than a clear message about the missing feature.

        After the fix, predict() must raise ValueError immediately upon
        detecting any missing feature, naming the absent column(s).
        """
        hgf = HybridGeneFilter()
        hgf.model = MagicMock()  # non-None: passes the "model not loaded" guard

        # Build a DataFrame that is missing the last expected feature
        missing_feature = hgf.feature_names[-1]
        incomplete_df = pd.DataFrame(
            [{f: 0.0 for f in hgf.feature_names[:-1]}]
        )

        with patch.object(hgf, "extract_features", return_value=incomplete_df):
            with pytest.raises(ValueError, match=missing_feature):
                hgf.predict([synthetic_candidate], genome_id="test")

    def test_value_error_message_contains_missing_feature_name_issue_47(
        self, synthetic_candidate
    ):
        """The ValueError message must name the missing feature(s) explicitly."""
        hgf = HybridGeneFilter()
        hgf.model = MagicMock()

        sentinel = "polar_fraction"  # last feature in the 25-feature list
        incomplete_df = pd.DataFrame(
            [{f: 0.0 for f in hgf.feature_names if f != sentinel}]
        )

        with patch.object(hgf, "extract_features", return_value=incomplete_df):
            with pytest.raises(ValueError) as exc_info:
                hgf.predict([synthetic_candidate], genome_id="test")

        assert sentinel in str(exc_info.value)


# ===========================================================================
# Static helper methods
# ===========================================================================


class TestHybridGeneFilterStaticHelpers:
    """Unit tests for the static utility methods on HybridGeneFilter."""

    def test_calculate_enc_returns_float_in_unit_interval(self):
        enc = HybridGeneFilter._calculate_enc("ATGAAACCCGGGTAA")
        assert isinstance(enc, float)
        assert 0.0 <= enc <= 1.0

    def test_calculate_enc_empty_sequence_returns_zero(self):
        assert HybridGeneFilter._calculate_enc("") == 0.0

    def test_calculate_cbi_returns_float_in_unit_interval(self):
        cbi = HybridGeneFilter._calculate_cbi("ATGAAACCCGGGTAA")
        assert isinstance(cbi, float)
        assert 0.0 <= cbi <= 1.0

    def test_calculate_cbi_empty_sequence_returns_zero(self):
        assert HybridGeneFilter._calculate_cbi("") == 0.0

    def test_detect_hairpin_returns_zero_or_one(self):
        result = HybridGeneFilter._detect_hairpin_near_stop("ATGAAACCCGGGTAA")
        assert result in (0.0, 1.0)

    def test_detect_hairpin_short_sequence_returns_zero(self):
        """Sequence shorter than window (default 30 bp) must return 0.0."""
        result = HybridGeneFilter._detect_hairpin_near_stop("ATGTAA", window=30)
        assert result == 0.0
