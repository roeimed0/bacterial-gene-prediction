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

    def test_strand_fractions_use_forward_reverse_labels_issue_97(self, two_orf_group):
        """Regression #97: strand_plus_frac / strand_minus_frac must use
        'forward' / 'reverse' labels (not '+' / '-').

        Before the fix both features were always 0.0 because the pipeline
        stores strand as 'forward'/'reverse' but the code counted '+'/'-'.
        """
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        # All ORFs in two_orf_group are on the 'forward' strand (conftest.py)
        assert df.loc[0, "strand_plus_frac"] == pytest.approx(
            1.0, abs=1e-6
        ), "strand_plus_frac should be 1.0 for a forward-strand group"
        assert df.loc[0, "strand_minus_frac"] == pytest.approx(
            0.0, abs=1e-6
        ), "strand_minus_frac should be 0.0 for a forward-strand group"

    def test_strand_fractions_reverse_group_issue_97(self):
        """Regression #97: verify reverse-strand group gives correct fractions."""
        from tests.conftest import _BASE_ORF

        reverse_orf = dict(_BASE_ORF)
        reverse_orf["strand"] = "reverse"
        reverse_group = {"group_1": [reverse_orf, dict(reverse_orf)]}

        clf = OrfGroupClassifier()
        df = clf.extract_group_features(reverse_group, genome_id="test")
        assert df.loc[0, "strand_plus_frac"] == pytest.approx(0.0, abs=1e-6)
        assert df.loc[0, "strand_minus_frac"] == pytest.approx(1.0, abs=1e-6)

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

    def test_value_error_message_contains_missing_feature_name_issue_47(self, two_orf_group):
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

    def test_raises_value_error_on_missing_features_issue_47(self, synthetic_candidate):
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
        incomplete_df = pd.DataFrame([{f: 0.0 for f in hgf.feature_names[:-1]}])

        with patch.object(hgf, "extract_features", return_value=incomplete_df):
            with pytest.raises(ValueError, match=missing_feature):
                hgf.predict([synthetic_candidate], genome_id="test")

    def test_value_error_message_contains_missing_feature_name_issue_47(self, synthetic_candidate):
        """The ValueError message must name the missing feature(s) explicitly."""
        hgf = HybridGeneFilter()
        hgf.model = MagicMock()

        sentinel = "polar_fraction"  # last feature in the 25-feature list
        incomplete_df = pd.DataFrame([{f: 0.0 for f in hgf.feature_names if f != sentinel}])

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


# ===========================================================================
# OrfGroupClassifier — train / save / calibrate_threshold  (issue #111)
# ===========================================================================


def _make_feature_df(n: int, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    """Return a small synthetic feature DataFrame with named columns."""
    rng = np.random.default_rng(seed)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(rng.random((n, n_features)), columns=cols)


def _make_labels(n: int, pos_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """Return a binary label array with ~pos_frac positive examples."""
    rng = np.random.default_rng(seed)
    return (rng.random(n) < pos_frac).astype(np.int32)


class TestOrfGroupClassifierTrain:
    """Unit tests for OrfGroupClassifier.train() — issue #111."""

    def test_train_sets_model_attribute(self):
        clf = OrfGroupClassifier()
        X = _make_feature_df(200)
        y = _make_labels(200, pos_frac=0.15)
        clf.train(X, y)
        assert clf.model is not None

    def test_train_sets_feature_names(self):
        clf = OrfGroupClassifier()
        X = _make_feature_df(200)
        y = _make_labels(200, pos_frac=0.15)
        clf.train(X, y)
        assert clf.feature_names == list(X.columns)

    def test_train_model_can_predict_proba(self):
        clf = OrfGroupClassifier()
        X = _make_feature_df(200)
        y = _make_labels(200, pos_frac=0.15)
        clf.train(X, y)
        probs = np.asarray(clf.model.predict_proba(X.values))[:, 1]
        assert probs.shape == (200,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_train_raises_on_all_negative_labels(self):
        clf = OrfGroupClassifier()
        X = _make_feature_df(50)
        y = np.zeros(50, dtype=np.int32)
        with pytest.raises(ValueError, match="no positive examples"):
            clf.train(X, y)

    def test_train_with_val_set_uses_early_stopping(self):
        """Passing X_val/y_val must not raise — early stopping path exercised."""
        clf = OrfGroupClassifier()
        X_tr = _make_feature_df(200, seed=0)
        y_tr = _make_labels(200, pos_frac=0.15, seed=0)
        X_val = _make_feature_df(50, seed=1)
        y_val = _make_labels(50, pos_frac=0.15, seed=1)
        clf.train(X_tr, y_tr, X_val, y_val)
        assert clf.model is not None

    def test_scale_pos_weight_applied(self):
        """scale_pos_weight must be set to n_neg/n_pos in the trained model."""
        clf = OrfGroupClassifier()
        X = _make_feature_df(300)
        y = _make_labels(300, pos_frac=0.1)
        clf.train(X, y)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        expected_spw = n_neg / n_pos
        actual_spw = clf.model.get_params()["scale_pos_weight"]
        assert actual_spw == pytest.approx(expected_spw, rel=1e-6)


class TestOrfGroupClassifierCalibrateThreshold:
    """Unit tests for OrfGroupClassifier.calibrate_threshold() — issue #111."""

    def test_returns_float_in_unit_interval(self):
        clf = OrfGroupClassifier()
        X = _make_feature_df(300)
        y = _make_labels(300, pos_frac=0.15)
        clf.train(X, y)
        t = clf.calibrate_threshold(X, y)
        assert isinstance(t, float)
        assert 0.0 < t < 1.0

    def test_calibrated_threshold_better_than_default(self):
        """F1 at the calibrated threshold must be ≥ F1 at 0.5 on training data."""
        from sklearn.metrics import f1_score

        clf = OrfGroupClassifier()
        X = _make_feature_df(500)
        y = _make_labels(500, pos_frac=0.1)
        clf.train(X, y)
        t = clf.calibrate_threshold(X, y)

        probs = np.asarray(clf.model.predict_proba(X.values))[:, 1]
        f1_calibrated = f1_score(y, probs >= t, zero_division=0)
        f1_default = f1_score(y, probs >= 0.5, zero_division=0)
        assert f1_calibrated >= f1_default - 1e-6


class TestOrfGroupClassifierSave:
    """Unit tests for OrfGroupClassifier.save() / load() round-trip — issue #111."""

    def test_save_creates_model_file(self, tmp_path):
        clf = OrfGroupClassifier()
        X = _make_feature_df(200)
        y = _make_labels(200, pos_frac=0.15)
        clf.train(X, y)
        out = tmp_path / "lgb_test.pkl"
        clf.save(str(out))
        assert out.exists()

    def test_save_creates_feature_names_file(self, tmp_path):
        clf = OrfGroupClassifier()
        X = _make_feature_df(200)
        y = _make_labels(200, pos_frac=0.15)
        clf.train(X, y)
        clf.save(str(tmp_path / "lgb_test.pkl"))
        assert (tmp_path / "feature_names.pkl").exists()

    def test_save_load_roundtrip_preserves_predictions(self, tmp_path):
        """A model saved and re-loaded must produce identical probabilities."""
        clf = OrfGroupClassifier()
        X = _make_feature_df(200)
        y = _make_labels(200, pos_frac=0.15)
        clf.train(X, y)
        probs_before = np.asarray(clf.model.predict_proba(X.values))[:, 1]

        out = tmp_path / "lgb_test.pkl"
        clf.save(str(out))

        clf2 = OrfGroupClassifier()
        clf2.load(str(out))
        probs_after = np.asarray(clf2.model.predict_proba(X.values))[:, 1]

        np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5)

    def test_save_load_roundtrip_preserves_feature_names(self, tmp_path):
        clf = OrfGroupClassifier()
        X = _make_feature_df(200)
        y = _make_labels(200, pos_frac=0.15)
        clf.train(X, y)
        clf.save(str(tmp_path / "lgb_test.pkl"))

        clf2 = OrfGroupClassifier()
        clf2.load(str(tmp_path / "lgb_test.pkl"))
        assert clf2.feature_names == clf.feature_names


# ===========================================================================
# scripts/train_lgb.py — label_groups / build_splits  (issue #111)
# ===========================================================================


class TestLabelGroups:
    """Unit tests for scripts/train_lgb.label_groups()."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from scripts.train_lgb import label_groups

        self.label_groups = label_groups

    def _make_group(self, genome_start: int, genome_end: int) -> pd.DataFrame:
        return pd.DataFrame(
            [{"genome_start": genome_start, "genome_end": genome_end, "start": 1, "end": 100}]
        )

    def test_positive_when_orf_in_ref_set(self):
        groups = {"g1": self._make_group(100, 400)}
        ref = {(100, 400)}
        labels = self.label_groups(groups, ref)
        assert labels[0] == 1

    def test_negative_when_orf_not_in_ref_set(self):
        groups = {"g1": self._make_group(100, 400)}
        ref = {(200, 500)}
        labels = self.label_groups(groups, ref)
        assert labels[0] == 0

    def test_group_positive_if_any_orf_matches(self):
        df = pd.DataFrame(
            [
                {"genome_start": 100, "genome_end": 400},
                {"genome_start": 200, "genome_end": 500},
            ]
        )
        groups = {"g1": df}
        ref = {(200, 500)}
        labels = self.label_groups(groups, ref)
        assert labels[0] == 1

    def test_empty_ref_set_all_negative(self):
        groups = {"g1": self._make_group(100, 400), "g2": self._make_group(500, 800)}
        labels = self.label_groups(groups, set())
        assert list(labels) == [0, 0]

    def test_label_array_length_matches_group_count(self):
        groups = {f"g{i}": self._make_group(i * 100, i * 100 + 300) for i in range(5)}
        labels = self.label_groups(groups, set())
        assert len(labels) == 5

    def test_falls_back_to_start_end_when_genome_coords_absent(self):
        df = pd.DataFrame([{"start": 100, "end": 400}])
        groups = {"g1": df}
        ref = {(100, 400)}
        labels = self.label_groups(groups, ref)
        assert labels[0] == 1


class TestBuildSplits:
    """Unit tests for scripts/train_lgb.build_splits()."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from scripts.train_lgb import build_splits

        self.build_splits = build_splits

    def _groups(self, n_per_group: int = 25):
        return {
            "Proteobacteria": [f"PA_{i}" for i in range(n_per_group)],
            "Firmicutes": [f"FA_{i}" for i in range(n_per_group)],
            "Actinobacteria": [f"AA_{i}" for i in range(n_per_group)],
            "Archaea": [f"AR_{i}" for i in range(n_per_group)],
        }

    def test_train_plus_val_plus_test_covers_all_genomes(self):
        groups = self._groups(25)
        total = sum(len(v) for v in groups.values())
        train, val, test = self.build_splits(groups, val_per_group=4, test_per_group=4, seed=42)
        assert len(train) + len(val) + len(test) == total

    def test_no_overlap_between_any_splits(self):
        groups = self._groups(25)
        train, val, test = self.build_splits(groups, val_per_group=4, test_per_group=4, seed=42)
        assert set(train).isdisjoint(set(val))
        assert set(train).isdisjoint(set(test))
        assert set(val).isdisjoint(set(test))

    def test_val_and_test_sizes_respect_per_group_cap(self):
        groups = self._groups(25)
        _, val, test = self.build_splits(groups, val_per_group=4, test_per_group=4, seed=42)
        assert len(val) <= 4 * len(groups)
        assert len(test) <= 4 * len(groups)

    def test_train_is_majority_of_data(self):
        groups = self._groups(25)
        train, val, test = self.build_splits(groups, val_per_group=4, test_per_group=4, seed=42)
        total = len(train) + len(val) + len(test)
        assert len(train) > total * 0.6

    def test_deterministic_with_same_seed(self):
        groups = self._groups(25)
        r1 = self.build_splits(groups, val_per_group=4, test_per_group=4, seed=7)
        r2 = self.build_splits(groups, val_per_group=4, test_per_group=4, seed=7)
        assert r1 == r2

    def test_different_seeds_produce_different_splits(self):
        groups = self._groups(25)
        _, val1, test1 = self.build_splits(groups, val_per_group=4, test_per_group=4, seed=1)
        _, val2, test2 = self.build_splits(groups, val_per_group=4, test_per_group=4, seed=2)
        assert val1 != val2 or test1 != test2
