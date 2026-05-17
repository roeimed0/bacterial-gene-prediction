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

from src.ml_models import HybridGeneFilter, OrfGroupClassifier, StartSelectionClassifier

# ---------------------------------------------------------------------------
# Expected constants (must stay in sync with models/feature_names.pkl and
# HybridGeneFilter.feature_names — see role_ml_engineer.md for the full list)
# ---------------------------------------------------------------------------

EXPECTED_LGB_FEATURE_COUNT = 26  # 31 - 7 removed + 2 added (#123, #125)
EXPECTED_HYBRID_FEATURE_COUNT = 26  # 25 - 2 removed + 3 added (#124, #126, #127)


# ===========================================================================
# OrfGroupClassifier
# ===========================================================================


class TestOrfGroupClassifierFeatureExtraction:
    """Unit tests for OrfGroupClassifier.extract_group_features()."""

    def test_returns_correct_number_of_feature_columns(self, two_orf_group):
        # extract_group_features returns a superset; must have AT LEAST the
        # expected count (allows backward/forward compat across model versions).
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        feature_cols = [c for c in df.columns if c != "group_id"]
        assert len(feature_cols) >= EXPECTED_LGB_FEATURE_COUNT, (
            f"Expected >= {EXPECTED_LGB_FEATURE_COUNT} features, got {len(feature_cols)}: "
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

    def test_top_orf_is_longest_is_binary(self, two_orf_group):
        # top_orf_is_longest must be 0 or 1 (int boolean)
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        assert df.loc[0, "top_orf_is_longest"] in (0, 1)

    def test_length_ratio_max_min_ge_one(self, two_orf_group):
        # length_ratio_max_min >= 1.0 by definition
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(two_orf_group, genome_id="test")
        assert df.loc[0, "length_ratio_max_min"] >= 1.0

    def test_top_orf_is_longest_single_orf_group(self, single_orf_group):
        # Single-ORF group: top scorer == longest by definition
        clf = OrfGroupClassifier()
        df = clf.extract_group_features(single_orf_group, genome_id="test")
        assert df.loc[0, "top_orf_is_longest"] == 1
        assert df.loc[0, "length_ratio_max_min"] == pytest.approx(1.0, abs=1e-6)

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
        # extract_features returns a superset of all known features so that any
        # model version can select what it needs.  The DataFrame must contain AT
        # LEAST the features the current instance expects.
        hgf = HybridGeneFilter()
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        assert len(df.columns) >= EXPECTED_HYBRID_FEATURE_COUNT, (
            f"Expected >= {EXPECTED_HYBRID_FEATURE_COUNT} feature columns, "
            f"got {len(df.columns)}: {sorted(df.columns.tolist())}"
        )

    def test_feature_names_match_instance_list(self, synthetic_candidate):
        """All feature_names must be present in the extracted DataFrame (subset check)."""
        hgf = HybridGeneFilter()
        df = hgf.extract_features([synthetic_candidate], genome_id="test")
        missing = [f for f in hgf.feature_names if f not in df.columns]
        assert not missing, f"feature_names items missing from DataFrame: {missing}"

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

    def test_empty_list_return_types_are_1d_issue_150(self):
        # Regression for #150: empty input must return 1-D arrays (shape (0,))
        # not 2-D arrays (shape (0, 1)) which would silently break unpacking.
        import numpy as np

        hgf = HybridGeneFilter()
        hgf.model = MagicMock()
        preds, probs, gene_ids = hgf.predict([], genome_id="test")
        for arr in (preds, probs):
            if isinstance(arr, np.ndarray):
                assert arr.ndim == 1, f"Expected 1-D, got shape {arr.shape}"

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


# ===========================================================================
# StartSelectionClassifier
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

_SS_FEATURES = [
    "d_baseline",
    "d_rbs",
    "d_start",
    "d_codon",
    "d_imm",
    "d_length",
    "gap",
    "score_range",
    "rel_gap",
    "n_orfs",
    "gc_pct",
]
_WEIGHTS = {"codon": 4.8562, "imm": 1.0107, "rbs": 0.6383, "length": 7.4367, "start": 0.2755}
_GENOME = "ATGCATGCAT" * 500  # 5000 bp synthetic genome, no real biology needed


def _make_ss(prob_keep=0.8, contest_t=1.0, flip_t=0.80, temp_T=None):
    """Build a StartSelectionClassifier with attributes set directly (no pickle needed)."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.zeros((2, len(_SS_FEATURES))))

    clf = MagicMock()
    clf.predict_proba = MagicMock(return_value=np.array([[1 - prob_keep, prob_keep]]))

    ss = StartSelectionClassifier()
    ss.clf = clf
    ss.scaler = scaler
    ss.features = list(_SS_FEATURES)
    ss.contest_t = contest_t
    ss.flip_t = flip_t
    ss.calibrated = False
    ss.temperature_T = temp_T
    return ss


def _picklable_bundle(contest_t=1.0, flip_t=0.80, temp_T=None):
    """Return a pickle-safe bundle using a real tiny LightGBM model."""
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler

    X = np.random.default_rng(0).standard_normal((40, len(_SS_FEATURES)))
    y = np.array([0, 1] * 20)
    clf = lgb.LGBMClassifier(n_estimators=5, verbose=-1, n_jobs=1)
    clf.fit(X, y)

    scaler = StandardScaler()
    scaler.fit(X)

    return {
        "clf": clf,
        "scaler": scaler,
        "features": list(_SS_FEATURES),
        "contest_t": contest_t,
        "flip_t": flip_t,
        "calibrated": False,
        "temperature_T": temp_T,
    }


def _load_ss(tmp_path, **kwargs):
    """Persist a picklable bundle and load it into a StartSelectionClassifier."""
    import pickle

    bundle = _picklable_bundle(**kwargs)
    path = str(tmp_path / "ss_test.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    ss = StartSelectionClassifier()
    ss.load(path)
    return ss


def _orf(genome_start, genome_end, strand="forward", **overrides):
    """Build a minimal ORF dict with all fields needed by select_best_starts()."""
    base = {
        "genome_start": genome_start,
        "genome_end": genome_end,
        "start": genome_start,
        "end": genome_end,
        "strand": strand,
        "length": genome_end - genome_start,
        "sequence": _GENOME[genome_start - 1 : genome_end],
        "start_codon": "ATG",
        "codon_score_norm": 0.5,
        "imm_score_norm": 0.4,
        "rbs_score_norm": 0.6,
        "length_score_norm": 0.5,
        "start_score_norm": 0.6,
    }
    base.update(overrides)
    return base


def _mock_models():
    """Minimal scoring_models dict; real scoring functions patched out in callers."""
    return {
        "codon_model": {},
        "background_codon_model": {},
        "coding_imm": {},
        "noncoding_imm": {},
        "max_order": 2,
    }


def _run_ss(ss, groups):
    """Call select_best_starts with scoring helpers patched to return 0.0."""
    with (
        patch.object(ss, "_ext_codon_score", return_value=0.0),
        patch.object(ss, "_post_start_score", return_value=0.0),
        patch.object(ss, "_upstream_imm", return_value=0.0),
    ):
        return ss.select_best_starts(groups, _GENOME, _mock_models(), _WEIGHTS)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestStartSelectionClassifierPersistence:

    def test_load_save_roundtrip_preserves_thresholds(self, tmp_path):
        """load() → save() → load() must preserve contest_t, flip_t, features, calibrated."""
        ss1 = _load_ss(tmp_path, contest_t=0.5, flip_t=0.75)
        path2 = str(tmp_path / "ss_rt.pkl")
        ss1.save(path2)

        ss2 = StartSelectionClassifier()
        ss2.load(path2)

        assert ss2.contest_t == pytest.approx(0.5)
        assert ss2.flip_t == pytest.approx(0.75)
        assert ss2.features == ss1.features
        assert ss2.calibrated is False

    def test_load_sets_temperature_T_when_present(self, tmp_path):
        """load() populates temperature_T when the bundle includes it."""
        ss = _load_ss(tmp_path, temp_T=1.5)
        assert ss.temperature_T == pytest.approx(1.5)

    def test_load_temperature_T_defaults_to_none(self, tmp_path):
        """load() sets temperature_T=None when the bundle omits the key."""
        ss = _load_ss(tmp_path, temp_T=None)
        assert ss.temperature_T is None


# ---------------------------------------------------------------------------
# select_best_starts — guard rails
# ---------------------------------------------------------------------------


class TestStartSelectionClassifierGuards:

    def test_raises_runtime_error_if_not_loaded(self):
        """select_best_starts() raises RuntimeError before load() is called."""
        ss = StartSelectionClassifier()
        with pytest.raises(RuntimeError, match="not loaded"):
            ss.select_best_starts({}, _GENOME, _mock_models(), _WEIGHTS)

    def test_empty_groups_returns_empty_dataframe(self):
        """select_best_starts() with an empty groups dict returns an empty DataFrame."""
        ss = _make_ss()
        result = ss.select_best_starts({}, _GENOME, _mock_models(), _WEIGHTS)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# select_best_starts — contest logic
# ---------------------------------------------------------------------------


class TestStartSelectionClassifierContest:

    def test_returns_one_row_per_non_empty_group(self):
        """select_best_starts() yields exactly one winner per group."""
        ss = _make_ss(prob_keep=0.9, contest_t=10.0)
        groups = {
            "g1": pd.DataFrame([_orf(100, 400)]),
            "g2": pd.DataFrame([_orf(500, 800), _orf(550, 800)]),
        }
        result = _run_ss(ss, groups)
        assert len(result) == 2

    def test_no_flip_when_gap_exceeds_contest_threshold(self):
        """Classifier is never called when baseline gap >= contest_t."""
        ss = _make_ss(prob_keep=0.1, contest_t=1.0)
        # manufacture a large gap: high codon+length for orf1, zero for orf2
        orf1 = _orf(100, 700, codon_score_norm=1.0, length_score_norm=1.0)
        orf2 = _orf(200, 700, codon_score_norm=0.0, length_score_norm=0.0)
        groups = {"g1": pd.DataFrame([orf1, orf2])}

        _run_ss(ss, groups)

        ss.clf.predict_proba.assert_not_called()

    def test_keeps_top1_when_classifier_confident_to_keep(self):
        """prob_keep=0.9 > (1 - flip_t=0.80) → top-1 is retained."""
        ss = _make_ss(prob_keep=0.9, contest_t=100.0, flip_t=0.80)
        orf1 = _orf(100, 400, codon_score_norm=0.51)
        orf2 = _orf(150, 400, codon_score_norm=0.50)
        groups = {"g1": pd.DataFrame([orf1, orf2])}

        result = _run_ss(ss, groups)

        assert result.iloc[0]["genome_start"] == 100

    def test_flips_to_top2_when_classifier_confident_to_flip(self):
        """prob_keep=0.1 < (1 - flip_t=0.80) = 0.20 → top-2 wins."""
        ss = _make_ss(prob_keep=0.1, contest_t=100.0, flip_t=0.80)
        orf1 = _orf(100, 400, codon_score_norm=0.51)
        orf2 = _orf(150, 400, codon_score_norm=0.50)
        groups = {"g1": pd.DataFrame([orf1, orf2])}

        result = _run_ss(ss, groups)

        assert result.iloc[0]["genome_start"] == 150

    def test_no_flip_when_uncertain(self):
        """prob_keep=0.5 is between flip boundaries → keep top-1."""
        ss = _make_ss(prob_keep=0.5, contest_t=100.0, flip_t=0.80)
        orf1 = _orf(100, 400, codon_score_norm=0.51)
        orf2 = _orf(150, 400, codon_score_norm=0.50)
        groups = {"g1": pd.DataFrame([orf1, orf2])}

        result = _run_ss(ss, groups)

        assert result.iloc[0]["genome_start"] == 100

    def test_singleton_group_skips_classifier(self):
        """A single-ORF group is uncontested — classifier must not be invoked."""
        ss = _make_ss(prob_keep=0.05, contest_t=100.0)
        groups = {"g1": pd.DataFrame([_orf(100, 400)])}

        _run_ss(ss, groups)

        ss.clf.predict_proba.assert_not_called()

    def test_temperature_scaling_path_does_not_crash(self):
        """When temperature_T is set, the scaled probability path runs without error."""
        ss = _make_ss(prob_keep=0.9, contest_t=100.0, temp_T=2.0)
        orf1 = _orf(100, 400, codon_score_norm=0.51)
        orf2 = _orf(150, 400, codon_score_norm=0.50)
        groups = {"g1": pd.DataFrame([orf1, orf2])}

        result = _run_ss(ss, groups)

        assert len(result) == 1  # ran to completion; temperature branch exercised


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestStartSelectionClassifierInternals:

    def test_baseline_score_single_component(self):
        """_baseline_score with only codon_score_norm=1 equals the codon weight."""
        ss = StartSelectionClassifier()
        row = pd.Series(
            {
                "codon_score_norm": 1.0,
                "imm_score_norm": 0.0,
                "rbs_score_norm": 0.0,
                "length_score_norm": 0.0,
                "start_score_norm": 0.0,
            }
        )
        assert ss._baseline_score(row, _WEIGHTS) == pytest.approx(_WEIGHTS["codon"])

    def test_baseline_score_all_ones_equals_sum_of_weights(self):
        """_baseline_score with all components=1 equals the sum of all weights."""
        ss = StartSelectionClassifier()
        row = pd.Series(
            {k + "_score_norm": 1.0 for k in ("codon", "imm", "rbs", "length", "start")}
        )
        assert ss._baseline_score(row, _WEIGHTS) == pytest.approx(sum(_WEIGHTS.values()))

    def test_get_upstream_forward_strand_length(self):
        """_get_upstream returns at most `window` characters on the forward strand."""
        ss = StartSelectionClassifier()
        orf = {"genome_start": 100, "genome_end": 400, "strand": "forward"}
        up = ss._get_upstream(_GENOME, orf, window=25)
        assert len(up) <= 25

    def test_get_upstream_reverse_strand_does_not_crash(self):
        """_get_upstream handles reverse-strand ORFs without IndexError."""
        ss = StartSelectionClassifier()
        orf = {"genome_start": 100, "genome_end": 400, "strand": "reverse"}
        up = ss._get_upstream(_GENOME, orf, window=25)
        assert isinstance(up, str)
        assert len(up) <= 25

    def test_f4_spacer_positive_for_sd_in_range(self):
        """_f4_spacer > 0 when an SD motif sits 4–12 bp upstream of the start."""
        ss = StartSelectionClassifier()
        # "AGGAG" then 7-bp spacer → spacer distance = 7 (in range 4–12)
        upstream = "AGGAG" + "N" * 7
        assert ss._f4_spacer(upstream) > 0.0

    def test_f4_spacer_zero_for_no_sd_motif(self):
        """_f4_spacer returns 0.0 when no SD motif is present."""
        ss = StartSelectionClassifier()
        assert ss._f4_spacer("T" * 20) == 0.0

    def test_f5_gc_bias_in_range(self):
        """_f5_gc_bias returns a value in [0, 1] for a valid upstream sequence."""
        ss = StartSelectionClassifier()
        val = ss._f5_gc_bias("ATGATGATGATGATGATG")
        assert 0.0 <= val <= 1.0

    def test_f5_gc_bias_short_sequence_returns_zero(self):
        """_f5_gc_bias returns 0.0 for sequences shorter than 9 bp."""
        ss = StartSelectionClassifier()
        assert ss._f5_gc_bias("ATG") == 0.0

    def test_build_pwm_correct_shape(self):
        """_build_pwm returns ndarray of shape (window, 4)."""
        ss = StartSelectionClassifier()
        seqs = ["ATGCATGCATGCATGCATGC"] * 10
        pwm = ss._build_pwm(seqs, 20)
        assert pwm.shape == (20, 4)

    def test_score_pwm_returns_zero_for_none(self):
        """_score_pwm returns 0.0 when pwm is None (too few training sequences)."""
        ss = StartSelectionClassifier()
        assert ss._score_pwm("ATGATGATGATGATGATGATG", None) == 0.0

    def test_score_pwm_returns_zero_when_upstream_too_short(self):
        """_score_pwm returns 0.0 when upstream is shorter than PWM window."""
        ss = StartSelectionClassifier()
        pwm = np.zeros((20, 4))
        assert ss._score_pwm("ATG", pwm) == 0.0


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
        from scripts.training.train_lgb import label_groups

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
        from scripts.training.train_lgb import build_splits

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


# ===========================================================================
# HybridGeneFilter — train / calibrate_threshold / save  (issue #131)
# ===========================================================================

_SEQ = "ATG" + "CAG" * 30 + "TAA"  # 96 bp synthetic ORF
_CANDIDATE = {
    "sequence": _SEQ,
    "length": len(_SEQ),
    "start_codon": "ATG",
    "codon_score_norm": 0.5,
    "imm_score_norm": 0.5,
    "rbs_score_norm": 0.5,
    "length_score_norm": 0.5,
    "start_score_norm": 0.5,
    "combined_score": 0.7,
    "rbs_score": 2.0,
}


def _make_candidates(n: int, rng=None) -> list:
    """Return n candidate dicts with slightly varied scores."""
    rng = rng or np.random.default_rng(0)
    cands = []
    for _ in range(n):
        c = dict(_CANDIDATE)
        c["codon_score_norm"] = float(rng.uniform(0, 1))
        c["combined_score"] = float(rng.uniform(0, 1))
        cands.append(c)
    return cands


def _make_binary_labels(n: int, pos_frac: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.float32)
    pos_idx = rng.choice(n, size=int(n * pos_frac), replace=False)
    labels[pos_idx] = 1.0
    return labels


class TestHybridGeneFilterTrain:
    """Unit tests for HybridGeneFilter.train() — issue #131."""

    def test_train_sets_model_attribute(self):
        hgf = HybridGeneFilter()
        cands = _make_candidates(40)
        labels = _make_binary_labels(40, pos_frac=0.25)
        hgf.train(cands, labels, epochs=2, batch_size=16)
        assert hgf.model is not None

    def test_train_raises_on_all_negative_labels(self):
        hgf = HybridGeneFilter()
        cands = _make_candidates(20)
        labels = np.zeros(20, dtype=np.float32)
        with pytest.raises(ValueError, match="no positive examples"):
            hgf.train(cands, labels, epochs=1)

    def test_trained_model_produces_probabilities_in_unit_interval(self):
        hgf = HybridGeneFilter()
        cands = _make_candidates(40)
        labels = _make_binary_labels(40, pos_frac=0.25)
        hgf.train(cands, labels, epochs=2, batch_size=16)
        _, probs, _ = hgf.predict(cands, batch_size=16)
        assert probs.shape == (40,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_train_with_val_set_runs_without_error(self):
        hgf = HybridGeneFilter()
        tr = _make_candidates(40, rng=np.random.default_rng(0))
        vl = _make_candidates(10, rng=np.random.default_rng(1))
        hgf.train(
            tr,
            _make_binary_labels(40, pos_frac=0.25, seed=0),
            val_candidates=vl,
            val_labels=_make_binary_labels(10, pos_frac=0.3, seed=1),
            epochs=3,
            batch_size=16,
        )
        assert hgf.model is not None

    def test_focal_loss_flag_runs_without_error(self):
        hgf = HybridGeneFilter()
        cands = _make_candidates(30)
        labels = _make_binary_labels(30, pos_frac=0.3)
        hgf.train(cands, labels, epochs=2, batch_size=16, focal_loss=True)
        assert hgf.model is not None


class TestHybridGeneFilterCalibrateThreshold:
    """Unit tests for HybridGeneFilter.calibrate_threshold() — issue #131."""

    @pytest.fixture(autouse=True)
    def trained_hgf(self):
        hgf = HybridGeneFilter()
        cands = _make_candidates(60)
        labels = _make_binary_labels(60, pos_frac=0.25)
        hgf.train(cands, labels, epochs=3, batch_size=16)
        self.hgf = hgf
        self.cands = cands
        self.labels = labels

    def test_returns_float_in_unit_interval(self):
        t = self.hgf.calibrate_threshold(self.cands, self.labels)
        assert isinstance(t, float)
        assert 0.0 < t < 1.0

    def test_does_not_mutate_threshold_attribute(self):
        original = self.hgf.threshold
        self.hgf.calibrate_threshold(self.cands, self.labels)
        assert self.hgf.threshold == original


class TestHybridGeneFilterSave:
    """Unit tests for HybridGeneFilter.save() / load() round-trip — issue #131."""

    def _trained(self) -> HybridGeneFilter:
        hgf = HybridGeneFilter()
        cands = _make_candidates(40)
        labels = _make_binary_labels(40, pos_frac=0.25)
        hgf.train(cands, labels, epochs=2, batch_size=16)
        return hgf

    def test_save_creates_file(self, tmp_path):
        hgf = self._trained()
        path = tmp_path / "model.pkl"
        hgf.save(str(path))
        assert path.exists()

    def test_save_raises_when_not_trained(self, tmp_path):
        hgf = HybridGeneFilter()
        with pytest.raises(RuntimeError, match="train()"):
            hgf.save(str(tmp_path / "model.pkl"))

    def test_load_restores_threshold(self, tmp_path):
        hgf = self._trained()
        hgf.threshold = 0.42
        path = tmp_path / "model.pkl"
        hgf.save(str(path))

        hgf2 = HybridGeneFilter()
        hgf2.load(str(path))
        assert hgf2.threshold == pytest.approx(0.42)

    def test_load_roundtrip_preserves_predictions(self, tmp_path):
        cands = _make_candidates(20)
        hgf = self._trained()
        path = tmp_path / "model.pkl"
        hgf.save(str(path))

        hgf2 = HybridGeneFilter()
        hgf2.load(str(path))

        _, probs1, _ = hgf.predict(cands, batch_size=16)
        _, probs2, _ = hgf2.predict(cands, batch_size=16)
        np.testing.assert_allclose(probs1, probs2, rtol=1e-5)
