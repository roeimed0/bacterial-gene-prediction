"""
Unit tests for src/traditional_methods.py — core ORF detection and scoring.

Covers:
  - find_orfs_candidates()
  - predict_rbs_simple()
  - build_codon_model()
  - score_codon_bias_ratio()
  - build_interpolated_markov_model()
  - score_imm_ratio()
  - normalize_all_orf_scores()
  - add_combined_scores()
"""

import math

import numpy as np
import pytest

from src.traditional_methods import (
    _ASCII_TO_INT,
    _LOG_EPSILON,
    _NUMBA_AVAILABLE,
    _score_imm_fast,
    _seq_to_int_fast,
    add_combined_scores,
    build_codon_log_ratio_table,
    build_codon_model,
    build_flat_log_table,
    build_interpolated_markov_model,
    build_numba_log_table,
    clear_imm_cache,
    find_orfs_candidates,
    normalize_all_orf_scores,
    predict_rbs_simple,
    score_codon_bias_ratio,
    score_imm_ratio,
    select_training_flexible,
    select_training_glimmer,
)

# ---------------------------------------------------------------------------
# Module-level constants for synthetic test sequences
# ---------------------------------------------------------------------------

# 162 bp forward-strand sequence with exactly one detectable ORF.
# ATG at position 31 (1-indexed); TAA stop ends at position 132.
# ORF length = 102 bp (>= min_length default of 100).
# Flanks are all-C to prevent accidental start codons (ATG/GTG/TTG).
FORWARD_ONLY_SEQ = "C" * 30 + "ATG" + "CAG" * 32 + "TAA" + "C" * 30

# 162 bp sequence designed to have an ORF on the reverse (minus) strand.
# The minus-strand ORF = ATG + CAG*32 + TAA (102 bp).
# Its forward-strand representation is the reverse complement of that ORF.
# RC(ATG + CAG*32 + TAA) = TTA + CTG*32 + CAT
# Flanks are all-G (RC of C) to prevent extra start codons on either strand.
_RC_ORF_FORWARD = "TTA" + "CTG" * 32 + "CAT"
REVERSE_STRAND_SEQ = "G" * 30 + _RC_ORF_FORWARD + "G" * 30

# 120 bp sequence with a canonical Shine-Dalgarno (AGGAGG) placed 7 bp
# upstream of an ATG — within the optimal 6-8 bp spacing window.
# ATG is at position 44 (1-indexed), comfortably past the 20-bp upstream_length.
SD_UPSTREAM_SEQ = "A" * 30 + "AGGAGG" + "A" * 7 + "ATG" + "CAG" * 32 + "TAA"
SD_ATG_POSITION = 44  # 1-indexed position of the ATG start codon


# ---------------------------------------------------------------------------
# Helpers for scored / normalized ORF dicts
# ---------------------------------------------------------------------------


def _make_scored_orf(codon=1.0, imm=0.5, rbs=2.0, length=0.8, start=1.0):
    return {
        "codon_score": codon,
        "imm_score": imm,
        "rbs_score": rbs,
        "length_score": length,
        "start_score": start,
    }


def _make_normalized_orf(c=0.6, i=0.4, r=0.7, l=0.5, s=0.6):
    return {
        "codon_score_norm": c,
        "imm_score_norm": i,
        "rbs_score_norm": r,
        "length_score_norm": l,
        "start_score_norm": s,
    }


# ===========================================================================
# Tests: find_orfs_candidates()
# ===========================================================================


class TestFindOrfCandidates:
    def test_returns_list(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ)
        assert isinstance(orfs, list)

    def test_detects_at_least_one_forward_orf(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ)
        assert len(orfs) >= 1

    def test_forward_orf_has_atg_start_codon(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ)
        forward_orfs = [o for o in orfs if o["strand"] == "forward"]
        assert any(o["start_codon"] == "ATG" for o in forward_orfs)

    def test_detects_reverse_strand_orf(self):
        orfs = find_orfs_candidates(REVERSE_STRAND_SEQ)
        reverse_orfs = [o for o in orfs if o["strand"] == "reverse"]
        assert len(reverse_orfs) >= 1

    def test_orf_has_required_keys(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ)
        required = {
            "start",
            "end",
            "genome_start",
            "genome_end",
            "length",
            "frame",
            "strand",
            "start_codon",
            "sequence",
            "rbs_score",
        }
        for orf in orfs:
            assert required.issubset(orf.keys()), f"ORF missing keys: {required - orf.keys()}"

    def test_start_codon_is_valid(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ)
        valid = {"ATG", "GTG", "TTG"}
        for orf in orfs:
            assert orf["start_codon"] in valid

    def test_strand_is_forward_or_reverse(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ)
        for orf in orfs:
            assert orf["strand"] in {"forward", "reverse"}

    def test_all_orfs_meet_minimum_length(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ, min_length=100)
        for orf in orfs:
            assert orf["length"] >= 100

    def test_length_matches_coordinate_span(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ)
        for orf in orfs:
            assert orf["end"] - orf["start"] + 1 == orf["length"]

    def test_sequence_too_short_returns_empty_list(self):
        # 21 bp ORF — well below default min_length=100
        short_seq = "ATG" + "CAG" * 5 + "TAA"
        orfs = find_orfs_candidates(short_seq, min_length=100)
        assert orfs == []

    def test_rbs_score_is_numeric(self):
        orfs = find_orfs_candidates(FORWARD_ONLY_SEQ)
        for orf in orfs:
            assert isinstance(orf["rbs_score"], (int, float))


# ===========================================================================
# Tests: predict_rbs_simple()
# ===========================================================================


class TestPredictRbsSimple:
    def test_returns_dict(self):
        result = predict_rbs_simple(SD_UPSTREAM_SEQ, {"start": SD_ATG_POSITION})
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = predict_rbs_simple(SD_UPSTREAM_SEQ, {"start": SD_ATG_POSITION})
        required = {
            "rbs_score",
            "spacing_score",
            "motif_score",
            "best_sequence",
            "best_motif",
            "spacing",
        }
        assert required.issubset(result.keys())

    def test_rbs_score_is_float(self):
        result = predict_rbs_simple(SD_UPSTREAM_SEQ, {"start": SD_ATG_POSITION})
        assert isinstance(result["rbs_score"], float)

    def test_canonical_sd_upstream_yields_positive_rbs_score(self):
        # AGGAGG at optimal 7 bp spacing should produce a positive score
        result = predict_rbs_simple(SD_UPSTREAM_SEQ, {"start": SD_ATG_POSITION})
        assert result["rbs_score"] > 0

    def test_orf_too_close_to_sequence_start_returns_minimum_score(self):
        # ORF starting at position 2 has no upstream region (< upstream_length=20)
        short_seq = "ATG" + "CAG" * 32 + "TAA"
        result = predict_rbs_simple(short_seq, {"start": 2})
        assert result["rbs_score"] == -5.0


# ===========================================================================
# Tests: build_codon_model()
# ===========================================================================


class TestBuildCodonModel:
    _SEQS = [{"sequence": "ATG" + "CAG" * 10 + "TAA"}]

    def test_returns_dict(self):
        assert isinstance(build_codon_model(self._SEQS), dict)

    def test_empty_input_returns_empty_dict(self):
        assert build_codon_model([]) == {}

    def test_frequencies_sum_to_one(self):
        model = build_codon_model(self._SEQS)
        assert abs(sum(model.values()) - 1.0) < 1e-6

    def test_all_frequencies_non_negative(self):
        model = build_codon_model(self._SEQS)
        assert all(v >= 0.0 for v in model.values())

    def test_codon_keys_are_three_characters(self):
        model = build_codon_model(self._SEQS)
        assert all(len(k) == 3 for k in model)

    def test_n_containing_codons_excluded(self):
        seqs = [{"sequence": "ATGNNN" + "CAG" * 5 + "TAA"}]
        model = build_codon_model(seqs)
        assert all("N" not in k for k in model)

    def test_known_codon_appears_in_model(self):
        model = build_codon_model(self._SEQS)
        # CAG appears 10 times and should be the dominant codon
        assert "CAG" in model
        assert "ATG" in model

    def test_dominant_codon_has_highest_frequency(self):
        # Sequence is 10 CAG codons + ATG + TAA — CAG should dominate
        model = build_codon_model(self._SEQS)
        assert model["CAG"] == max(model.values())


# ===========================================================================
# Tests: score_codon_bias_ratio()
# ===========================================================================


class TestScoreCodonBiasRatio:
    # Simple models: coding favours CAG, background favours GGG
    _CODING_MODEL = {"ATG": 0.05, "CAG": 0.85, "TAA": 0.10}
    _BG_MODEL = {"ATG": 0.03, "CAG": 0.02, "TAA": 0.05, "GGG": 0.90}
    _CAG_SEQ = "ATG" + "CAG" * 20 + "TAA"  # matches coding model
    _GGG_SEQ = "GGG" * 20  # matches background model

    def test_returns_float(self):
        score = score_codon_bias_ratio(self._CAG_SEQ, self._CODING_MODEL, self._BG_MODEL)
        assert isinstance(score, float)

    def test_empty_sequence_returns_zero(self):
        assert score_codon_bias_ratio("", self._CODING_MODEL, self._BG_MODEL) == 0.0

    def test_sequence_shorter_than_three_returns_zero(self):
        assert score_codon_bias_ratio("AT", self._CODING_MODEL, self._BG_MODEL) == 0.0

    def test_coding_like_sequence_scores_positive(self):
        score = score_codon_bias_ratio(self._CAG_SEQ, self._CODING_MODEL, self._BG_MODEL)
        assert score > 0

    def test_coding_sequence_scores_higher_than_noncoding(self):
        coding_score = score_codon_bias_ratio(self._CAG_SEQ, self._CODING_MODEL, self._BG_MODEL)
        bg_score = score_codon_bias_ratio(self._GGG_SEQ, self._CODING_MODEL, self._BG_MODEL)
        assert coding_score > bg_score


# ===========================================================================
# Tests: build_interpolated_markov_model()
# ===========================================================================


class TestBuildInterpolatedMarkovModel:
    # Five identical sequences — enough observations to exceed min_observations=10
    _SEQS = ["ATG" + "CAG" * 20 + "TAA"] * 5

    def test_returns_list_of_length_3(self):
        imm = build_interpolated_markov_model(self._SEQS, max_order=2)
        assert isinstance(imm, list)
        assert len(imm) == 3

    def test_each_element_is_dict(self):
        imm = build_interpolated_markov_model(self._SEQS, max_order=2)
        for pos_model in imm:
            assert isinstance(pos_model, dict)

    def test_empty_input_returns_three_empty_dicts(self):
        imm = build_interpolated_markov_model([], max_order=2)
        assert imm == [{}, {}, {}]

    def test_probabilities_sum_to_one_per_context(self):
        imm = build_interpolated_markov_model(self._SEQS, max_order=1)
        for pos_idx, pos_model in enumerate(imm):
            for context, dist in pos_model.items():
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-6, (
                    f"Position {pos_idx}, context '{context}': " f"probabilities sum to {total}"
                )

    def test_all_probabilities_non_negative(self):
        imm = build_interpolated_markov_model(self._SEQS, max_order=2)
        for pos_model in imm:
            for dist in pos_model.values():
                assert all(p >= 0.0 for p in dist.values())

    def test_nucleotide_keys_are_single_characters(self):
        imm = build_interpolated_markov_model(self._SEQS, max_order=1)
        for pos_model in imm:
            for dist in pos_model.values():
                for nuc in dist:
                    assert len(nuc) == 1

    def test_numba_output_identical_to_python_reference(self):
        """Regression: Numba k-mer counter must produce byte-identical model to
        the pure-Python triple-nested loop for sequences without N."""
        from collections import defaultdict

        def _python_ref(seqs, max_order, min_obs=10):
            pm = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
            for s in seqs:
                for i in range(len(s)):
                    nuc, cp = s[i], i % 3
                    for order in range(min(i + 1, max_order + 1)):
                        ctx = "" if order == 0 else s[i - order : i]
                        pm[cp][ctx][nuc] += 1
            result = []
            for pos in range(3):
                probs = {}
                for ctx, counts in pm[pos].items():
                    total = sum(counts.values())
                    if total >= min_obs:
                        probs[ctx] = {n: c / total for n, c in counts.items()}
                result.append(probs)
            return result

        seqs = ["ATG" + "CAG" * 30 + "TAA"] * 10 + ["GGG" * 31] * 10
        max_order = 3
        old = _python_ref(seqs, max_order)
        new = build_interpolated_markov_model(seqs, max_order)

        for pos in range(3):
            assert set(old[pos]) == set(new[pos]), f"Context key mismatch at position {pos}"
            for ctx in old[pos]:
                for nuc, p_old in old[pos][ctx].items():
                    p_new = new[pos][ctx].get(nuc)
                    assert p_new is not None and abs(p_old - p_new) < 1e-12, (
                        f"Prob mismatch at pos={pos} ctx={ctx!r} nuc={nuc}: "
                        f"old={p_old} new={p_new}"
                    )


# ===========================================================================
# Tests: score_imm_ratio()
# ===========================================================================


class TestScoreImmRatio:
    _CODING_SEQS = ["ATG" + "CAG" * 30 + "TAA"] * 5
    _NONCODING_SEQS = ["GGG" * 31] * 5

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the IMM LRU cache before each test to prevent cross-test pollution."""
        clear_imm_cache()
        yield
        clear_imm_cache()

    def test_returns_float(self):
        coding_imm = build_interpolated_markov_model(self._CODING_SEQS, max_order=2)
        noncoding_imm = build_interpolated_markov_model(self._NONCODING_SEQS, max_order=2)
        score = score_imm_ratio("ATG" + "CAG" * 10 + "TAA", coding_imm, noncoding_imm, max_order=2)
        assert isinstance(score, float)

    def test_sequence_shorter_than_three_returns_zero(self):
        coding_imm = build_interpolated_markov_model(self._CODING_SEQS, max_order=2)
        noncoding_imm = build_interpolated_markov_model(self._NONCODING_SEQS, max_order=2)
        assert score_imm_ratio("AT", coding_imm, noncoding_imm, max_order=2) == 0.0

    def test_coding_like_sequence_scores_higher_than_noncoding(self):
        coding_imm = build_interpolated_markov_model(self._CODING_SEQS, max_order=2)
        noncoding_imm = build_interpolated_markov_model(self._NONCODING_SEQS, max_order=2)
        coding_score = score_imm_ratio(
            "ATG" + "CAG" * 20 + "TAA", coding_imm, noncoding_imm, max_order=2
        )
        bg_score = score_imm_ratio("GGG" * 20, coding_imm, noncoding_imm, max_order=2)
        assert coding_score > bg_score

    def test_sequential_scoring_no_cache_poisoning_issue_68(self):
        """Regression: scoring genome B after genome A must not return genome A's
        cached probabilities.  Previously _GLOBAL_CODING_IMM was set once and the
        lru_cache keyed only on the k-mer string, so the second model's scores were
        silently wrong (sign-flipped in the worst case)."""
        # Model A: ATG is 'coding', CCC is 'noncoding'
        coding_a = build_interpolated_markov_model(["ATG" * 50], max_order=2)
        noncoding_a = build_interpolated_markov_model(["CCC" * 50], max_order=2)
        # Model B: opposite — ATG is 'noncoding', CCC is 'coding'
        coding_b = build_interpolated_markov_model(["CCC" * 50], max_order=2)
        noncoding_b = build_interpolated_markov_model(["ATG" * 50], max_order=2)

        test_seq = "ATG" * 50

        score_a = score_imm_ratio(test_seq, coding_a, noncoding_a, max_order=2)
        # Do NOT clear cache — this is the batch-mode scenario that triggered the bug
        score_b = score_imm_ratio(test_seq, coding_b, noncoding_b, max_order=2)

        assert score_a > 0, "Model A should score ATG-rich sequence as coding"
        assert score_b < 0, "Model B should score ATG-rich sequence as non-coding"
        assert score_a != score_b, "Two opposite models must produce different scores"

    def test_concurrent_scoring_thread_safe_issue_68(self):
        """Regression: concurrent score_imm_ratio calls must not corrupt each other's
        results via the previously shared _GLOBAL_CODING_IMM / _GLOBAL_NONCODING_IMM
        globals."""
        import threading

        coding_a = build_interpolated_markov_model(["ATG" * 50], max_order=2)
        noncoding_a = build_interpolated_markov_model(["CCC" * 50], max_order=2)
        coding_b = build_interpolated_markov_model(["CCC" * 50], max_order=2)
        noncoding_b = build_interpolated_markov_model(["ATG" * 50], max_order=2)

        test_seq = "ATG" * 50
        results = {}

        def run(name, coding, noncoding):
            scores = [score_imm_ratio(test_seq, coding, noncoding, max_order=2) for _ in range(100)]
            results[name] = scores

        t1 = threading.Thread(target=run, args=("a", coding_a, noncoding_a))
        t2 = threading.Thread(target=run, args=("b", coding_b, noncoding_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert all(s > 0 for s in results["a"]), "Thread A: all scores should be positive"
        assert all(s < 0 for s in results["b"]), "Thread B: all scores should be negative"


# ===========================================================================
# Tests: build_flat_log_table()
# ===========================================================================


class TestBuildFlatLogTable:
    """Tests for the pre-baked IMM log-probability table (Fix A optimisation)."""

    _CODING_SEQS = ["ATG" + "CAG" * 30 + "TAA"] * 5
    _NONCODING_SEQS = ["GGG" * 31] * 5
    _MAX_ORDER = 2

    @pytest.fixture(scope="class")
    def imm_model(self):
        return build_interpolated_markov_model(self._CODING_SEQS, self._MAX_ORDER)

    @pytest.fixture(scope="class")
    def log_table(self, imm_model):
        return build_flat_log_table(imm_model, self._MAX_ORDER)

    # ── Structure ────────────────────────────────────────────────────────────

    def test_returns_list_of_three_dicts(self, log_table):
        assert isinstance(log_table, list)
        assert len(log_table) == 3
        assert all(isinstance(d, dict) for d in log_table)

    def test_all_values_are_non_positive(self, log_table):
        """Log-probabilities must be ≤ 0 (probabilities ≤ 1)."""
        for pos_dict in log_table:
            for val in pos_dict.values():
                assert val <= 0.0, f"Log-prob {val} > 0 — probability > 1"

    def test_all_values_are_at_least_log_epsilon(self, log_table):
        """No value should be below log(1e-10) — the EPSILON floor."""
        for pos_dict in log_table:
            for val in pos_dict.values():
                assert val >= _LOG_EPSILON - 1e-12

    def test_single_nucleotide_keys_present(self, log_table):
        """Empty-context keys (length-1: just the nucleotide) must exist in every position."""
        for nuc in ("A", "C", "G", "T"):
            for pos_dict in log_table:
                assert nuc in pos_dict, f"Single-nucleotide key '{nuc}' missing"

    def test_key_lengths_up_to_max_order_plus_one(self, log_table):
        """Keys should be at most max_order+1 characters (context + nucleotide)."""
        for pos_dict in log_table:
            for key in pos_dict:
                assert len(key) <= self._MAX_ORDER + 1

    def test_log_value_matches_manual_log_of_probability(self, imm_model, log_table):
        """Pre-baked log value must equal math.log(probability) from the model directly."""
        from src.traditional_methods import _interpolate_prob

        for pos in range(3):
            for nuc in ("A", "C", "G", "T"):
                key = nuc  # empty context
                expected = math.log(max(_interpolate_prob(imm_model, pos, "", nuc), 1e-10))
                assert abs(log_table[pos][key] - expected) < 1e-12

    def test_table_size_grows_with_order(self):
        """Higher max_order must produce a larger table."""
        model = build_interpolated_markov_model(self._CODING_SEQS, max_order=3)
        table_2 = build_flat_log_table(model, max_order=2)
        table_3 = build_flat_log_table(model, max_order=3)
        assert len(table_3[0]) > len(table_2[0])

    def test_empty_model_returns_fallback_values(self):
        """An empty model (no training data) should fill the table with log(fallback)."""
        empty_model = [{}, {}, {}]
        table = build_flat_log_table(empty_model, max_order=1)
        for pos_dict in table:
            for val in pos_dict.values():
                assert val == pytest.approx(math.log(0.25))


# ===========================================================================
# Tests: _score_imm_fast()
# ===========================================================================


class TestScoreImmFast:
    """Tests for the fast IMM scoring path using pre-baked log tables."""

    _CODING_SEQS = ["ATG" + "CAG" * 30 + "TAA"] * 5
    _NONCODING_SEQS = ["GGG" * 31] * 5
    _MAX_ORDER = 2

    @pytest.fixture(scope="class")
    def models(self):
        coding_imm = build_interpolated_markov_model(self._CODING_SEQS, self._MAX_ORDER)
        noncoding_imm = build_interpolated_markov_model(self._NONCODING_SEQS, self._MAX_ORDER)
        coding_log = build_flat_log_table(coding_imm, self._MAX_ORDER)
        noncoding_log = build_flat_log_table(noncoding_imm, self._MAX_ORDER)
        return coding_imm, noncoding_imm, coding_log, noncoding_log

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_sequence_shorter_than_three_returns_zero(self, models):
        _, _, c_log, nc_log = models
        assert _score_imm_fast("AT", c_log, nc_log, self._MAX_ORDER) == 0.0

    def test_empty_sequence_returns_zero(self, models):
        _, _, c_log, nc_log = models
        assert _score_imm_fast("", c_log, nc_log, self._MAX_ORDER) == 0.0

    def test_returns_float(self, models):
        _, _, c_log, nc_log = models
        result = _score_imm_fast("ATG" + "CAG" * 10 + "TAA", c_log, nc_log, self._MAX_ORDER)
        assert isinstance(result, float)

    # ── Correctness vs score_imm_ratio ──────────────────────────────────────

    def test_identical_to_score_imm_ratio_on_coding_sequence(self, models):
        """Regression: fast path must produce bit-identical scores to old path."""
        clear_imm_cache()
        coding_imm, noncoding_imm, c_log, nc_log = models
        seq = "ATG" + "CAG" * 20 + "TAA"
        old = score_imm_ratio(seq, coding_imm, noncoding_imm, self._MAX_ORDER)
        new = _score_imm_fast(seq, c_log, nc_log, self._MAX_ORDER)
        assert old == pytest.approx(new, abs=1e-12)

    def test_identical_to_score_imm_ratio_on_noncoding_sequence(self, models):
        clear_imm_cache()
        coding_imm, noncoding_imm, c_log, nc_log = models
        seq = "GGG" * 30
        old = score_imm_ratio(seq, coding_imm, noncoding_imm, self._MAX_ORDER)
        new = _score_imm_fast(seq, c_log, nc_log, self._MAX_ORDER)
        assert old == pytest.approx(new, abs=1e-12)

    def test_identical_on_100_random_sequences(self, models):
        """Regression: scores must match for all sequence types."""
        import random

        random.seed(42)
        clear_imm_cache()
        coding_imm, noncoding_imm, c_log, nc_log = models
        for _ in range(100):
            length = random.randint(30, 300)
            seq = "".join(random.choice("ACGT") for _ in range(length))
            old = score_imm_ratio(seq, coding_imm, noncoding_imm, self._MAX_ORDER)
            new = _score_imm_fast(seq, c_log, nc_log, self._MAX_ORDER)
            assert old == pytest.approx(new, abs=1e-12), f"Mismatch on seq[:20]={seq[:20]!r}"

    def test_coding_scores_higher_than_noncoding(self, models):
        """Sanity: ATG-rich coding-like sequence scores higher than GGG noncoding."""
        _, _, c_log, nc_log = models
        coding_score = _score_imm_fast("ATG" + "CAG" * 20 + "TAA", c_log, nc_log, self._MAX_ORDER)
        noncoding_score = _score_imm_fast("GGG" * 20, c_log, nc_log, self._MAX_ORDER)
        assert coding_score > noncoding_score

    def test_unknown_nucleotide_uses_fallback(self, models):
        """Sequences with N should not crash — unknown k-mer falls back to LOG_EPSILON."""
        _, _, c_log, nc_log = models
        result = _score_imm_fast("ATGNNNNTAA" * 5, c_log, nc_log, self._MAX_ORDER)
        assert isinstance(result, float)
        assert not math.isnan(result) and not math.isinf(result)


# ===========================================================================
# Tests: _seq_to_int_fast() and build_numba_log_table()
# ===========================================================================


class TestSeqToIntFast:
    """Tests for the vectorised DNA→int32 encoder."""

    def test_known_nucleotides_map_correctly(self):
        arr = _seq_to_int_fast("ACGT")
        assert list(arr) == [0, 1, 2, 3]

    def test_unknown_nucleotide_maps_to_four(self):
        arr = _seq_to_int_fast("N")
        assert arr[0] == 4

    def test_output_dtype_is_int32(self):
        import numpy as np

        arr = _seq_to_int_fast("ACGT")
        assert arr.dtype == np.int32

    def test_output_length_matches_input(self):
        seq = "ATGCATGCATGC"
        assert len(_seq_to_int_fast(seq)) == len(seq)

    def test_lut_constant_covers_all_four_bases(self):
        assert _ASCII_TO_INT[65] == 0  # A
        assert _ASCII_TO_INT[67] == 1  # C
        assert _ASCII_TO_INT[71] == 2  # G
        assert _ASCII_TO_INT[84] == 3  # T

    def test_lut_default_is_four_for_unknown(self):
        # Every entry not explicitly set should be 4
        import numpy as np

        other_indices = [i for i in range(128) if i not in (65, 67, 71, 84)]
        assert all(_ASCII_TO_INT[i] == 4 for i in other_indices)

    def test_round_trip_on_coding_sequence(self):
        seq = "ATG" + "CAG" * 10 + "TAA"
        arr = _seq_to_int_fast(seq)
        assert len(arr) == len(seq)
        assert arr[0] == 0  # A
        assert arr[1] == 3  # T
        assert arr[2] == 2  # G


class TestBuildNumbaLogTable:
    """Tests for the integer-indexed numpy log table."""

    _CODING_SEQS = ["ATG" + "CAG" * 30 + "TAA"] * 5
    _MAX_ORDER = 2

    @pytest.fixture(scope="class")
    def log_table(self):
        imm = build_interpolated_markov_model(self._CODING_SEQS, self._MAX_ORDER)
        return build_flat_log_table(imm, self._MAX_ORDER)

    @pytest.fixture(scope="class")
    def numba_table(self, log_table):
        return build_numba_log_table(log_table, self._MAX_ORDER)

    def test_returns_numpy_array(self, numba_table):
        import numpy as np

        assert isinstance(numba_table, np.ndarray)

    def test_shape_is_positions_by_entries(self, numba_table):
        # 3 positions, (4^(max_order+2)-1)//3 entries
        expected_cols = (4 ** (self._MAX_ORDER + 2) - 1) // 3
        assert numba_table.shape == (3, expected_cols)

    def test_all_values_non_positive(self, numba_table):
        assert (numba_table <= 0).all()

    def test_all_values_at_least_log_epsilon(self, numba_table):
        assert (numba_table >= _LOG_EPSILON - 1e-12).all()

    def test_single_nucleotide_entries_match_flat_table(self, log_table, numba_table):
        """Index for single-char key 'A' (no context) must match flat table value."""
        # key='A': len=1, offset=(4^1-1)//3=1, idx=1+0=1 (A=0)
        assert abs(numba_table[0, 1] - log_table[0].get("A", _LOG_EPSILON)) < 1e-12


@pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
class TestScoreImmNumba:
    """Regression tests for the Numba JIT scoring path."""

    _CODING_SEQS = ["ATG" + "CAG" * 30 + "TAA"] * 5
    _NONCODING_SEQS = ["GGG" * 31] * 5
    _MAX_ORDER = 2

    @pytest.fixture(scope="class")
    def tables(self):
        from src.traditional_methods import _score_imm_numba

        coding_imm = build_interpolated_markov_model(self._CODING_SEQS, self._MAX_ORDER)
        noncoding_imm = build_interpolated_markov_model(self._NONCODING_SEQS, self._MAX_ORDER)
        flat_c = build_flat_log_table(coding_imm, self._MAX_ORDER)
        flat_nc = build_flat_log_table(noncoding_imm, self._MAX_ORDER)
        c_tbl = build_numba_log_table(flat_c, self._MAX_ORDER)
        nc_tbl = build_numba_log_table(flat_nc, self._MAX_ORDER)
        return coding_imm, noncoding_imm, c_tbl, nc_tbl, _score_imm_numba

    def test_short_sequence_returns_zero(self, tables):
        import numpy as np

        _, _, c_tbl, nc_tbl, fn = tables
        arr = np.array([0, 1], dtype=np.int32)
        assert fn(arr, c_tbl, nc_tbl, self._MAX_ORDER, _LOG_EPSILON) == 0.0

    def test_identical_to_score_imm_ratio_on_coding_seq(self, tables):
        coding_imm, noncoding_imm, c_tbl, nc_tbl, fn = tables
        seq = "ATG" + "CAG" * 20 + "TAA"
        clear_imm_cache()
        old = score_imm_ratio(seq, coding_imm, noncoding_imm, self._MAX_ORDER)
        new = float(fn(_seq_to_int_fast(seq), c_tbl, nc_tbl, self._MAX_ORDER, _LOG_EPSILON))
        assert old == pytest.approx(new, abs=1e-12)

    def test_identical_to_score_imm_ratio_on_noncoding_seq(self, tables):
        coding_imm, noncoding_imm, c_tbl, nc_tbl, fn = tables
        seq = "GGG" * 30
        clear_imm_cache()
        old = score_imm_ratio(seq, coding_imm, noncoding_imm, self._MAX_ORDER)
        new = float(fn(_seq_to_int_fast(seq), c_tbl, nc_tbl, self._MAX_ORDER, _LOG_EPSILON))
        assert old == pytest.approx(new, abs=1e-12)

    def test_identical_on_100_random_sequences(self, tables):
        import random

        random.seed(99)
        coding_imm, noncoding_imm, c_tbl, nc_tbl, fn = tables
        clear_imm_cache()
        for _ in range(100):
            seq = "".join(random.choice("ACGT") for _ in range(random.randint(30, 300)))
            old = score_imm_ratio(seq, coding_imm, noncoding_imm, self._MAX_ORDER)
            new = float(fn(_seq_to_int_fast(seq), c_tbl, nc_tbl, self._MAX_ORDER, _LOG_EPSILON))
            assert old == pytest.approx(new, abs=1e-12), f"mismatch on {seq[:20]!r}"

    def test_coding_scores_higher_than_noncoding(self, tables):
        _, _, c_tbl, nc_tbl, fn = tables
        s_cod = float(
            fn(
                _seq_to_int_fast("ATG" + "CAG" * 20 + "TAA"),
                c_tbl,
                nc_tbl,
                self._MAX_ORDER,
                _LOG_EPSILON,
            )
        )
        s_non = float(
            fn(_seq_to_int_fast("GGG" * 20), c_tbl, nc_tbl, self._MAX_ORDER, _LOG_EPSILON)
        )
        assert s_cod > s_non

    def test_unknown_nucleotide_no_crash(self, tables):
        _, _, c_tbl, nc_tbl, fn = tables
        result = float(
            fn(_seq_to_int_fast("ATGNNNNTAA" * 5), c_tbl, nc_tbl, self._MAX_ORDER, _LOG_EPSILON)
        )
        assert not math.isnan(result) and not math.isinf(result)


# ===========================================================================
# Tests: build_codon_log_ratio_table() and _score_codon_bias_numba()
# ===========================================================================


class TestBuildCodonLogRatioTable:
    _CODING_SEQS = [{"sequence": "ATG" + "CAG" * 30 + "TAA"}] * 5
    _NONCODING_SEQS = [{"sequence": "GGG" * 31}] * 5

    @pytest.fixture(scope="class")
    def models(self):
        c = build_codon_model(self._CODING_SEQS)
        bg = build_codon_model(self._NONCODING_SEQS)
        return c, bg

    @pytest.fixture(scope="class")
    def table(self, models):
        return build_codon_log_ratio_table(*models)

    def test_returns_numpy_array(self, table):
        import numpy as np

        assert isinstance(table, np.ndarray)

    def test_shape_is_64(self, table):
        assert table.shape == (64,)

    def test_all_values_are_finite(self, table):
        import numpy as np

        assert np.all(np.isfinite(table))

    def test_neutral_for_unseen_codon(self):
        # Empty models → all unseen → all entries = log(1e-4) - log(1e-4) = 0.0
        table = build_codon_log_ratio_table({}, {})
        assert (table == 0.0).all()

    def test_known_codon_has_nonzero_ratio(self, table):
        # CAG is in coding but not noncoding → ratio should be nonzero
        assert any(v != 0.0 for v in table)

    def test_atg_index(self, models):
        # ATG: A=0, T=3, G=2 → 0*16 + 3*4 + 2 = 14
        c, bg = models
        table = build_codon_log_ratio_table(c, bg)
        c_freq = c.get("ATG", 1e-4)
        bg_freq = bg.get("ATG", 1e-4)
        expected = math.log(c_freq) - math.log(bg_freq)
        assert abs(table[14] - expected) < 1e-12


@pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
class TestScoreCodonBiasNumba:
    _CODING_SEQS = [{"sequence": "ATG" + "CAG" * 30 + "TAA"}] * 5
    _NONCODING_SEQS = [{"sequence": "GGG" * 31}] * 5

    @pytest.fixture(scope="class")
    def setup(self):
        from src.traditional_methods import _score_codon_bias_numba

        c = build_codon_model(self._CODING_SEQS)
        bg = build_codon_model(self._NONCODING_SEQS)
        tbl = build_codon_log_ratio_table(c, bg)
        return c, bg, tbl, _score_codon_bias_numba

    def test_empty_sequence_returns_zero(self, setup):
        import numpy as np

        _, _, tbl, fn = setup
        assert fn(np.array([], dtype=np.int32), tbl) == 0.0

    def test_all_n_returns_zero(self, setup):
        _, _, tbl, fn = setup
        arr = _seq_to_int_fast("NNN" * 10)
        assert fn(arr, tbl) == 0.0

    def test_returns_float(self, setup):
        _, _, tbl, fn = setup
        result = fn(_seq_to_int_fast("ATG" + "CAG" * 10 + "TAA"), tbl)
        assert isinstance(float(result), float)

    def test_identical_to_score_codon_bias_ratio_on_coding_seq(self, setup):
        """Regression: Numba path must produce same result as Python fallback."""
        c, bg, tbl, fn = setup
        seq = "ATG" + "CAG" * 20 + "TAA"
        old = score_codon_bias_ratio(seq, c, bg)
        new = float(fn(_seq_to_int_fast(seq), tbl))
        assert old == pytest.approx(new, abs=1e-10)

    def test_identical_to_score_codon_bias_ratio_on_noncoding_seq(self, setup):
        c, bg, tbl, fn = setup
        seq = "GGG" * 30
        old = score_codon_bias_ratio(seq, c, bg)
        new = float(fn(_seq_to_int_fast(seq), tbl))
        assert old == pytest.approx(new, abs=1e-10)

    def test_identical_on_100_random_sequences(self, setup):
        import random

        random.seed(7)
        c, bg, tbl, fn = setup
        for _ in range(100):
            seq = "".join(random.choice("ACGT") for _ in range(random.randint(9, 300)))
            old = score_codon_bias_ratio(seq, c, bg)
            new = float(fn(_seq_to_int_fast(seq), tbl))
            assert old == pytest.approx(new, abs=1e-10), f"mismatch on {seq[:20]!r}"

    def test_coding_scores_higher_than_noncoding(self, setup):
        _, _, tbl, fn = setup
        s_cod = float(fn(_seq_to_int_fast("ATG" + "CAG" * 20 + "TAA"), tbl))
        s_non = float(fn(_seq_to_int_fast("GGG" * 20), tbl))
        assert s_cod > s_non


# ===========================================================================
# Tests: normalize_all_orf_scores()
# ===========================================================================


class TestNormalizeAllOrfScores:
    def _make_orfs(self):
        return [
            _make_scored_orf(codon=1.0, imm=0.5, rbs=2.0, length=0.8, start=1.0),
            _make_scored_orf(codon=0.5, imm=0.3, rbs=1.0, length=0.4, start=0.7),
            _make_scored_orf(codon=1.5, imm=0.8, rbs=3.0, length=1.2, start=1.0),
        ]

    def test_returns_list(self):
        result = normalize_all_orf_scores(self._make_orfs())
        assert isinstance(result, list)

    def test_normalized_keys_added(self):
        orfs = self._make_orfs()
        normalize_all_orf_scores(orfs)
        expected = {
            "codon_score_norm",
            "imm_score_norm",
            "rbs_score_norm",
            "length_score_norm",
            "start_score_norm",
        }
        for orf in orfs:
            assert expected.issubset(orf.keys())

    def test_normalized_scores_have_near_zero_mean(self):
        orfs = self._make_orfs()
        normalize_all_orf_scores(orfs)
        codon_norms = [o["codon_score_norm"] for o in orfs]
        assert abs(np.mean(codon_norms)) < 1e-6

    def test_identical_scores_normalize_to_zero(self):
        # When std == 0, z-score should be 0 for all
        orfs = [_make_scored_orf(codon=1.0) for _ in range(3)]
        normalize_all_orf_scores(orfs)
        for orf in orfs:
            assert orf["codon_score_norm"] == 0.0

    def test_raw_scores_preserved_after_normalization(self):
        orfs = self._make_orfs()
        originals = [o["codon_score"] for o in orfs]
        normalize_all_orf_scores(orfs)
        for orf, original in zip(orfs, originals):
            assert orf["codon_score"] == original


# ===========================================================================
# Tests: add_combined_scores()
# ===========================================================================


class TestAddCombinedScores:
    def test_returns_list(self):
        orfs = [_make_normalized_orf()]
        assert isinstance(add_combined_scores(orfs), list)

    def test_combined_score_key_added(self):
        orfs = [_make_normalized_orf() for _ in range(3)]
        add_combined_scores(orfs)
        for orf in orfs:
            assert "combined_score" in orf

    def test_combined_score_is_float(self):
        orfs = [_make_normalized_orf()]
        add_combined_scores(orfs)
        assert isinstance(orfs[0]["combined_score"], float)

    def test_combined_score_equals_default_weighted_sum(self):
        from src.config import SCORE_WEIGHTS

        orf = _make_normalized_orf(c=1.0, i=2.0, r=3.0, l=4.0, s=5.0)
        add_combined_scores([orf])
        expected = (
            1.0 * SCORE_WEIGHTS["codon"]
            + 2.0 * SCORE_WEIGHTS["imm"]
            + 3.0 * SCORE_WEIGHTS["rbs"]
            + 4.0 * SCORE_WEIGHTS["length"]
            + 5.0 * SCORE_WEIGHTS["start"]
        )
        assert abs(orf["combined_score"] - expected) < 1e-9

    def test_custom_weights_applied(self):
        weights = {
            "codon": 2.0,
            "imm": 0.0,
            "rbs": 0.0,
            "length": 0.0,
            "start": 0.0,
        }
        orf = _make_normalized_orf(c=3.0, i=99.0, r=99.0, l=99.0, s=99.0)
        add_combined_scores([orf], weights=weights)
        assert abs(orf["combined_score"] - 6.0) < 1e-9  # 2.0 * 3.0

    def test_higher_component_scores_yield_higher_combined_score(self):
        orf_low = _make_normalized_orf(c=0.0, i=0.0, r=0.0, l=0.0, s=0.0)
        orf_high = _make_normalized_orf(c=1.0, i=1.0, r=1.0, l=1.0, s=1.0)
        add_combined_scores([orf_low, orf_high])
        assert orf_high["combined_score"] > orf_low["combined_score"]


# ===========================================================================
# Tests: select_training_glimmer() and select_training_flexible() — issue #59
# ===========================================================================


def _make_orf(genome_start, genome_end, length=None, strand="forward", start_codon="ATG"):
    if length is None:
        length = genome_end - genome_start + 1
    return {
        "start": genome_start,
        "end": genome_end,
        "genome_start": genome_start,
        "genome_end": genome_end,
        "length": length,
        "strand": strand,
        "start_codon": start_codon,
    }


class TestSelectTrainingGlimmer:
    def test_returns_list(self):
        orfs = [_make_orf(1, 500, 500)]
        assert isinstance(select_training_glimmer(orfs, min_length=300), list)

    def test_empty_input_returns_empty(self):
        assert select_training_glimmer([], min_length=300) == []

    def test_short_orfs_excluded(self):
        orfs = [_make_orf(1, 100, 100)]
        assert select_training_glimmer(orfs, min_length=300) == []

    def test_single_long_orf_selected(self):
        orf = _make_orf(1, 500, 500)
        result = select_training_glimmer([orf], min_length=300)
        assert len(result) == 1
        assert result[0]["genome_start"] == 1

    def test_non_overlapping_orfs_all_selected(self):
        orfs = [
            _make_orf(1, 400, 400),
            _make_orf(500, 900, 400),
            _make_orf(1000, 1400, 400),
        ]
        result = select_training_glimmer(orfs, min_length=300)
        assert len(result) == 3

    def test_overlapping_shorter_orf_excluded(self):
        long_orf = _make_orf(1, 800, 800)
        short_orf = _make_orf(400, 700, 300)  # overlaps long_orf
        result = select_training_glimmer([long_orf, short_orf], min_length=300)
        assert len(result) == 1
        assert result[0]["genome_start"] == 1

    def test_respects_max_training_size(self):
        orfs = [_make_orf(i * 500, i * 500 + 400, 400) for i in range(10)]
        result = select_training_glimmer(orfs, min_length=300, max_training_size=3)
        assert len(result) == 3

    def test_selects_longest_first(self):
        """Longest ORF wins when two intervals overlap."""
        short = _make_orf(1, 400, 400)
        long_ = _make_orf(200, 900, 700)  # overlaps short, but longer
        result = select_training_glimmer([short, long_], min_length=300)
        assert len(result) == 1
        assert result[0]["genome_start"] == 200  # the longer one

    def test_no_two_selected_orfs_overlap(self):
        """Invariant: every pair of selected intervals must be non-overlapping."""
        import random

        random.seed(0)
        orfs, pos = [], 1
        for _ in range(500):
            length = random.randint(100, 2000)
            start = pos + random.randint(0, 200)
            orfs.append(_make_orf(start, start + length - 1, length))
            pos = start + length

        result = select_training_glimmer(orfs, min_length=300)
        intervals = sorted(
            (o.get("genome_start", o["start"]), o.get("genome_end", o["end"])) for o in result
        )
        for i in range(len(intervals) - 1):
            s1, e1 = intervals[i]
            s2, e2 = intervals[i + 1]
            assert e1 < s2, f"Overlap detected: [{s1},{e1}] and [{s2},{e2}]"


class TestSelectTrainingFlexible:
    def test_returns_list(self):
        orfs = [_make_orf(1, 500, 500)]
        assert isinstance(select_training_flexible(orfs, min_length=300), list)

    def test_empty_input_returns_empty(self):
        assert select_training_flexible([], min_length=300) == []

    def test_out_of_range_orfs_excluded(self):
        short = _make_orf(1, 100, 100)  # below min_length
        long_ = _make_orf(1, 30000, 30000)  # above max_length
        result = select_training_flexible([short, long_], min_length=300, max_length=2400)
        assert result == []

    def test_non_overlapping_orfs_all_selected(self):
        orfs = [
            _make_orf(1, 400, 400),
            _make_orf(500, 900, 400),
            _make_orf(1000, 1400, 400),
        ]
        result = select_training_flexible(orfs, min_length=300, target_size=10)
        assert len(result) == 3

    def test_heavy_overlap_excluded(self):
        base = _make_orf(1, 800, 800)
        # Overlaps 799/800 = 99.9% → exceeds max_overlap_fraction=0.3
        heavy = _make_orf(2, 801, 800)
        result = select_training_flexible([base, heavy], min_length=300, max_overlap_fraction=0.3)
        assert len(result) == 1

    def test_light_overlap_included(self):
        base = _make_orf(1, 800, 800)
        # 20 bp overlap on a 400 bp orf = 5% → below 0.3
        light = _make_orf(781, 1180, 400)
        result = select_training_flexible(
            [base, light], min_length=300, max_overlap_fraction=0.3, target_size=10
        )
        assert len(result) == 2

    def test_respects_target_size(self):
        orfs = [_make_orf(i * 500, i * 500 + 400, 400) for i in range(20)]
        result = select_training_flexible(orfs, min_length=300, target_size=5)
        assert len(result) == 5

    def test_max_overlap_fraction_never_exceeded(self):
        """Invariant: every pair of same-strand selected intervals respects max_overlap_fraction."""
        import random

        random.seed(42)
        max_frac = 0.3
        orfs, pos = [], 1
        for _ in range(500):
            length = random.randint(300, 2400)
            start = pos + random.randint(0, 100)
            orfs.append(_make_orf(start, start + length - 1, length))
            pos = start + length

        result = select_training_flexible(orfs, min_length=300, max_overlap_fraction=max_frac)
        for i, a in enumerate(result):
            sa = a.get("genome_start", a["start"])
            ea = a.get("genome_end", a["end"])
            if sa > ea:
                sa, ea = ea, sa
            for b in result[i + 1 :]:
                if b.get("strand", "forward") != a.get("strand", "forward"):
                    continue
                sb = b.get("genome_start", b["start"])
                eb = b.get("genome_end", b["end"])
                if sb > eb:
                    sb, eb = eb, sb
                overlap = max(0, min(ea, eb) - max(sa, sb) + 1)
                frac_a = overlap / a["length"]
                frac_b = overlap / b["length"]
                assert (
                    frac_a <= max_frac + 1e-9
                ), f"Overlap fraction {frac_a:.3f} exceeds {max_frac}"
                assert (
                    frac_b <= max_frac + 1e-9
                ), f"Overlap fraction {frac_b:.3f} exceeds {max_frac}"
