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

import numpy as np
import pytest

from src.traditional_methods import (
    add_combined_scores,
    build_codon_model,
    build_interpolated_markov_model,
    clear_imm_cache,
    find_orfs_candidates,
    normalize_all_orf_scores,
    predict_rbs_simple,
    score_codon_bias_ratio,
    score_imm_ratio,
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
            assert required.issubset(
                orf.keys()
            ), f"ORF missing keys: {required - orf.keys()}"

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
        score = score_codon_bias_ratio(
            self._CAG_SEQ, self._CODING_MODEL, self._BG_MODEL
        )
        assert isinstance(score, float)

    def test_empty_sequence_returns_zero(self):
        assert score_codon_bias_ratio("", self._CODING_MODEL, self._BG_MODEL) == 0.0

    def test_sequence_shorter_than_three_returns_zero(self):
        assert score_codon_bias_ratio("AT", self._CODING_MODEL, self._BG_MODEL) == 0.0

    def test_coding_like_sequence_scores_positive(self):
        score = score_codon_bias_ratio(
            self._CAG_SEQ, self._CODING_MODEL, self._BG_MODEL
        )
        assert score > 0

    def test_coding_sequence_scores_higher_than_noncoding(self):
        coding_score = score_codon_bias_ratio(
            self._CAG_SEQ, self._CODING_MODEL, self._BG_MODEL
        )
        bg_score = score_codon_bias_ratio(
            self._GGG_SEQ, self._CODING_MODEL, self._BG_MODEL
        )
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
                    f"Position {pos_idx}, context '{context}': "
                    f"probabilities sum to {total}"
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
        noncoding_imm = build_interpolated_markov_model(
            self._NONCODING_SEQS, max_order=2
        )
        score = score_imm_ratio(
            "ATG" + "CAG" * 10 + "TAA", coding_imm, noncoding_imm, max_order=2
        )
        assert isinstance(score, float)

    def test_sequence_shorter_than_three_returns_zero(self):
        coding_imm = build_interpolated_markov_model(self._CODING_SEQS, max_order=2)
        noncoding_imm = build_interpolated_markov_model(
            self._NONCODING_SEQS, max_order=2
        )
        assert score_imm_ratio("AT", coding_imm, noncoding_imm, max_order=2) == 0.0

    def test_coding_like_sequence_scores_higher_than_noncoding(self):
        coding_imm = build_interpolated_markov_model(self._CODING_SEQS, max_order=2)
        noncoding_imm = build_interpolated_markov_model(
            self._NONCODING_SEQS, max_order=2
        )
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
            scores = [
                score_imm_ratio(test_seq, coding, noncoding, max_order=2)
                for _ in range(100)
            ]
            results[name] = scores

        t1 = threading.Thread(target=run, args=("a", coding_a, noncoding_a))
        t2 = threading.Thread(target=run, args=("b", coding_b, noncoding_b))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert all(s > 0 for s in results["a"]), "Thread A: all scores should be positive"
        assert all(s < 0 for s in results["b"]), "Thread B: all scores should be negative"


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
