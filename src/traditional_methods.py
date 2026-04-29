"""
Traditional Gene Prediction Methods for Bacterial Genomes

This module implements classic (non-ML) approaches to bacterial gene prediction:
- ORF (Open Reading Frame) detection
- RBS (Ribosome Binding Site) prediction
- Codon usage bias analysis
- Interpolated Markov Models (IMM)
- Start codon and length scoring
- Training set selection strategies

These methods form the foundation of programs like Glimmer, GeneMark, and Prodigal.
"""

import bisect
import itertools
import math
import time
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import numba as _numba

    @_numba.njit(cache=True)
    def _score_imm_numba(
        seq_arr: np.ndarray,
        coding_t: np.ndarray,
        noncoding_t: np.ndarray,
        max_order: int,
        log_epsilon: float,
    ) -> float:
        """Numba-JIT IMM scoring. seq_arr: int32 array (A=0,C=1,G=2,T=3, other=4)."""
        n = len(seq_arr)
        if n < 3:
            return 0.0
        c_total = nc_total = 0.0
        for i in range(n):
            start = i - max_order if i >= max_order else 0
            kmer_int = 0
            valid = True
            for j in range(start, i + 1):
                nj = seq_arr[j]
                if nj >= 4:
                    valid = False
                    break
                kmer_int = kmer_int * 4 + nj
            if valid:
                L = i - start + 1
                offset = (4**L - 1) // 3
                idx = offset + kmer_int
                c_total += coding_t[i % 3, idx]
                nc_total += noncoding_t[i % 3, idx]
            else:
                c_total += log_epsilon
                nc_total += log_epsilon
        return (c_total - nc_total) / n

    @_numba.njit(cache=True)
    def _score_codon_bias_numba(
        seq_arr: np.ndarray,
        log_ratio_table: np.ndarray,
    ) -> float:
        """Numba-JIT codon bias log-ratio scoring.

        seq_arr        : int32 array (A=0, C=1, G=2, T=3, other=4)
        log_ratio_table: float64 array of shape (64,) indexed by
                         a*16 + b*4 + c  (base-4 encoding of the codon)
        """
        n = len(seq_arr)
        score = 0.0
        count = 0
        for i in range(0, n - 2, 3):
            a = seq_arr[i]
            b = seq_arr[i + 1]
            c = seq_arr[i + 2]
            if a < 4 and b < 4 and c < 4:
                score += log_ratio_table[a * 16 + b * 4 + c]
                count += 1
        if count == 0:
            return 0.0
        return float(score / count)

    @_numba.njit(cache=True)
    def _scan_orfs_numba(
        seq_arr: np.ndarray,
        min_length: int,
        max_active: int = 8000,
    ) -> np.ndarray:
        """JIT-compiled ORF scanner for one strand.

        Returns int32 array of shape (n_found, 4):
          col 0 : start_pos   (0-based, inclusive)
          col 1 : stop_end    (0-based, exclusive — position after stop codon)
          col 2 : start_codon integer (ATG=14, GTG=46, TTG=62)
          col 3 : frame       (0, 1, or 2)
        """
        n = len(seq_arr)
        max_results = n // max(min_length // 2, 1) + 2000
        results = np.empty((max_results, 4), dtype=np.int32)
        count = 0

        # Per-frame active start buffer (start_pos, codon_int)
        act_pos = np.empty(max_active, dtype=np.int32)
        act_cod = np.empty(max_active, dtype=np.int32)

        for frame in range(3):
            n_active = 0
            i = frame
            while i <= n - 3:
                a = seq_arr[i]
                b = seq_arr[i + 1]
                c = seq_arr[i + 2]
                if a < 4 and b < 4 and c < 4:
                    codon = a * 16 + b * 4 + c
                    # ATG=14, GTG=46, TTG=62
                    if codon == 14 or codon == 46 or codon == 62:
                        if n_active < max_active:
                            act_pos[n_active] = i
                            act_cod[n_active] = codon
                            n_active += 1
                    # TAA=48, TAG=50, TGA=56
                    elif (codon == 48 or codon == 50 or codon == 56) and n_active > 0:
                        stop_end = i + 3
                        for k in range(n_active):
                            orf_len = stop_end - act_pos[k]
                            if orf_len >= min_length and count < max_results:
                                results[count, 0] = act_pos[k]
                                results[count, 1] = stop_end
                                results[count, 2] = act_cod[k]
                                results[count, 3] = frame
                                count += 1
                        n_active = 0
                i += 3

        return results[:count]

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False
    _score_imm_numba = None
    _score_codon_bias_numba = None

from Bio.Seq import Seq

from .config import (
    FIRST_FILTER_THRESHOLD,
    KNOWN_RBS_MOTIFS,
    LENGTH_REFERENCE_BP,
    MIN_ORF_LENGTH,
    SCORE_WEIGHTS,
    SECOND_FILTER_THRESHOLD,
    START_CODON_WEIGHTS,
    START_CODONS,
    START_SELECTION_WEIGHTS,
    STOP_CODONS,
)

# Registry mapping id(model_list) -> model so that get_interpolated_probability
# can look up the correct model via the cache key without using global mutation.
# Each genome's models get separate cache partitions because their ids differ.
_IMM_MODEL_REGISTRY: Dict[int, List[Dict]] = {}

_LOG_EPSILON: float = math.log(1e-10)  # fallback for unseen k-mers
_IMM_NUCLEOTIDES: tuple = ("A", "C", "G", "T")


def _interpolate_prob(
    imm_model: List[Dict],
    codon_pos: int,
    context: str,
    nucleotide: str,
    fallback: float = 0.25,
) -> float:
    """Run IMM interpolation without caching — used only at table-build time."""
    probabilities = imm_model[codon_pos]
    for order in range(len(context), -1, -1):
        ctx = context[-order:] if order > 0 else ""
        if ctx in probabilities and nucleotide in probabilities[ctx]:
            return probabilities[ctx][nucleotide]
    return fallback


def build_flat_log_table(imm_model: List[Dict], max_order: int) -> List[Dict[str, float]]:
    """Pre-bake a {context+nucleotide → log_prob} table for each codon position.

    Called once at model-build time.  Enumerates every k-mer of length 1 to
    max_order+1 over {A,C,G,T}, resolves the IMM interpolation, and stores the
    log-probability directly.  During scoring the hot loop performs two
    dict.get() calls per nucleotide — no math.log, no max(), no lru_cache.

    Table size: ~87K keys × 3 positions × 8 bytes ≈ 2 MB per model.
    Build time: ~100 ms (one-time).
    """
    n_positions = len(imm_model)  # 3 for frame-aware, 1 for non-frame-aware
    tables: List[Dict[str, float]] = [{} for _ in range(n_positions)]

    for key_len in range(1, max_order + 2):  # key = context + nucleotide
        for chars in itertools.product(_IMM_NUCLEOTIDES, repeat=key_len):
            key = "".join(chars)
            context = key[:-1]
            nucleotide = key[-1]
            for pos in range(n_positions):
                prob = _interpolate_prob(imm_model, pos, context, nucleotide)
                tables[pos][key] = math.log(max(prob, 1e-10))

    return tables


def _score_imm_fast(
    sequence: str,
    coding_log: List[Dict[str, float]],
    noncoding_log: List[Dict[str, float]],
    max_order: int,
) -> float:
    """Fast IMM log-likelihood ratio using pre-baked log tables.

    Replaces per-nucleotide math.log() + max(EPSILON) + lru_cache lookup with
    two dict.get() calls per nucleotide.  Numerically identical to
    score_imm_ratio() for the same model.
    """
    n = len(sequence)
    if n < 3:
        return 0.0
    c_total = nc_total = 0.0
    for i in range(n):
        key = sequence[i - max_order if i >= max_order else 0 : i + 1]
        pos = i % 3
        c_total += coding_log[pos].get(key, _LOG_EPSILON)
        nc_total += noncoding_log[pos].get(key, _LOG_EPSILON)
    return (c_total - nc_total) / n


# Pre-built ASCII→int lookup table (length 128): A=0, C=1, G=2, T=3, other=4.
# A single numpy fancy-index replaces four boolean scan passes.
_ASCII_TO_INT: np.ndarray = np.full(128, 4, dtype=np.int32)
_ASCII_TO_INT[65] = 0  # A
_ASCII_TO_INT[67] = 1  # C
_ASCII_TO_INT[71] = 2  # G
_ASCII_TO_INT[84] = 3  # T

# Maps integer-encoded codon back to its string (for ORF dict construction).
# Encoding: a*16 + b*4 + c  where A=0,C=1,G=2,T=3.
# ATG=14, GTG=46, TTG=62  (only start codons needed here)
_INT_TO_CODON: Dict[int, str] = {14: "ATG", 46: "GTG", 62: "TTG"}


def _seq_to_int_fast(s: str) -> np.ndarray:
    """Vectorised DNA → int32 encoding (A=0, C=1, G=2, T=3, other=4).

    Uses a pre-built ASCII lookup table — one fancy-index op per call,
    significantly faster than four boolean-scan passes for short ORFs.
    """
    return _ASCII_TO_INT[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]


def build_numba_log_table(log_table: List[Dict[str, float]], max_order: int) -> np.ndarray:
    """Convert a flat string-keyed log table to an integer-indexed numpy array.

    Index layout for a k-mer of total length L (context L-1 chars + nucleotide):
        offset(L) = (4^L - 1) // 3   (cumulative count of shorter k-mers)
        index     = offset(L) + base4_integer_encoding(k-mer)

    Array shape: (n_positions, (4^(max_order+2) - 1) // 3)
    Typically shape (3, 87381) for max_order=7 — ~4 MB for the pair.
    """
    _NUC = {"A": 0, "C": 1, "G": 2, "T": 3}
    total = (4 ** (max_order + 2) - 1) // 3
    n_pos = len(log_table)
    tbl = np.full((n_pos, total), _LOG_EPSILON, dtype=np.float64)

    for pos in range(n_pos):
        for key, val in log_table[pos].items():
            idx, valid = 0, True
            for ch in key:
                ni = _NUC.get(ch, -1)
                if ni == -1:
                    valid = False
                    break
                idx = idx * 4 + ni
            if valid:
                offset = (4 ** len(key) - 1) // 3
                tbl[pos, offset + idx] = val

    return tbl


def build_codon_log_ratio_table(
    codon_model: Dict[str, float],
    background_codon_model: Dict[str, float],
) -> np.ndarray:
    """Build a 64-entry log-ratio table for Numba codon bias scoring.

    Index = a*16 + b*4 + c  (base-4 encoding, A=0 C=1 G=2 T=3).
    Value = log(coding_freq) - log(background_freq).
    Codons absent from either model default to 0.0 (neutral).
    """
    _NUC = {"A": 0, "C": 1, "G": 2, "T": 3}
    table = np.zeros(64, dtype=np.float64)
    all_codons = set(codon_model) | set(background_codon_model)
    for codon in all_codons:
        if len(codon) != 3 or any(c not in _NUC for c in codon):
            continue
        a = _NUC[codon[0]] * 16 + _NUC[codon[1]] * 4 + _NUC[codon[2]]
        c_freq = codon_model.get(codon, 1e-4)
        bg_freq = background_codon_model.get(codon, 1e-4)
        table[a] = math.log(c_freq) - math.log(bg_freq)
    return table


# =============================================================================
# RBS (RIBOSOME BINDING SITE) PREDICTION
# =============================================================================


def find_purine_rich_regions(
    sequence: str, min_length: int = 4, min_purine_content: float = 0.6
) -> List[Dict]:
    """Find purine-rich regions using sliding window optimization."""
    purine_regions = []
    seq_len = len(sequence)

    if seq_len < min_length:
        return purine_regions

    is_purine = [1 if base in "AG" else 0 for base in sequence]

    for start in range(seq_len):
        max_length = min(9, seq_len - start + 1)

        if max_length > min_length:
            purine_count = sum(is_purine[start : start + min_length])

            length = min_length
            if length <= seq_len - start:
                purine_fraction = purine_count / length
                if purine_fraction >= min_purine_content:
                    purine_regions.append(
                        {
                            "sequence": sequence[start : start + length],
                            "start": start,
                            "end": start + length,
                            "purine_content": purine_fraction,
                            "length": length,
                        }
                    )

            for length in range(min_length + 1, max_length):
                if start + length > seq_len:
                    break

                purine_count += is_purine[start + length - 1]

                purine_fraction = purine_count / length
                if purine_fraction >= min_purine_content:
                    purine_regions.append(
                        {
                            "sequence": sequence[start : start + length],
                            "start": start,
                            "end": start + length,
                            "purine_content": purine_fraction,
                            "length": length,
                        }
                    )

    return purine_regions


@lru_cache(maxsize=100000)
def score_motif_similarity(sequence: str) -> Tuple[float, str]:
    """Score sequence similarity to known RBS motifs."""
    best_score = 0.0
    best_motif = None
    seq_len = len(sequence)  # cache once — was called 22M times inside loop

    for motif in KNOWN_RBS_MOTIFS:
        motif_len = len(motif)
        motif_weight = motif_len / 6.0
        for offset in range(max(seq_len, motif_len)):
            matches = 0
            total_positions = 0

            for i in range(seq_len):
                motif_pos = i + offset
                if 0 <= motif_pos < motif_len:
                    total_positions += 1
                    if sequence[i] == motif[motif_pos]:
                        matches += 1

            if total_positions > 0:
                score = (matches / total_positions) * total_positions * motif_weight
                if score > best_score:
                    best_score = score
                    best_motif = motif

    return best_score, best_motif


def predict_rbs_simple(sequence: str, orf: Dict, upstream_length: int = 20) -> Dict:
    """Predict RBS using purine content, spacing, and motif similarity."""
    start_pos = orf["start"]

    if start_pos < upstream_length:
        return {
            "rbs_score": -5.0,
            "spacing_score": 0.0,
            "motif_score": 0.0,
            "best_sequence": None,
            "best_motif": None,
            "spacing": 0,
            "position": 0,
        }

    upstream_start = start_pos - upstream_length
    upstream_seq = sequence[upstream_start:start_pos]

    purine_regions = find_purine_rich_regions(upstream_seq, min_length=4, min_purine_content=0.6)

    best_score = -5.0
    best_prediction = None

    for region in purine_regions:
        sd_candidate = region["sequence"]
        spacing = len(upstream_seq) - region["end"]

        if spacing < 4 or spacing > 12:
            continue
        elif 6 <= spacing <= 8:
            spacing_score = 3.0  # Optimal
        elif 5 <= spacing <= 10:
            spacing_score = 2.5  # good
        elif 4 <= spacing <= 12:
            spacing_score = 1.5  # ok

        motif_score, best_motif = score_motif_similarity(sd_candidate)
        purine_bonus = (region["purine_content"] - 0.6) * 2.0

        combined_score = spacing_score * 2.0 + motif_score * 1.5 + purine_bonus

        if combined_score > best_score:
            best_score = combined_score
            best_prediction = {
                "rbs_score": combined_score,
                "spacing_score": spacing_score,
                "motif_score": motif_score,
                "best_sequence": sd_candidate,
                "best_motif": best_motif,
                "spacing": spacing,
                "position": region["start"],
                "purine_content": region["purine_content"],
                "length": region["length"],
            }

    return best_prediction or {
        "rbs_score": -5.0,
        "spacing_score": 0.0,
        "motif_score": 0.0,
        "best_sequence": None,
        "best_motif": None,
        "spacing": 0,
        "position": 0,
    }


# =============================================================================
# ORF DETECTION
# =============================================================================


_ORF_COLUMNS = [
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
    "rbs_motif",
    "rbs_spacing",
    "rbs_sequence",
]


def _empty_orf_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_ORF_COLUMNS)


def _build_orf_df(
    raw: np.ndarray,
    seq: str,
    seq_len: int,
    is_forward: bool,
) -> pd.DataFrame:
    """Convert _scan_orfs_numba output into an ORF DataFrame with RBS scores.

    Builds column arrays once (no per-row dict allocation) and scores RBS
    with a single Python loop over the compact int32 result array.
    """
    if len(raw) == 0:
        return _empty_orf_df()

    starts = raw[:, 0]  # 0-based start positions
    stop_ends = raw[:, 1]  # 0-based exclusive end positions
    lengths = stop_ends - starts
    strand_label = "forward" if is_forward else "reverse"

    if is_forward:
        genome_starts = starts + 1
        genome_ends = stop_ends
    else:
        genome_starts = seq_len - stop_ends + 1
        genome_ends = seq_len - starts

    codon_strs = [_INT_TO_CODON[int(c)] for c in raw[:, 2]]
    sequences = [seq[int(s) : int(e)] for s, e in zip(starts, stop_ends)]

    rbs_scores, rbs_motifs, rbs_spacings, rbs_sequences = [], [], [], []
    for i in range(len(raw)):
        rbs = predict_rbs_simple(seq, {"start": int(starts[i]) + 1}, upstream_length=20)
        rbs_scores.append(rbs["rbs_score"])
        rbs_motifs.append(rbs.get("best_motif"))
        rbs_spacings.append(rbs.get("spacing", 0))
        rbs_sequences.append(rbs.get("best_sequence"))

    return pd.DataFrame(
        {
            "start": starts + 1,
            "end": stop_ends,
            "genome_start": genome_starts,
            "genome_end": genome_ends,
            "length": lengths,
            "frame": raw[:, 3],
            "strand": strand_label,
            "start_codon": codon_strs,
            "sequence": sequences,
            "rbs_score": rbs_scores,
            "rbs_motif": rbs_motifs,
            "rbs_spacing": rbs_spacings,
            "rbs_sequence": rbs_sequences,
        }
    )


def find_orfs_candidates(sequence: str, min_length: int = 100) -> pd.DataFrame:
    """Detect all ORF candidates with dual coordinates and RBS scores.

    Returns a DataFrame (columns: start, end, genome_start, genome_end, length,
    frame, strand, start_codon, sequence, rbs_score, rbs_motif, rbs_spacing,
    rbs_sequence).  Using a DataFrame instead of List[Dict] reduces peak RAM by
    ~5–8× for large genomes (176K ORFs → ~100 MB vs ~440 MB).
    """
    if hasattr(score_motif_similarity, "cache_clear"):
        score_motif_similarity.cache_clear()

    seq_len = len(sequence)
    reverse_seq = str(Seq(sequence).reverse_complement())

    print("Detecting ORFs and calculating RBS...")

    if _NUMBA_AVAILABLE:
        fwd_raw = _scan_orfs_numba(_seq_to_int_fast(sequence), min_length)
        rev_raw = _scan_orfs_numba(_seq_to_int_fast(reverse_seq), min_length)
        parts = []
        if len(fwd_raw):
            parts.append(_build_orf_df(fwd_raw, sequence, seq_len, True))
        if len(rev_raw):
            parts.append(_build_orf_df(rev_raw, reverse_seq, seq_len, False))
        orfs_df = pd.concat(parts, ignore_index=True) if parts else _empty_orf_df()
    else:
        # Python fallback: accumulate into column lists, build DataFrame once
        cols: Dict[str, list] = {c: [] for c in _ORF_COLUMNS}
        for strand_name, seq in [("forward", sequence), ("reverse", reverse_seq)]:
            for frame in range(3):
                active_starts: list = []
                for i in range(frame, len(seq) - 2, 3):
                    codon = seq[i : i + 3]
                    if len(codon) != 3:
                        break
                    if codon in START_CODONS:
                        active_starts.append((i, codon))
                    elif codon in STOP_CODONS and active_starts:
                        for start_pos, start_codon in active_starts:
                            orf_length = i + 3 - start_pos
                            if orf_length >= min_length:
                                stop = i + 3
                                s1 = start_pos + 1
                                gs = s1 if strand_name == "forward" else seq_len - stop + 1
                                ge = stop if strand_name == "forward" else seq_len - start_pos
                                rbs = predict_rbs_simple(seq, {"start": s1}, upstream_length=20)
                                cols["start"].append(s1)
                                cols["end"].append(stop)
                                cols["genome_start"].append(gs)
                                cols["genome_end"].append(ge)
                                cols["length"].append(orf_length)
                                cols["frame"].append(frame)
                                cols["strand"].append(strand_name)
                                cols["start_codon"].append(start_codon)
                                cols["sequence"].append(seq[start_pos:stop])
                                cols["rbs_score"].append(rbs["rbs_score"])
                                cols["rbs_motif"].append(rbs.get("best_motif"))
                                cols["rbs_spacing"].append(rbs.get("spacing", 0))
                                cols["rbs_sequence"].append(rbs.get("best_sequence"))
                        active_starts = []
        orfs_df = pd.DataFrame(cols)

    print(f"Complete: {len(orfs_df):,} ORFs detected with RBS scores")
    return orfs_df


# =============================================================================
# TRAINING SET SELECTION
# =============================================================================


def select_training_glimmer(
    all_orfs: List[Dict], min_length: int = 300, max_training_size: int = 2000
) -> List[Dict]:
    """GLIMMER Pure - select long, non-overlapping ORFs.

    Overlap check is O(log k) via bisect on the sorted start-position list.
    For non-overlapping intervals stored in start order, the end positions are
    also sorted, so two bisect probes suffice to rule out any overlap.
    """
    long_orfs = [orf for orf in all_orfs if orf["length"] >= min_length]
    long_orfs.sort(key=lambda x: x["length"], reverse=True)

    training_set = []
    cov_starts: List[int] = []  # sorted start positions of covered intervals
    cov_ends: List[int] = []  # cov_ends[i] is the end of the interval at cov_starts[i]

    for orf in long_orfs:
        start = orf.get("genome_start", orf["start"])
        end = orf.get("genome_end", orf["end"])
        if start > end:
            start, end = end, start

        # Find where this interval would be inserted (by start position)
        i = bisect.bisect_left(cov_starts, start)

        # Check overlap with the predecessor (its end must be < our start)
        overlaps = (i > 0 and cov_ends[i - 1] >= start) or (
            # Check overlap with the successor (its start must be > our end)
            i < len(cov_starts)
            and cov_starts[i] <= end
        )

        if not overlaps:
            training_set.append(orf)
            cov_starts.insert(i, start)
            cov_ends.insert(i, end)
            if max_training_size is not None and len(training_set) >= max_training_size:
                break

    return training_set


def select_training_flexible(
    all_orfs: List[Dict],
    target_size: int = 500,
    min_length: int = 300,
    max_length: int = 2400,
    max_overlap_fraction: float = 0.3,
    prefer_atg: bool = True,
) -> List[Dict]:
    """Flexible training set selection with controlled overlap.

    Overlap check is O(log k) on average via bisect: we find the rightmost
    interval whose start ≤ candidate_end, then walk left only until we reach
    an interval whose end < candidate_start (no further overlap possible).
    """
    filtered = [orf for orf in all_orfs if min_length <= orf["length"] <= max_length]

    if prefer_atg:
        atg_orfs = [orf for orf in filtered if orf.get("start_codon") == "ATG"]
        non_atg_orfs = [orf for orf in filtered if orf.get("start_codon") != "ATG"]

        atg_orfs.sort(key=lambda x: x["length"], reverse=True)
        non_atg_orfs.sort(key=lambda x: x["length"], reverse=True)

        candidates = atg_orfs + non_atg_orfs
    else:
        candidates = sorted(filtered, key=lambda x: x["length"], reverse=True)

    selected = []
    # Per-strand sorted list of (start, end) for O(log k) lookup
    strand_ivs: Dict[str, List[Tuple[int, int]]] = {"forward": [], "reverse": []}

    for orf in candidates:
        orf_start = orf.get("genome_start", orf["start"])
        orf_end = orf.get("genome_end", orf["end"])
        if orf_start > orf_end:
            orf_start, orf_end = orf_end, orf_start

        orf_strand = orf.get("strand", "forward")
        orf_length = orf["length"]
        ivs = strand_ivs[orf_strand]

        # Find rightmost interval with start ≤ orf_end
        i = bisect.bisect_right(ivs, (orf_end, orf_end)) - 1
        max_overlap = 0.0
        while i >= 0:
            sel_start, sel_end = ivs[i]
            if sel_end < orf_start:
                break  # this and all earlier intervals end before us — no overlap
            overlap_bp = max(0, min(orf_end, sel_end) - max(orf_start, sel_start) + 1)
            max_overlap = max(max_overlap, overlap_bp / orf_length)
            i -= 1

        if max_overlap <= max_overlap_fraction:
            selected.append(orf)
            pos = bisect.bisect_left(ivs, (orf_start, orf_end))
            ivs.insert(pos, (orf_start, orf_end))
            if len(selected) >= target_size:
                break

    return selected


# =============================================================================
# INTERGENIC REGION EXTRACTION
# =============================================================================


def extract_intergenic_regions(
    sequence: str, training_orfs: List[Dict], buffer: int = 50, min_length: int = 150
) -> Tuple[str, List[Tuple[int, int]]]:
    """Extract intergenic regions using high-confidence genes."""
    gene_regions = []
    for orf in training_orfs:
        start = orf.get("genome_start", orf["start"])
        end = orf.get("genome_end", orf["end"])
        if start > end:
            start, end = end, start
        gene_regions.append((max(1, start - buffer), min(len(sequence), end + buffer)))

    merged = []
    for s, e in sorted(gene_regions):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    intergenic_seqs = []
    intergenic_coords = []
    last_end = 1
    for s, e in merged:
        if s - last_end >= min_length:
            intergenic_coords.append((last_end, s - 1))
            intergenic_seqs.append(sequence[last_end - 1 : s - 1])
        last_end = e + 1
    if len(sequence) - last_end + 1 >= min_length:
        intergenic_coords.append((last_end, len(sequence)))
        intergenic_seqs.append(sequence[last_end - 1 :])

    concatenated = "".join(intergenic_seqs)
    return concatenated, intergenic_coords


def extract_non_orf_regions_conservative(
    sequence: str,
    all_orfs: List[Dict],
    min_rbs_threshold: float = 3.0,
    min_length: int = 150,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Extract non-ORF regions using RBS-filtering."""
    filtered = [orf for orf in all_orfs if orf.get("rbs_score", 0) >= min_rbs_threshold]
    occupied = []
    for orf in filtered:
        start = orf.get("genome_start", orf["start"])
        end = orf.get("genome_end", orf["end"])
        if start > end:
            start, end = end, start
        occupied.append((start, end))

    merged = []
    for s, e in sorted(occupied):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    non_orf_seqs = []
    non_orf_coords = []
    last_end = 1
    for s, e in merged:
        if s - last_end >= min_length:
            non_orf_coords.append((last_end, s - 1))
            non_orf_seqs.append(sequence[last_end - 1 : s - 1])
        last_end = e + 1
    if len(sequence) - last_end + 1 >= min_length:
        non_orf_coords.append((last_end, len(sequence)))
        non_orf_seqs.append(sequence[last_end - 1 :])

    concatenated = "".join(non_orf_seqs)
    return concatenated, non_orf_coords


def extract_all_non_orf_regions(
    sequence: str, all_orfs: List[Dict], min_length: int = 150
) -> Tuple[str, List[Tuple[int, int]]]:
    """Extract all non-ORF regions (no filtering)."""
    occupied = []
    for orf in all_orfs:
        start = orf.get("genome_start", orf["start"])
        end = orf.get("genome_end", orf["end"])
        if start > end:
            start, end = end, start
        occupied.append((start, end))

    merged = []
    for s, e in sorted(occupied):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    non_orf_seqs = []
    non_orf_coords = []
    last_end = 1
    for s, e in merged:
        if s - last_end >= min_length:
            non_orf_coords.append((last_end, s - 1))
            non_orf_seqs.append(sequence[last_end - 1 : s - 1])
        last_end = e + 1
    if len(sequence) - last_end + 1 >= min_length:
        non_orf_coords.append((last_end, len(sequence)))
        non_orf_seqs.append(sequence[last_end - 1 :])

    concatenated = "".join(non_orf_seqs)
    return concatenated, non_orf_coords


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def create_training_set(
    sequence: str = None,
    all_orfs: List[Dict] = None,
    genome_id: str = None,
    cached_data: Dict = None,
    glimmer_max_size: int = 2000,
    flexible_target_size: int = 2000,
) -> List[Dict]:
    """
    Create training set by intersecting Glimmer and Flexible selections.

    Works in two modes:
    1. Live mode: Provide sequence + all_orfs
    2. Cached mode: Provide genome_id + cached_data

    Args:
        sequence: Genome sequence (for live mode)
        all_orfs: All ORFs with scores (for live mode)
        genome_id: Genome identifier (for cached mode)
        cached_data: Precomputed data (for cached mode)
        glimmer_max_size: Max training set size for Glimmer
        flexible_target_size: Target size for flexible selection

    Returns:
        List of training ORFs (intersection of both methods)
    """
    if sequence is not None and all_orfs is not None:
        pass
    elif genome_id is not None and cached_data is not None:
        genome_data = cached_data.get(genome_id)
        if genome_data is None:
            raise ValueError(f"No precomputed ORFs found for genome_id {genome_id}")
        sequence = genome_data["sequence"]
        all_orfs = genome_data["orfs"]
    else:
        raise ValueError(
            "Must provide either (sequence + all_orfs) for live mode "
            "or (genome_id + cached_data) for cached mode"
        )

    # select_training_glimmer/flexible expect List[Dict]; convert if needed.
    # Pre-filter to length >= 100 before converting to reduce dict allocation.
    if isinstance(all_orfs, pd.DataFrame):
        orfs_list = all_orfs[all_orfs["length"] >= 100].to_dict("records")
    else:
        orfs_list = all_orfs

    glimmer_set = select_training_glimmer(
        orfs_list, min_length=300, max_training_size=glimmer_max_size
    )

    flexible_set = select_training_flexible(
        orfs_list,
        target_size=flexible_target_size,
        min_length=300,
        max_length=20000,
        max_overlap_fraction=0.3,
        prefer_atg=True,
    )

    glimmer_coords = set(
        (orf.get("genome_start", orf["start"]), orf.get("genome_end", orf["end"]))
        for orf in glimmer_set
    )
    flexible_coords = set(
        (orf.get("genome_start", orf["start"]), orf.get("genome_end", orf["end"]))
        for orf in flexible_set
    )

    intersection_coords = glimmer_coords & flexible_coords

    intersection_orfs = [
        orf
        for orf in orfs_list
        if (orf.get("genome_start", orf["start"]), orf.get("genome_end", orf["end"]))
        in intersection_coords
    ]

    return intersection_orfs


def create_intergenic_set(
    sequence: str = None,
    all_orfs: List[Dict] = None,
    genome_id: str = None,
    cached_data: Dict = None,
    buffer: int = 50,
    min_length: int = 150,
    min_rbs_threshold: float = 3.0,
) -> List[Dict]:
    """
    Create intergenic regions by taking union of multiple strategies.

    Works in two modes:
    1. Live mode: Provide sequence + all_orfs
    2. Cached mode: Provide genome_id + cached_data

    Args:
        sequence: Genome sequence (for live mode)
        all_orfs: All ORFs with scores (for live mode)
        genome_id: Genome identifier (for cached mode)
        cached_data: Precomputed data (for cached mode)
        buffer: Buffer around genes (bp)
        min_length: Minimum intergenic region length
        min_rbs_threshold: Minimum RBS score for conservative extraction

    Returns:
        List of intergenic region dictionaries
    """
    if sequence is not None and all_orfs is not None:
        pass
    elif genome_id is not None and cached_data is not None:
        genome_data = cached_data.get(genome_id)
        if genome_data is None:
            raise ValueError(f"No precomputed ORFs found for genome {genome_id}")
        sequence = genome_data["sequence"]
        all_orfs = genome_data["orfs"]
    else:
        raise ValueError(
            "Must provide either (sequence + all_orfs) for live mode "
            "or (genome_id + cached_data) for cached mode"
        )

    # extract_* functions expect List[Dict]; convert if needed
    if isinstance(all_orfs, pd.DataFrame):
        all_orfs = all_orfs.to_dict("records")
    likely_genes = [orf for orf in all_orfs if orf["length"] >= 200]

    _, intergenic_coords_1 = extract_intergenic_regions(
        sequence, likely_genes, buffer=buffer, min_length=min_length
    )
    _, intergenic_coords_2 = extract_non_orf_regions_conservative(
        sequence, all_orfs, min_rbs_threshold=min_rbs_threshold, min_length=min_length
    )
    _, intergenic_coords_3 = extract_all_non_orf_regions(sequence, all_orfs, min_length=min_length)

    all_union_coords = merge_intervals(
        intergenic_coords_1 + intergenic_coords_2 + intergenic_coords_3
    )

    intergenic_regions = []
    for start, end in all_union_coords:
        seq = sequence[start - 1 : end]
        intergenic_regions.append(
            {
                "start": start,
                "end": end,
                "length": len(seq),
                "sequence": seq,
                "type": "intergenic",
            }
        )

    return intergenic_regions


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================


def create_training_set_live(
    sequence: str,
    all_orfs: List[Dict],
    glimmer_max_size: int = 2000,
    flexible_target_size: int = 2000,
) -> List[Dict]:
    """
    Convenience wrapper for live mode (new genomes).

    Use this when analyzing NCBI downloads or user FASTA files.
    """
    return create_training_set(
        sequence=sequence,
        all_orfs=all_orfs,
        glimmer_max_size=glimmer_max_size,
        flexible_target_size=flexible_target_size,
    )


def create_training_set_cached(
    genome_id: str,
    cached_data: Dict,
    glimmer_max_size: int = 2000,
    flexible_target_size: int = 2000,
) -> List[Dict]:
    """
    Convenience wrapper for cached mode (catalog genomes).

    Use this when working with pre-analyzed genomes from the catalog.
    """
    return create_training_set(
        genome_id=genome_id,
        cached_data=cached_data,
        glimmer_max_size=glimmer_max_size,
        flexible_target_size=flexible_target_size,
    )


def create_intergenic_set_live(
    sequence: str,
    all_orfs: List[Dict],
    buffer: int = 50,
    min_length: int = 150,
    min_rbs_threshold: float = 3.0,
) -> List[Dict]:
    """
    Convenience wrapper for live mode (new genomes).

    Use this when analyzing NCBI downloads or user FASTA files.
    """
    return create_intergenic_set(
        sequence=sequence,
        all_orfs=all_orfs,
        buffer=buffer,
        min_length=min_length,
        min_rbs_threshold=min_rbs_threshold,
    )


def create_intergenic_set_cached(
    genome_id: str,
    cached_data: Dict,
    buffer: int = 50,
    min_length: int = 150,
    min_rbs_threshold: float = 3.0,
) -> List[Dict]:
    """
    Convenience wrapper for cached mode (catalog genomes).

    Use this when working with pre-analyzed genomes from the catalog.
    """
    return create_intergenic_set(
        genome_id=genome_id,
        cached_data=cached_data,
        buffer=buffer,
        min_length=min_length,
        min_rbs_threshold=min_rbs_threshold,
    )


# =============================================================================
# CODON USAGE MODELS
# =============================================================================


def build_codon_model(sequences: List[Dict]) -> Dict[str, float]:
    """Build species-specific codon frequency model."""
    codon_counts = Counter()
    total_codons = 0

    for seq in sequences:
        sequence = seq["sequence"]
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i : i + 3]
            if len(codon) == 3 and "N" not in codon:
                codon_counts[codon] += 1
                total_codons += 1

    if total_codons == 0:
        return {}

    frequencies = {codon: count / total_codons for codon, count in codon_counts.items()}
    return frequencies


# =============================================================================
# INTERPOLATED MARKOV MODELS (IMM)
# =============================================================================


def build_interpolated_markov_model(
    training_sequences: List[str], max_order: int, min_observations: int = 10
) -> List[Dict]:
    """Build frame-aware IMM (3 position-specific models)."""
    # Initialize 3 models, one for each codon position
    position_models = [
        defaultdict(lambda: defaultdict(int)),  # Position 0 (1st base of codon)
        defaultdict(lambda: defaultdict(int)),  # Position 1 (2nd base of codon)
        defaultdict(lambda: defaultdict(int)),  # Position 2 (3rd base of codon)
    ]

    # Build counts for each position
    for sequence in training_sequences:
        for i in range(len(sequence)):
            nucleotide = sequence[i]
            codon_position = i % 3

            # Build context with different orders
            for order in range(min(i + 1, max_order + 1)):
                if order == 0:
                    context = ""
                else:
                    context = sequence[i - order : i]

                position_models[codon_position][context][nucleotide] += 1

    # Convert counts to probabilities
    position_probabilities = []

    for pos in range(3):
        probabilities = {}
        for context, nucleotide_counts in position_models[pos].items():
            total_count = sum(nucleotide_counts.values())

            if total_count >= min_observations:
                probabilities[context] = {}
                for nucleotide, count in nucleotide_counts.items():
                    probabilities[context][nucleotide] = count / total_count

        position_probabilities.append(probabilities)

    return position_probabilities


@lru_cache(maxsize=200000)
def get_interpolated_probability(
    nucleotide: str,
    context: str,
    codon_pos: int,
    model_id: int,
    fallback_prob: float = 0.25,
) -> float:
    probabilities = _IMM_MODEL_REGISTRY[model_id]

    for order in range(len(context), -1, -1):
        current_context = context[-order:] if order > 0 else ""

        if current_context in probabilities[codon_pos]:
            if nucleotide in probabilities[codon_pos][current_context]:
                return probabilities[codon_pos][current_context][nucleotide]

    return fallback_prob


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================


def score_orf_length(length: int) -> float:
    """Score ORF based on length."""
    return math.log(max(length, MIN_ORF_LENGTH) / LENGTH_REFERENCE_BP)


def score_start_codon(start_codon: str) -> float:
    """Score start codon preference."""
    return START_CODON_WEIGHTS.get(start_codon, 0.4)


def score_imm_ratio(
    sequence: str, coding_imm: List[Dict], noncoding_imm: List[Dict], max_order: int
) -> float:
    """Score sequence using frame-aware IMM log-likelihood ratio."""
    coding_id = id(coding_imm)
    noncoding_id = id(noncoding_imm)
    _IMM_MODEL_REGISTRY[coding_id] = coding_imm
    _IMM_MODEL_REGISTRY[noncoding_id] = noncoding_imm

    if len(sequence) < 3:
        return 0.0

    is_frame_aware = isinstance(coding_imm, list) and len(coding_imm) == 3

    coding_log_prob = 0.0
    noncoding_log_prob = 0.0
    EPSILON = 1e-10

    for i in range(len(sequence)):
        nucleotide = sequence[i]

        context_start = max(0, i - max_order)
        context = sequence[context_start:i]

        if is_frame_aware:
            codon_position = i % 3
            coding_prob = get_interpolated_probability(
                nucleotide, context, codon_position, coding_id
            )
            noncoding_prob = get_interpolated_probability(
                nucleotide, context, codon_position, noncoding_id
            )
        else:
            coding_prob = get_interpolated_probability(nucleotide, context, 0, coding_id)
            noncoding_prob = get_interpolated_probability(nucleotide, context, 0, noncoding_id)
            coding_prob = get_interpolated_probability(nucleotide, context, 0, coding_id)
            noncoding_prob = get_interpolated_probability(nucleotide, context, 0, noncoding_id)

        coding_prob = max(coding_prob, EPSILON)
        noncoding_prob = max(noncoding_prob, EPSILON)

        coding_log_prob += math.log(coding_prob)
        noncoding_log_prob += math.log(noncoding_prob)

    return (coding_log_prob - noncoding_log_prob) / len(sequence)


def score_codon_bias_ratio(
    orf_sequence: str,
    codon_model: Dict[str, float],
    background_codon_model: Dict[str, float],
) -> float:
    """Score ORF by comparing coding vs background codon usage."""
    if len(orf_sequence) < 3:
        return 0.0

    coding_score = 0.0
    background_score = 0.0
    codon_count = 0

    for i in range(0, len(orf_sequence) - 2, 3):
        codon = orf_sequence[i : i + 3]
        if len(codon) == 3 and "N" not in codon:
            # Get frequencies (with small pseudocount for unseen codons)
            coding_freq = codon_model.get(codon, 0.0001)
            background_freq = background_codon_model.get(codon, 0.0001)

            coding_score += math.log(coding_freq)
            background_score += math.log(background_freq)
            codon_count += 1

    if codon_count == 0:
        return 0.0

    return (coding_score - background_score) / codon_count


# =============================================================================
# MODEL BUILDING
# =============================================================================
def clear_imm_cache():
    """Clear the LRU cache and model registry for IMM scoring."""
    get_interpolated_probability.cache_clear()
    _IMM_MODEL_REGISTRY.clear()


def build_all_scoring_models(
    training_set: List[Dict], intergenic_set: List[Dict], min_observations: int = 10
) -> Dict:
    """Build all traditional scoring models from training data."""
    print("Building traditional scoring models...")
    start_time = time.time()

    clear_imm_cache()

    print("  Building codon usage models...")
    codon_model = build_codon_model(training_set)
    background_codon_model = build_codon_model(intergenic_set)

    print("  Building IMM models...")
    training_seqs = [orf["sequence"] for orf in training_set]
    intergenic_seqs = [orf["sequence"] for orf in intergenic_set]

    n_training = sum(len(seq) for seq in training_seqs)
    n_intergenic = sum(len(seq) for seq in intergenic_seqs)
    effective_n = min(n_training, n_intergenic)

    if effective_n < min_observations:
        estimated_order = 0
    else:
        estimated_order = math.floor(math.log2(effective_n / min_observations) / 2)

    estimated_order = min(estimated_order, 8)
    estimated_order = max(estimated_order, 3)

    coding_imm = build_interpolated_markov_model(training_seqs, estimated_order, min_observations)
    noncoding_imm = build_interpolated_markov_model(
        intergenic_seqs, estimated_order, min_observations
    )

    print("  Building IMM log tables...")
    coding_log_table = build_flat_log_table(coding_imm, estimated_order)
    noncoding_log_table = build_flat_log_table(noncoding_imm, estimated_order)

    numba_coding_table = numba_noncoding_table = None
    codon_log_ratio_table = None
    if _NUMBA_AVAILABLE:
        print("  Building Numba integer tables...")
        numba_coding_table = build_numba_log_table(coding_log_table, estimated_order)
        numba_noncoding_table = build_numba_log_table(noncoding_log_table, estimated_order)
        codon_log_ratio_table = build_codon_log_ratio_table(codon_model, background_codon_model)
        # Warm up both JIT functions so first scoring call has no compile latency
        _warmup = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int32)
        _score_imm_numba(
            _warmup,
            numba_coding_table,
            numba_noncoding_table,
            estimated_order,
            _LOG_EPSILON,
        )
        _score_codon_bias_numba(_warmup, codon_log_ratio_table)

    print(f"✓ All models built in {time.time() - start_time:.1f}s")
    print(f"  IMM order: {estimated_order}")
    print(f"  IMM backend: {'numba' if _NUMBA_AVAILABLE else 'python'}")
    print(f"  Training sequences: {len(training_seqs)} ({n_training:,} bp)")
    print(f"  Intergenic sequences: {len(intergenic_seqs)} ({n_intergenic:,} bp)")

    return {
        "codon_model": codon_model,
        "background_codon_model": background_codon_model,
        "coding_imm": coding_imm,
        "noncoding_imm": noncoding_imm,
        "max_order": estimated_order,
        "coding_log_table": coding_log_table,
        "noncoding_log_table": noncoding_log_table,
        "numba_coding_table": numba_coding_table,
        "numba_noncoding_table": numba_noncoding_table,
        "codon_log_ratio_table": codon_log_ratio_table,
    }


# =============================================================================
# SCORING PIPELINE
# =============================================================================


def normalize_scores_zscore(scores) -> np.ndarray:
    """Normalize scores using z-score (mean=0, std=1)."""
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)

    if std == 0:  # All scores identical
        return np.zeros_like(scores)

    return (scores - mean) / std


def normalize_all_orf_scores(scored_orfs) -> pd.DataFrame:
    """Z-score normalize all five score columns; adds *_score_norm columns."""
    if isinstance(scored_orfs, list):
        scored_orfs = pd.DataFrame(scored_orfs)
    print(f"\nNormalizing {len(scored_orfs):,} ORF scores...")
    scored_orfs = scored_orfs.copy()
    for col in ("codon_score", "imm_score", "rbs_score", "length_score", "start_score"):
        vals = scored_orfs[col].values.astype(np.float64)
        scored_orfs[f"{col}_norm"] = normalize_scores_zscore(vals)
    print("✓ Normalization complete")
    return scored_orfs


def add_combined_scores(scored_orfs, weights: Dict = None) -> pd.DataFrame:
    """Vectorised weighted sum of normalized score columns."""
    if isinstance(scored_orfs, list):
        scored_orfs = pd.DataFrame(scored_orfs)
    if weights is None:
        weights = SCORE_WEIGHTS
    print("\nCalculating combined scores...")
    scored_orfs = scored_orfs.copy()
    scored_orfs["combined_score"] = (
        scored_orfs["codon_score_norm"] * weights["codon"]
        + scored_orfs["imm_score_norm"] * weights["imm"]
        + scored_orfs["rbs_score_norm"] * weights["rbs"]
        + scored_orfs["length_score_norm"] * weights["length"]
        + scored_orfs["start_score_norm"] * weights["start"]
    )
    print("Combined scores added")
    return scored_orfs


def score_all_orfs(
    all_orfs: pd.DataFrame,
    models: Dict,
    normalize: bool = True,
    add_combined: bool = True,
    weights: Dict = None,
) -> pd.DataFrame:
    """Score all ORFs using pre-built models.  Operates on a DataFrame in-place
    (copy made internally) and returns an enriched DataFrame."""
    print(f"Scoring {len(all_orfs):,} ORFs with traditional methods...")
    start_time = time.time()

    max_order = models["max_order"]
    numba_coding = models.get("numba_coding_table")
    numba_noncoding = models.get("numba_noncoding_table")
    codon_ratio_tbl = models.get("codon_log_ratio_table")
    coding_log = models.get("coding_log_table")
    noncoding_log = models.get("noncoding_log_table")
    codon_model = models["codon_model"]
    bg_codon_model = models["background_codon_model"]
    use_numba = _NUMBA_AVAILABLE and numba_coding is not None and codon_ratio_tbl is not None
    if use_numba:
        assert numba_coding is not None
        assert numba_noncoding is not None
        assert codon_ratio_tbl is not None

    sequences = all_orfs["sequence"].values
    n = len(sequences)
    codon_scores = np.empty(n, dtype=np.float64)
    imm_scores = np.empty(n, dtype=np.float64)

    for i, seq in enumerate(sequences):
        if i % 25000 == 0 and i > 0:
            print(f"  {i:,}...")
        if use_numba:
            arr = _seq_to_int_fast(seq)
            codon_scores[i] = float(_score_codon_bias_numba(arr, codon_ratio_tbl))
            imm_scores[i] = float(
                _score_imm_numba(arr, numba_coding, numba_noncoding, max_order, _LOG_EPSILON)
            )
        else:
            codon_scores[i] = score_codon_bias_ratio(seq, codon_model, bg_codon_model)
            imm_scores[i] = _score_imm_fast(seq, coding_log, noncoding_log, max_order)

    all_orfs = all_orfs.copy()
    all_orfs["codon_score"] = codon_scores
    all_orfs["imm_score"] = imm_scores
    lengths = all_orfs["length"].values
    all_orfs["length_score"] = np.log(np.maximum(lengths, MIN_ORF_LENGTH) / LENGTH_REFERENCE_BP)
    all_orfs["start_score"] = all_orfs["start_codon"].map(lambda c: START_CODON_WEIGHTS.get(c, 0.4))
    if "rbs_score" not in all_orfs.columns:
        all_orfs["rbs_score"] = 0.0

    print(f"✓ Scoring complete in {(time.time() - start_time)/60:.1f} minutes")

    if normalize:
        all_orfs = normalize_all_orf_scores(all_orfs)
    if add_combined:
        all_orfs = add_combined_scores(all_orfs, weights)
    return all_orfs


#  =============================================================================
# FILTERING
# =============================================================================
def filter_candidates(
    all_orfs: pd.DataFrame,
    codon_threshold: float = 0,
    imm_threshold: float = 0,
    length_threshold: float = 0,
    combined_threshold: float = 0,
) -> pd.DataFrame:
    """Boolean-mask filter: removes ORFs where all three scores are below their
    thresholds OR combined_score is below its threshold."""
    all_three_below = (
        (all_orfs["length_score"] < length_threshold)
        & (all_orfs["codon_score"] < codon_threshold)
        & (all_orfs["imm_score"] < imm_threshold)
    )
    combined_below = all_orfs["combined_score"] < combined_threshold
    keep = ~(all_three_below | combined_below)
    result = all_orfs[keep].reset_index(drop=True)
    print(f"Filtered: {len(result):,} kept, {(~keep).sum():,} removed")
    return result


# =============================================================================
# ORF GROUPING AND START SELECTION
# =============================================================================


def organize_nested_orfs(all_orfs: pd.DataFrame) -> Dict:
    """Group ORFs by (strand, end) key. Converts to List[Dict] per group so
    downstream ml_models functions receive the expected type."""
    groups: Dict = defaultdict(list)
    for row in all_orfs.itertuples(index=False):
        groups[(row.strand, row.end)].append(row._asdict())
    for key in groups:
        groups[key].sort(key=lambda x: x["start"])
    return groups


def select_best_starts(nested_groups: Dict, weights: Dict = None) -> pd.DataFrame:
    """Select best start position for each stop codon using multi-factor scoring.
    Returns a DataFrame."""
    if weights is None:
        from .config import START_SELECTION_WEIGHTS

        weights = START_SELECTION_WEIGHTS

    print(f"\nSelecting best start for {len(nested_groups):,} groups")

    selected_orfs = []
    single_option = 0
    multiple_options = 0

    for (strand, end), orfs in nested_groups.items():
        if len(orfs) == 1:
            selected_orfs.append(orfs[0])
            single_option += 1
        else:
            for orf in orfs:
                orf["start_selection_score"] = (
                    orf["codon_score_norm"] * weights["codon"]
                    + orf["imm_score_norm"] * weights["imm"]
                    + orf["rbs_score_norm"] * weights["rbs"]
                    + orf["length_score_norm"] * weights["length"]
                    + orf["start_score_norm"] * weights["start"]
                )
            selected_orfs.append(max(orfs, key=lambda x: x["start_selection_score"]))
            multiple_options += 1

    print(f"  Single option groups: {single_option:,}")
    print(f"  Multiple option groups: {multiple_options:,}")

    return pd.DataFrame(selected_orfs) if selected_orfs else _empty_orf_df()


# =============================================================================
# COMPLETE PIPELINE ON TEST GENOMES FOR DEBUGGING
# =============================================================================


def process_genome(genome_id: str, cached_data: Dict) -> List[Dict]:
    """Process a single genome through the complete ORF prediction pipeline."""
    print(f"\n{'='*80}")
    print(f"PROCESSING GENOME: {genome_id}")
    print(f"{'='*80}")

    # Load ORFs
    genome_data = cached_data[genome_id]
    all_orfs = genome_data["orfs"]
    print(f"Total ORFs detected: {len(all_orfs):,}")

    # Create training sets
    print(f"\n{'='*80}")
    print("STEP 1: CREATE TRAINING SETS")
    print(f"{'='*80}")
    training_set = create_training_set_cached(genome_id, cached_data)
    intergenic_set = create_intergenic_set_cached(genome_id, cached_data)
    print(f"Training set: {len(training_set):,} ORFs")
    print(f"Intergenic set: {len(intergenic_set):,} regions")

    # Build models
    print(f"\n{'='*80}")
    print("STEP 2: BUILD SCORING MODELS")
    print(f"{'='*80}")
    models = build_all_scoring_models(training_set, intergenic_set)

    # Score ORFs
    print(f"\n{'='*80}")
    print("STEP 3: SCORE ALL ORFs")
    print(f"{'='*80}")
    scored_orfs = score_all_orfs(all_orfs, models)

    # Filter candidates - USE OPTIMIZED FIRST FILTER
    print(f"\n{'='*80}")
    print("STEP 4: FILTER CANDIDATES (INITIAL)")
    print(f"{'='*80}")
    candidates = filter_candidates(scored_orfs, **FIRST_FILTER_THRESHOLD)

    print(f"Candidates after initial filter: {len(candidates):,}")

    # Group and select best starts - USE OPTIMIZED WEIGHTS
    print(f"\n{'='*80}")
    print("STEP 5: SELECT BEST START CODONS")
    print(f"{'='*80}")
    grouped_orfs = organize_nested_orfs(candidates)
    top_candidates = select_best_starts(grouped_orfs, START_SELECTION_WEIGHTS)
    print(f"Top candidates after start selection: {len(top_candidates):,}")

    # Final filtering - USE OPTIMIZED SECOND FILTER
    print(f"\n{'='*80}")
    print("STEP 6: FINAL FILTERING")
    print(f"{'='*80}")
    final_predictions = filter_candidates(top_candidates, **SECOND_FILTER_THRESHOLD)

    print(f"Final predictions: {len(final_predictions):,}")

    return final_predictions
