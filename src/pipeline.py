"""
Gene prediction pipeline — single importable entry point.

All 10 pipeline steps live here. CLI, API, batch scripts, and tests all
call predict_genome() instead of re-implementing the steps inline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .config import (
    FIRST_FILTER_THRESHOLD,
    SECOND_FILTER_THRESHOLD,
    START_SELECTION_WEIGHTS,
)
from .data_management import load_genome_sequence
from .ml_models import HybridGeneFilter, OrfGroupClassifier
from .traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    organize_nested_orfs,
    score_all_orfs,
    select_best_starts,
)

__all__ = ["predict_genome", "predict_genome_from_file", "load_models", "write_gff"]

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "models"


# ── Model loader (cached) ─────────────────────────────────────────────────────

_lgb_cache: Optional[OrfGroupClassifier] = None
_hf_cache: Optional[HybridGeneFilter] = None


def load_models(
    lgb_path: Optional[str] = None,
    hf_path: Optional[str] = None,
) -> tuple[Optional[OrfGroupClassifier], Optional[HybridGeneFilter]]:
    """Load LGB and Hybrid models from disk, using module-level cache."""
    global _lgb_cache, _hf_cache

    lgb_file = Path(lgb_path) if lgb_path else _MODELS_DIR / "orf_classifier_lgb.pkl"
    hf_file = Path(hf_path) if hf_path else _MODELS_DIR / "hybrid_best_model.pkl"

    if _lgb_cache is None and lgb_file.exists():
        _lgb_cache = OrfGroupClassifier()
        _lgb_cache.load(str(lgb_file))

    if _hf_cache is None and hf_file.exists():
        _hf_cache = HybridGeneFilter()
        _hf_cache.load(str(hf_file))

    return _lgb_cache, _hf_cache


# ── Core pipeline ─────────────────────────────────────────────────────────────


def predict_genome(
    sequence: str,
    genome_id: str = "genome",
    lgb: Optional[OrfGroupClassifier] = None,
    lgb_threshold: float = 0.07,
    hf: Optional[HybridGeneFilter] = None,
    hf_threshold: Optional[float] = None,
    min_orf_length: int = 100,
) -> Union[pd.DataFrame, List[Dict]]:
    """
    Run the complete gene prediction pipeline on a DNA sequence.

    Steps:
      1. Find all ORF candidates
      2-3. Build self-training and intergenic sets
      4-5. Build IMM/codon scoring models and score all ORFs
      6. First filter (remove clearly non-coding ORFs)
      7. Group nested ORFs by shared stop codon
      8. LGB group filter (optional — removes groups unlikely to contain a gene)
      9. Select best start codon per group
      10. Second filter
      11. HybridGeneFilter final filter (optional)

    Args:
        sequence:      Raw DNA sequence (A/C/G/T).
        genome_id:     Identifier used in logging (not written to output).
        lgb:           Pre-loaded OrfGroupClassifier. Pass None to skip.
        lgb_threshold: LGB group-filter probability threshold (default: 0.07).
        hf:            Pre-loaded HybridGeneFilter. Pass None to skip.
        hf_threshold:  HybridGeneFilter threshold; uses model default if None.
        min_orf_length: Minimum ORF length in bp (default: 100).

    Returns:
        DataFrame (or list of dicts) of predicted genes with coordinates,
        strand, scores, and sequences.
    """
    # Steps 1-3: ORF candidates + self-training sets
    logger.info("[%s] Finding ORF candidates...", genome_id)
    orfs = find_orfs_candidates(sequence, min_length=min_orf_length)
    training = create_training_set(sequence=sequence, all_orfs=orfs)
    intergenic = create_intergenic_set(sequence=sequence, all_orfs=orfs)

    # Steps 4-5: Build models + score
    logger.info("[%s] Building scoring models and scoring %d ORFs...", genome_id, len(orfs))
    models = build_all_scoring_models(training, intergenic)
    scored = score_all_orfs(orfs, models)

    # Step 6: First filter
    filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
    logger.info("[%s] First filter: %d → %d candidates", genome_id, len(orfs), len(filtered))

    # Step 7: Group nested ORFs
    groups = organize_nested_orfs(filtered)

    # Step 8: LGB group filter (optional)
    if lgb is not None:
        pre = len(groups)
        groups = lgb.filter_groups(
            groups=groups,
            genome_id=genome_id,
            weights=START_SELECTION_WEIGHTS,
            threshold=lgb_threshold,
        )
        logger.info("[%s] LGB filter: %d → %d groups", genome_id, pre, len(groups))

    # Step 9: Select best start codon
    top = select_best_starts(groups, START_SELECTION_WEIGHTS)

    # Step 10: Second filter
    final = filter_candidates(top, **SECOND_FILTER_THRESHOLD)
    logger.info("[%s] Second filter: %d candidates", genome_id, len(final))

    # Step 11: HybridGeneFilter (optional)
    if hf is not None:
        t = hf_threshold if hf_threshold is not None else hf.threshold
        pre = len(final)
        final = hf.filter_candidates(
            candidates=final,
            genome_id=genome_id,
            threshold=t,
            batch_size=32,
        )
        logger.info("[%s] Hybrid filter: %d → %d predictions", genome_id, pre, len(final))

    logger.info("[%s] Done — %d genes predicted", genome_id, len(final))
    return final


def predict_genome_from_file(
    fasta_path: str,
    genome_id: Optional[str] = None,
    lgb: Optional[OrfGroupClassifier] = None,
    lgb_threshold: float = 0.07,
    hf: Optional[HybridGeneFilter] = None,
    hf_threshold: Optional[float] = None,
    min_orf_length: int = 100,
) -> Union[pd.DataFrame, List[Dict]]:
    """
    Convenience wrapper: load a FASTA file then call predict_genome().

    Args:
        fasta_path: Path to FASTA file.
        genome_id:  Identifier; defaults to the FASTA filename stem.
        (all other args): passed through to predict_genome().

    Returns:
        Same as predict_genome().
    """
    genome = load_genome_sequence(fasta_path)
    if not genome:
        raise ValueError(f"Could not load genome from {fasta_path}")
    seq = genome["sequence"]
    gid = genome_id or Path(fasta_path).stem
    return predict_genome(
        sequence=seq,
        genome_id=gid,
        lgb=lgb,
        lgb_threshold=lgb_threshold,
        hf=hf,
        hf_threshold=hf_threshold,
        min_orf_length=min_orf_length,
    )


# ── GFF output ────────────────────────────────────────────────────────────────


def write_gff(
    predictions,
    output_path: str,
    sequence_id: str = "genome",
) -> int:
    """
    Write gene predictions to GFF3 format.

    Args:
        predictions: List of dicts or DataFrame from predict_genome().
        output_path: Destination file path.
        sequence_id: Sequence identifier written in column 1 (default: "genome").

    Returns:
        Number of genes written.
    """
    rows = predictions.to_dict("records") if hasattr(predictions, "to_dict") else list(predictions)
    with open(output_path, "w") as f:
        f.write("##gff-version 3\n")
        for i, pred in enumerate(rows, 1):
            start = pred.get("genome_start", pred.get("start", 0))
            end = pred.get("genome_end", pred.get("end", 0))
            strand = "+" if pred.get("strand", "forward") == "forward" else "-"
            score = pred.get("combined_score", 0.0)
            rbs = pred.get("rbs_score", 0.0)
            attrs = f"ID=gene_{i};rbs_score={rbs:.2f};combined_score={score:.2f}"
            f.write(
                f"{sequence_id}\tHybridPredictor\tCDS\t"
                f"{start}\t{end}\t{score:.3f}\t{strand}\t0\t{attrs}\n"
            )
    return len(rows)
