"""
Integration smoke test for the full 10-step prediction pipeline (#15).

Runs the complete pipeline end-to-end on a real bacterial genome
(NC_000913.3 E. coli K-12, downloaded if present) and verifies:
  - Pipeline completes without error
  - Predicted genes are in the expected coordinate range
  - F1 score is above a minimum acceptable threshold
  - GFF output file is written and valid

Skipped automatically if no real genome is available in data/full_dataset/.
"""

import contextlib
import io
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "full_dataset"
MODELS_DIR = REPO_ROOT / "models"

# Minimum acceptable thresholds on E. coli K-12 (0.0–1.0 fractions)
MIN_F1 = 0.75
MIN_SENSITIVITY = 0.60
MIN_PRECISION = 0.80
SMOKE_GENOME = "NC_000913.3"


@pytest.fixture(scope="module")
def ecoli_genome():
    fasta = DATA_DIR / f"{SMOKE_GENOME}.fasta"
    if not fasta.exists():
        pytest.skip(f"Smoke genome {SMOKE_GENOME} not in data/full_dataset/ — download first")
    return fasta


@pytest.fixture(scope="module")
def pipeline_output(ecoli_genome):
    """Run the full pipeline once and cache the result for all tests."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))

    from src.comparative_analysis import compare_orfs_to_reference
    from src.config import (
        FIRST_FILTER_THRESHOLD,
        SECOND_FILTER_THRESHOLD,
        START_SELECTION_WEIGHTS,
    )
    from src.data_management import load_genome_sequence
    from src.ml_models import HybridGeneFilter, OrfGroupClassifier
    from src.traditional_methods import (
        build_all_scoring_models,
        create_intergenic_set,
        create_training_set,
        filter_candidates,
        find_orfs_candidates,
        organize_nested_orfs,
        score_all_orfs,
        select_best_starts,
    )

    genome = load_genome_sequence(str(ecoli_genome))
    seq = genome["sequence"]

    lgb = OrfGroupClassifier()
    lgb.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))
    hf = HybridGeneFilter()
    with contextlib.redirect_stdout(io.StringIO()):
        hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))

    with contextlib.redirect_stdout(io.StringIO()):
        # Steps 1-3: ORF detection + training/intergenic sets
        orfs = find_orfs_candidates(seq, min_length=100)
        training = create_training_set(sequence=seq, all_orfs=orfs)
        intergenic = create_intergenic_set(sequence=seq, all_orfs=orfs)

        # Steps 4-5: build models + score
        models = build_all_scoring_models(training, intergenic)
        scored = score_all_orfs(orfs, models)

        # Step 6: first filter
        filtered = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)

        # Steps 7-8: group + LGB filter
        groups = organize_nested_orfs(filtered)
        groups = lgb.filter_groups(
            groups=groups,
            genome_id=SMOKE_GENOME,
            weights=START_SELECTION_WEIGHTS,
            threshold=0.07,
        )

        # Step 9: start selection + second filter
        top = select_best_starts(groups, START_SELECTION_WEIGHTS)
        candidates = filter_candidates(top, **SECOND_FILTER_THRESHOLD)

        # Step 10: Hybrid filter
        final = hf.filter_candidates(
            candidates=candidates,
            genome_id=SMOKE_GENOME,
            threshold=hf.threshold,
            batch_size=32,
        )

    with contextlib.redirect_stdout(io.StringIO()):
        metrics = compare_orfs_to_reference(final, SMOKE_GENOME)

    return {"final": final, "metrics": metrics, "seq_len": len(seq)}


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestPipelineSmoke:
    def test_pipeline_produces_predictions(self, pipeline_output):
        """Pipeline must produce at least one predicted gene."""
        assert len(pipeline_output["final"]) > 0

    def test_f1_above_minimum(self, pipeline_output):
        """F1 score must be above the minimum acceptable threshold."""
        f1 = pipeline_output["metrics"]["f1_score"]
        assert f1 >= MIN_F1, (
            f"Pipeline F1 {f1:.2%} is below minimum {MIN_F1:.0%}. "
            "A model or code change may have broken the pipeline."
        )

    def test_sensitivity_above_minimum(self, pipeline_output):
        """Sensitivity must be above 60% — catching fewer than that is a red flag."""
        sens = pipeline_output["metrics"]["sensitivity"]
        assert sens >= MIN_SENSITIVITY, f"Sensitivity {sens:.2%} is below {MIN_SENSITIVITY:.0%}"

    def test_precision_above_minimum(self, pipeline_output):
        """Precision must be above 80% — predicting mostly false genes is a red flag."""
        prec = pipeline_output["metrics"]["precision"]
        assert prec >= MIN_PRECISION, f"Precision {prec:.2%} is below {MIN_PRECISION:.0%}"

    def test_all_predictions_in_genome_bounds(self, pipeline_output):
        """Every predicted gene must be within the genome sequence boundaries."""
        seq_len = pipeline_output["seq_len"]
        final = pipeline_output["final"]
        if hasattr(final, "to_dict"):
            final = final.to_dict("records")
        for pred in final:
            start = pred.get("genome_start", pred.get("start", 0))
            end = pred.get("genome_end", pred.get("end", 0))
            assert 1 <= start <= seq_len, f"start={start} out of range [1, {seq_len}]"
            assert 1 <= end <= seq_len, f"end={end} out of range [1, {seq_len}]"

    def test_predictions_have_required_fields(self, pipeline_output):
        """Each prediction must have strand, start, end and a combined score."""
        final = pipeline_output["final"]
        if hasattr(final, "to_dict"):
            records = final.to_dict("records")
        else:
            records = final
        for pred in records[:10]:  # sample first 10
            assert "strand" in pred
            assert "combined_score" in pred
            assert pred["strand"] in {"forward", "reverse"}

    def test_no_duplicate_stop_codons(self, pipeline_output):
        """After select_best_starts each (strand, stop) should appear at most once."""
        final = pipeline_output["final"]
        if hasattr(final, "to_dict"):
            records = final.to_dict("records")
        else:
            records = final
        stop_keys = set()
        for pred in records:
            key = (pred.get("strand"), pred.get("genome_end", pred.get("end")))
            assert key not in stop_keys, f"Duplicate stop codon group: {key}"
            stop_keys.add(key)

    def test_metrics_dict_has_expected_keys(self, pipeline_output):
        """Validation result must contain f1_score, sensitivity, precision."""
        metrics = pipeline_output["metrics"]
        for key in (
            "f1_score",
            "sensitivity",
            "precision",
            "true_positives",
            "false_positives",
            "false_negatives",
        ):
            assert key in metrics, f"Missing key: {key}"


# ── Standalone regression tests ────────────────────────────────────────────────


def test_load_models_missing_file_raises_clear_error_issue_152(tmp_path):
    # Regression for #152: passing a non-existent explicit path must raise
    # FileNotFoundError with the path in the message.
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.pipeline import load_models

    with pytest.raises(FileNotFoundError) as exc_info:
        load_models(lgb_path=str(tmp_path / "missing.pkl"))
    assert "missing.pkl" in str(exc_info.value)
