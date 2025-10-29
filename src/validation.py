"""
Validation Module - Wrapper for Gene Prediction Validation

This module provides a clean interface for validating gene predictions
by wrapping the existing compare_results_file_to_reference() function
from data_management.

Key Functions:
- validate_predictions(): Validate predictions against reference
- print_validation_report(): Display validation metrics nicely
"""

from typing import Dict
from pathlib import Path


def validate_predictions(pred_path: str, ref_path: str = None, genome_id: str = None) -> Dict:
    """
    Validate predictions against reference annotations.
    
    This is a wrapper around the existing compare_results_file_to_reference()
    function that handles the logic of finding reference files and computing metrics.
    """
    # Import the existing validation function
    try:
        from src.comparative_analysis import compare_results_file_to_reference
    except ImportError:
        from src.comparative_analysis import compare_results_file_to_reference
    
    # If genome_id not provided, try to extract from prediction filename
    if genome_id is None:
        genome_id = Path(pred_path).stem.replace('_predictions', '')
    
    metrics = compare_results_file_to_reference(genome_id)
    
    return metrics


def print_validation_report(metrics: Dict):
    """
    Print a nicely formatted validation report.
    """
    print(f"Genome ID:             {metrics['genome_id']}")
    print(f"Prediction file:       {metrics['results_file']}")
    print(f"Reference file:        {metrics['reference_file']}")
    print()
    print(f"Reference genes:       {metrics['reference_count']:,}")
    print(f"Predicted genes:       {metrics['predicted_count']:,}")
    print()
    print(f"True Positives (TP):   {metrics['true_positives']:,}")
    print(f"False Positives (FP):  {metrics['false_positives']:,}")
    print(f"False Negatives (FN):  {metrics['false_negatives']:,}")
    print()
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 30)
    print(f"{'Sensitivity (Recall)':<20} {metrics['sensitivity']:>9.2%}")
    print(f"{'Precision':<20} {metrics['precision']:>9.2%}")
    print(f"{'F1 Score':<20} {metrics['f1_score']:>10.4f}")


def validate_from_results_directory(genome_id: str) -> Dict:
    """
    Convenience function to validate results using only genome_id.
    """
    return validate_predictions(
        pred_path=f"results/{genome_id}_predictions.gff",
        genome_id=genome_id
    )
