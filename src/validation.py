"""
Validation Module - Wrapper for Gene Prediction Validation

This module provides a clean interface for validating gene predictions
by wrapping the existing compare_results_file_to_reference() function
from data_management.

Key Functions:
- validate_predictions(): Validate predictions against reference
- print_validation_report(): Display validation metrics nicely
- validate_batch(): Run validation across multiple genomes with per-group breakdown
"""

from pathlib import Path
from typing import Dict, List, Optional

from src.comparative_analysis import compare_results_file_to_reference


def validate_predictions(pred_path: str, ref_path: str = None, genome_id: str = None) -> Dict:
    """
    Validate predictions against reference annotations.

    This is a wrapper around the existing compare_results_file_to_reference()
    function that handles the logic of finding reference files and computing metrics.
    """
    # If genome_id not provided, try to extract from prediction filename
    if genome_id is None:
        genome_id = Path(pred_path).stem.replace("_predictions", "")

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
        pred_path=f"results/{genome_id}_predictions.gff", genome_id=genome_id
    )


def validate_batch(
    accessions: List[str],
    group_map: Optional[Dict[str, str]] = None,
    results_dir: str = "results",
    start_tolerance: int = 0,
    stop_tolerance: int = 0,
) -> Dict:
    """Run validation on multiple genomes and report per-group sensitivity/precision/F1.

    Args:
        accessions: List of genome accession strings to validate.
        group_map: Optional dict mapping accession → taxonomic group name.
            If None, looks up each accession in GENOME_CATALOG.
        results_dir: Directory containing *_predictions.gff files.
        start_tolerance: Passed to compare_orfs_to_reference (default 0 = exact).
        stop_tolerance: Passed to compare_orfs_to_reference (default 0 = exact).

    Returns:
        Dict with keys:
            "per_genome": {accession: metrics_dict}
            "per_group":  {group_name: {"n": int, "sensitivity": float,
                                         "precision": float, "f1_score": float}}
            "overall":    {"n": int, "sensitivity": float, "precision": float, "f1_score": float}
    """
    import statistics

    from src.config import GENOME_CATALOG

    # Build accession→group mapping from catalog if not supplied
    if group_map is None:
        group_map = {g["accession"]: g["group"] for g in GENOME_CATALOG}

    per_genome: Dict[str, Dict] = {}
    per_group: Dict[str, List[Dict]] = {}

    SEP = "=" * 72
    print(SEP)
    print(f"BATCH VALIDATION  ({len(accessions)} genomes)")
    if start_tolerance or stop_tolerance:
        print(f"  Tolerances: start±{start_tolerance} bp, stop±{stop_tolerance} bp")
    print(SEP)
    print(f"{'Accession':<16} {'Group':<18} {'#Pred':>7} {'Sens%':>7} {'Prec%':>7} {'F1%':>7}")
    print("-" * 72)

    for acc in accessions:
        try:
            metrics = compare_results_file_to_reference(acc)
        except Exception as e:
            print(f"  {acc:<16} SKIP ({e})")
            continue

        group = group_map.get(acc, "Unknown")
        per_genome[acc] = {**metrics, "group": group}
        per_group.setdefault(group, []).append(metrics)

        print(
            f"  {acc:<16} {group:<18} {metrics['predicted_count']:>7,}"
            f" {metrics['sensitivity'] * 100:>6.1f}%"
            f" {metrics['precision'] * 100:>6.1f}%"
            f" {metrics['f1_score'] * 100:>6.1f}%"
        )

    # Per-group averages
    print()
    print(f"{'Group':<20} {'N':>4} {'Sens%':>8} {'Prec%':>8} {'F1%':>8}")
    print("-" * 52)

    group_summaries: Dict[str, Dict] = {}
    all_sens, all_prec, all_f1 = [], [], []

    for group_name in sorted(per_group.keys()):
        rows = per_group[group_name]
        sens_vals = [r["sensitivity"] * 100 for r in rows]
        prec_vals = [r["precision"] * 100 for r in rows]
        f1_vals = [r["f1_score"] * 100 for r in rows]
        mean_s = statistics.mean(sens_vals)
        mean_p = statistics.mean(prec_vals)
        mean_f = statistics.mean(f1_vals)
        group_summaries[group_name] = {
            "n": len(rows),
            "sensitivity": mean_s,
            "precision": mean_p,
            "f1_score": mean_f,
        }
        all_sens.extend(sens_vals)
        all_prec.extend(prec_vals)
        all_f1.extend(f1_vals)
        print(
            f"  {group_name:<20} {len(rows):>4}" f" {mean_s:>7.1f}% {mean_p:>7.1f}% {mean_f:>7.1f}%"
        )

    if all_sens:
        overall = {
            "n": len(all_sens),
            "sensitivity": statistics.mean(all_sens),
            "precision": statistics.mean(all_prec),
            "f1_score": statistics.mean(all_f1),
        }
        print("-" * 52)
        print(
            f"  {'OVERALL':<20} {overall['n']:>4}"
            f" {overall['sensitivity']:>7.1f}%"
            f" {overall['precision']:>7.1f}%"
            f" {overall['f1_score']:>7.1f}%"
        )

        # Flag groups >5 pp below overall F1
        threshold = overall["f1_score"] - 5.0
        flagged = [g for g, s in group_summaries.items() if s["f1_score"] < threshold]
        if flagged:
            print(f"\n  ⚠  Groups >5 pp below overall F1: {', '.join(flagged)}")

        # Warn when per-group n is too small for reliable statistics.
        # With typical within-group std of ~10 pp across genomes:
        #   n=2  → SE ≈ 7 pp  (the mean is essentially meaningless)
        #   n=10 → SE ≈ 3 pp  (minimum for exploratory claims)
        #   n=20 → SE ≈ 2 pp  (needed for confident taxon-level conclusions)
        # Threshold: n < 10 triggers a warning; n < 20 flags as unconfirmed.
        thin = [(g, s["n"]) for g, s in group_summaries.items() if s["n"] < 10]
        if thin:
            label = ", ".join(f"{g} (n={n})" for g, n in thin)
            print(f"\n  ⚠  Low-n groups (n < 10 — exploratory only): {label}")
            print(
                "     With n < 10 and typical within-group std ~10 pp, the standard\n"
                "     error of the group mean is ≥ 3 pp — too large for reliable\n"
                "     taxon-level conclusions.  Aim for n ≥ 20 per group (see issue #101)."
            )
    else:
        overall = {"n": 0, "sensitivity": 0.0, "precision": 0.0, "f1_score": 0.0}

    return {
        "per_genome": per_genome,
        "per_group": group_summaries,
        "overall": overall,
    }
