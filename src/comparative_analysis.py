"""
Comparative Analysis Functions for Gene Prediction Validation

This module provides tools for comparing predicted ORFs against reference annotations:
- Score distribution analysis (TP vs FP)
- Visualization of score components
- Validation metrics (sensitivity, precision, F1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set

from .data_management import get_gff_path, load_reference_genes_from_gff


# =============================================================================
# COMPLETE ANALYSIS PIPELINE
# =============================================================================

def analyze_and_plot_scores(all_orfs: List[Dict], genome_id: str) -> Dict:
    """Complete pipeline: Load reference genes and analyze/plot score distributions."""
    # Load reference genes from GFF
    gff_path = get_gff_path(genome_id)
    ref_genes = load_reference_genes_from_gff(gff_path)
    
    # Generate text statistics
    print("\n" + "="*80)
    print("GENERATING DISTRIBUTION STATISTICS")
    print("="*80)
    stats = analyze_score_distributions(all_orfs, ref_genes, genome_id)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING DISTRIBUTION PLOTS")
    print("="*80)
    plot_score_distributions(all_orfs, ref_genes, genome_id)
    
    return stats


# =============================================================================
# SCORE DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_score_distributions(all_orfs: List[Dict], reference_genes: Set, genome_id: str = None) -> Dict:
    """Analyze score distributions for True Positives vs False Positives."""
    # Classify ORFs as TP or FP
    print("Classifying ORFs as True Positives or False Positives...")
    
    tp_orfs = []
    fp_orfs = []
    
    for orf in all_orfs:
        orf_start = orf.get('genome_start', orf['start'])
        orf_end = orf.get('genome_end', orf['end'])
        
        is_match = (orf_start, orf_end) in reference_genes
        
        if is_match:
            tp_orfs.append(orf)
        else:
            fp_orfs.append(orf)
    
    print(f"  True Positives: {len(tp_orfs):,}")
    print(f"  False Positives: {len(fp_orfs):,}")
    
    # Analyze each score component
    score_components = ['codon_score', 'imm_score', 'rbs_score', 'length_score', 'start_score']
    
    print("\n" + "="*80)
    print("UNNORMALIZED SCORE DISTRIBUTIONS")
    print("="*80)
    
    stats = {}
    
    for component in score_components:
        print(f"\n{component.upper().replace('_', ' ')}:")
        print("-" * 80)
        
        tp_scores = [orf.get(component, 0) for orf in tp_orfs if component in orf]
        fp_scores = [orf.get(component, 0) for orf in fp_orfs if component in orf]
        
        if not tp_scores or not fp_scores:
            print("  Insufficient data")
            continue
        
        # Calculate statistics
        tp_mean = np.mean(tp_scores)
        tp_median = np.median(tp_scores)
        tp_std = np.std(tp_scores)
        tp_min = np.min(tp_scores)
        tp_max = np.max(tp_scores)
        
        fp_mean = np.mean(fp_scores)
        fp_median = np.median(fp_scores)
        fp_std = np.std(fp_scores)
        fp_min = np.min(fp_scores)
        fp_max = np.max(fp_scores)
        
        # Display comparison
        print(f"{'Metric':<15} {'True Positives':>15} {'False Positives':>15} {'Difference':>15}")
        print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        print(f"{'Mean':<15} {tp_mean:>15.4f} {fp_mean:>15.4f} {tp_mean-fp_mean:>15.4f}")
        print(f"{'Median':<15} {tp_median:>15.4f} {fp_median:>15.4f} {tp_median-fp_median:>15.4f}")
        print(f"{'Std Dev':<15} {tp_std:>15.4f} {fp_std:>15.4f} {tp_std-fp_std:>15.4f}")
        print(f"{'Min':<15} {tp_min:>15.4f} {fp_min:>15.4f} {tp_min-fp_min:>15.4f}")
        print(f"{'Max':<15} {tp_max:>15.4f} {fp_max:>15.4f} {tp_max-fp_max:>15.4f}")
        
        # Calculate separation
        separation = abs(tp_mean - fp_mean) / ((tp_std + fp_std) / 2)
        print(f"  Separation (effect size): {separation:.4f}", end="")
        if separation > 0.8:
            print(" (Good separation)")
        elif separation > 0.5:
            print(" (Moderate separation)")
        else:
            print(" (Poor separation)")
        
        stats[component] = {
            'tp_mean': tp_mean,
            'fp_mean': fp_mean,
            'separation': separation
        }
    
    # Analyze normalized scores
    if 'codon_score_norm' in all_orfs[0]:
        print("\n" + "="*80)
        print("NORMALIZED SCORE DISTRIBUTIONS")
        print("="*80)
        
        norm_components = ['codon_score_norm', 'imm_score_norm', 'rbs_score_norm', 
                          'length_score_norm', 'start_score_norm']
        
        for component in norm_components:
            if component not in all_orfs[0]:
                continue
                
            print(f"\n{component.upper().replace('_', ' ')}:")
            print("-" * 80)
            
            tp_scores = [orf.get(component, 0) for orf in tp_orfs if component in orf]
            fp_scores = [orf.get(component, 0) for orf in fp_orfs if component in orf]
            
            if not tp_scores or not fp_scores:
                continue
            
            tp_mean = np.mean(tp_scores)
            fp_mean = np.mean(fp_scores)
            tp_median = np.median(tp_scores)
            fp_median = np.median(fp_scores)
            
            print(f"{'Metric':<15} {'True Positives':>15} {'False Positives':>15} {'Difference':>15}")
            print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")
            print(f"{'Mean':<15} {tp_mean:>15.4f} {fp_mean:>15.4f} {tp_mean-fp_mean:>15.4f}")
            print(f"{'Median':<15} {tp_median:>15.4f} {fp_median:>15.4f} {tp_median-fp_median:>15.4f}")
    
    # Analyze combined score
    if 'combined_score' in all_orfs[0]:
        print("\n" + "="*80)
        print("COMBINED SCORE DISTRIBUTION")
        print("="*80)
        
        tp_combined = [orf['combined_score'] for orf in tp_orfs if 'combined_score' in orf]
        fp_combined = [orf['combined_score'] for orf in fp_orfs if 'combined_score' in orf]
        
        if tp_combined and fp_combined:
            tp_mean = np.mean(tp_combined)
            fp_mean = np.mean(fp_combined)
            tp_median = np.median(tp_combined)
            fp_median = np.median(fp_combined)
            tp_std = np.std(tp_combined)
            fp_std = np.std(fp_combined)
            
            print(f"{'Metric':<15} {'True Positives':>15} {'False Positives':>15} {'Difference':>15}")
            print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")
            print(f"{'Mean':<15} {tp_mean:>15.4f} {fp_mean:>15.4f} {tp_mean-fp_mean:>15.4f}")
            print(f"{'Median':<15} {tp_median:>15.4f} {fp_median:>15.4f} {tp_median-fp_median:>15.4f}")
            print(f"{'Std Dev':<15} {tp_std:>15.4f} {fp_std:>15.4f}")
            
            # Percentiles
            tp_percentiles = np.percentile(tp_combined, [10, 25, 50, 75, 90])
            fp_percentiles = np.percentile(fp_combined, [10, 25, 50, 75, 90])
            
            print(f"\nPercentiles:")
            print(f"{'Percentile':<15} {'True Positives':>15} {'False Positives':>15}")
            print(f"{'-'*15} {'-'*15} {'-'*15}")
            for i, pct in enumerate([10, 25, 50, 75, 90]):
                print(f"{f'{pct}th':<15} {tp_percentiles[i]:>15.4f} {fp_percentiles[i]:>15.4f}")
    
    return stats


# =============================================================================
# SCORE VISUALIZATION
# =============================================================================

def plot_score_distributions(all_orfs: List[Dict], reference_genes: Set, genome_id: str = None):
    """Create comprehensive visualization of score distributions for TP vs FP."""
    print("Classifying ORFs...")
    tp_orfs = []
    fp_orfs = []
    
    for orf in all_orfs:
        orf_start = orf.get('genome_start', orf['start'])
        orf_end = orf.get('genome_end', orf['end'])
        
        is_match = (orf_start, orf_end) in reference_genes
        
        if is_match:
            tp_orfs.append(orf)
        else:
            fp_orfs.append(orf)
    
    print(f"  TP: {len(tp_orfs):,}, FP: {len(fp_orfs):,}")
    
    sns.set_style("whitegrid")
    
    # Figure 1: Unnormalized Score Distributions
    score_components = ['codon_score', 'imm_score', 'rbs_score', 'length_score', 'start_score']
    
    fig1, axes1 = plt.subplots(3, 2, figsize=(15, 12))
    axes1 = axes1.flatten()
    
    for idx, component in enumerate(score_components):
        ax = axes1[idx]
        
        tp_scores = [orf.get(component, 0) for orf in tp_orfs if component in orf]
        fp_scores = [orf.get(component, 0) for orf in fp_orfs if component in orf]
        
        if not tp_scores or not fp_scores:
            continue
        
        ax.hist(fp_scores, bins=50, alpha=0.6, color='red', label=f'FP (n={len(fp_scores):,})', density=True)
        ax.hist(tp_scores, bins=50, alpha=0.6, color='green', label=f'TP (n={len(tp_scores):,})', density=True)
        
        ax.axvline(np.mean(tp_scores), color='darkgreen', linestyle='--', linewidth=2, 
                   label=f'TP mean: {np.mean(tp_scores):.2f}')
        ax.axvline(np.mean(fp_scores), color='darkred', linestyle='--', linewidth=2, 
                   label=f'FP mean: {np.mean(fp_scores):.2f}')
        
        suggested = (np.median(tp_scores) + np.median(fp_scores)) / 2
        ax.axvline(suggested, color='blue', linestyle=':', linewidth=2, 
                   label=f'Suggested: {suggested:.2f}')
        
        ax.set_xlabel(component.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{component.replace("_", " ").title()} Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig1.delaxes(axes1[5])
    
    plt.suptitle(f'Unnormalized Score Distributions - TP vs FP\n{genome_id or ""}', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    # Figure 2: Normalized Scores + Combined Score
    fig2, axes2 = plt.subplots(3, 2, figsize=(15, 12))
    axes2 = axes2.flatten()
    
    if 'codon_score_norm' in all_orfs[0]:
        norm_components = ['codon_score_norm', 'imm_score_norm', 'rbs_score_norm', 
                          'length_score_norm', 'start_score_norm']
        
        for idx, component in enumerate(norm_components):
            if component not in all_orfs[0]:
                continue
                
            ax = axes2[idx]
            
            tp_scores = [orf.get(component, 0) for orf in tp_orfs if component in orf]
            fp_scores = [orf.get(component, 0) for orf in fp_orfs if component in orf]
            
            if not tp_scores or not fp_scores:
                continue
            
            ax.hist(fp_scores, bins=50, alpha=0.6, color='red', 
                   label=f'FP (n={len(fp_scores):,})', density=True)
            ax.hist(tp_scores, bins=50, alpha=0.6, color='green', 
                   label=f'TP (n={len(tp_scores):,})', density=True)
            ax.axvline(np.mean(tp_scores), color='darkgreen', linestyle='--', linewidth=2)
            ax.axvline(np.mean(fp_scores), color='darkred', linestyle='--', linewidth=2)
            
            ax.set_xlabel(component.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.set_title(f'{component.replace("_", " ").title()} Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Plot combined score (6th subplot)
    if 'combined_score' in all_orfs[0]:
        ax = axes2[5]
        
        tp_combined = [orf['combined_score'] for orf in tp_orfs if 'combined_score' in orf]
        fp_combined = [orf['combined_score'] for orf in fp_orfs if 'combined_score' in orf]
        
        if tp_combined and fp_combined:
            ax.hist(fp_combined, bins=50, alpha=0.6, color='red', 
                   label=f'FP (n={len(fp_combined):,})', density=True)
            ax.hist(tp_combined, bins=50, alpha=0.6, color='green', 
                   label=f'TP (n={len(tp_combined):,})', density=True)
            ax.axvline(np.mean(tp_combined), color='darkgreen', linestyle='--', linewidth=2, 
                      label=f'TP mean: {np.mean(tp_combined):.2f}')
            ax.axvline(np.mean(fp_combined), color='darkred', linestyle='--', linewidth=2, 
                      label=f'FP mean: {np.mean(fp_combined):.2f}')
            
            # Calculate effect size
            tp_std = np.std(tp_combined)
            fp_std = np.std(fp_combined)
            pooled_std = np.sqrt((tp_std**2 + fp_std**2) / 2)
            effect_size = abs(np.mean(tp_combined) - np.mean(fp_combined)) / pooled_std if pooled_std > 0 else 0
            
            ax.set_xlabel('Combined Score')
            ax.set_ylabel('Density')
            ax.set_title(f'Combined Score (Effect size: {effect_size:.3f})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    else:
        fig2.delaxes(axes2[5])
    
    plt.suptitle(f'Normalized + Combined Score Distributions - TP vs FP\n{genome_id or ""}', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    plt.show()
    
    print("\nPlots generated successfully")


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def compare_orfs_to_reference(orfs: List[Dict], genome_id: str, cached_data: Dict = None) -> Dict:
    """Compare predicted ORFs to reference CDS using GFF file."""
    # Get GFF path
    gff_path = get_gff_path(genome_id)

    # Convert ORFs to DataFrame
    pred = pd.DataFrame(orfs) if not isinstance(orfs, pd.DataFrame) else orfs.copy()

    # Extract coordinates
    if "genome_start" in pred.columns and "genome_end" in pred.columns:
        pred_coords = pred[["genome_start", "genome_end"]].rename(columns={
            "genome_start": "start",
            "genome_end": "end"
        })
    elif "start" in pred.columns and "end" in pred.columns:
        pred_coords = pred[["start", "end"]].copy()
    else:
        raise ValueError("ORF data must contain 'start'/'end' or 'genome_start'/'genome_end' columns.")

    # Load reference CDS from GFF
    ref = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
    ref = ref[ref[2] == "CDS"][[3, 4]].rename(columns={3: "start", 4: "end"})
    ref = ref.drop_duplicates()

    # Find exact matches
    matches = pd.merge(pred_coords, ref, on=["start", "end"])
    true_pos = len(matches)
    false_neg = len(ref) - true_pos
    false_pos = len(pred_coords) - true_pos

    # Calculate metrics
    sensitivity = (true_pos / len(ref) * 100) if len(ref) > 0 else 0
    precision = (true_pos / len(pred_coords) * 100) if len(pred_coords) > 0 else 0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0

    # Print summary
    print("=" * 80)
    print(f"VALIDATION SUMMARY: {genome_id}")
    print("=" * 80)
    print(f"Predicted ORFs:              {len(pred_coords):,}")
    print(f"Reference CDS (proteins):    {len(ref):,}")
    print(f"True positives (exact):      {true_pos:,}")
    print(f"False negatives (missed):    {false_neg:,}")
    print(f"False positives (spurious):  {false_pos:,}")
    print()
    print(f"Sensitivity (Recall):        {sensitivity:.2f}%")
    print(f"Precision:                   {precision:.2f}%")
    print(f"F1 Score:                    {f1:.2f}")
    print("=" * 80)

    return {
        "predicted": len(pred_coords),
        "reference": len(ref),
        "true_positives": true_pos,
        "false_negatives": false_neg,
        "false_positives": false_pos,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1_score": f1
    }

def compare_results_file_to_reference(genome_id: str) -> Dict:
    """
    Compare prediction results file to reference GFF file for a given genome_id.
    
    """
    from pathlib import Path
    from Bio import Entrez
    import sys
    
    print("\n" + "="*80)
    print("VALIDATION: COMPARING PREDICTIONS TO REFERENCE")
    print("="*80)
    print(f"Genome ID: {genome_id}")
    
    # Find results file
    results_path = Path('results') / f'{genome_id}_predictions.gff'
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    print(f"Results file: {results_path}")
    
    # Load predicted genes from results file
    print("\nLoading predicted genes from results file...")
    predicted_genes = set()
    
    with open(results_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                try:
                    start = int(parts[3])
                    end = int(parts[4])
                    predicted_genes.add((start, end))
                except ValueError:
                    continue
    
    print(f"  Loaded {len(predicted_genes):,} predictions")
    
    # Load reference genes - try existing file first, then download if needed
    print(f"\nLooking for reference genes for {genome_id}...")
    
    try:
        reference_gff = get_gff_path(genome_id)
        
        if reference_gff is None or not Path(reference_gff).exists():
            print(f"  Reference GFF not found locally, attempting to download from NCBI...")
            
            # Set up download path
            full_dataset_dir = Path('data') / 'full_dataset'
            full_dataset_dir.mkdir(parents=True, exist_ok=True)
            reference_gff = str(full_dataset_dir / f'{genome_id}.gff')
            
            # Try to download from NCBI
            try:
                # Get email from config if available
                try:
                    from src.config import NCBI_EMAIL
                    Entrez.email = NCBI_EMAIL
                except:
                    Entrez.email = "user@example.com"  # Fallback
                
                print(f"  Downloading GFF from NCBI...")
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=genome_id,
                    rettype="gff3",
                    retmode="text"
                )
                
                gff_content = handle.read()
                handle.close()
                
                # Check if we got actual GFF content (not an error message)
                if len(gff_content) > 100 and '##gff-version' in gff_content:
                    with open(reference_gff, 'w') as f:
                        f.write(gff_content)
                    print(f"  ✓ Downloaded reference GFF successfully")
                else:
                    raise ValueError("No valid GFF annotation available from NCBI")
                    
            except Exception as e:
                print(f"  ✗ Could not download GFF from NCBI: {e}")
                raise FileNotFoundError(
                    f"Reference GFF not found locally and could not be downloaded from NCBI.\n"
                    f"  Genome: {genome_id}\n"
                    f"  Expected location: data/full_dataset/{genome_id}.gff"
                )
        else:
            print(f"  Found reference GFF: {reference_gff}")
            
    except Exception as e:
        print(f"\n[!] Error loading reference: {e}", file=sys.stderr)
        raise
    
    reference_genes = load_reference_genes_from_gff(reference_gff)
    print(f"  Loaded {len(reference_genes):,} reference genes")
    
    # Calculate metrics (same as compare_orfs_to_reference)
    print("\nCalculating validation metrics...")
    
    true_positives = predicted_genes & reference_genes
    false_positives = predicted_genes - reference_genes
    false_negatives = reference_genes - predicted_genes
    
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    sensitivity = tp_count / len(reference_genes) if reference_genes else 0
    precision = tp_count / len(predicted_genes) if predicted_genes else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Display results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"\nReference genes:       {len(reference_genes):,}")
    print(f"Predicted genes:       {len(predicted_genes):,}")
    print(f"\nTrue Positives (TP):   {tp_count:,}")
    print(f"False Positives (FP):  {fp_count:,}")
    print(f"False Negatives (FN):  {fn_count:,}")
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 30)
    print(f"{'Sensitivity (Recall)':<20} {sensitivity:>9.2%}")
    print(f"{'Precision':<20} {precision:>9.2%}")
    print(f"{'F1 Score':<20} {f1_score:>10.4f}")
    print("="*80 + "\n")
    
    return {
        'genome_id': genome_id,
        'results_file': str(results_path),
        'reference_file': reference_gff,
        'reference_count': len(reference_genes),
        'predicted_count': len(predicted_genes),
        'true_positives': tp_count,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'sensitivity': sensitivity,
        'recall': sensitivity,
        'precision': precision,
        'f1_score': f1_score
    }

# =============================================================================
# KNOWN BACTERIAL GENOME ANALYSIS
# =============================================================================
def analyze_non_cds_genes(gff_path):
    """
    Analyze genes in the GFF that are not protein-coding (no CDS).
    Shows the difference between 'gene' and 'CDS' features.
    
    Args:
        gff_path (str): Path to GFF file
    """
    import pandas as pd
    
    print("="*100)
    print("NON-CDS GENE ANALYSIS")
    print("="*100)
    
    # Read GFF file
    gff_df = pd.read_csv(gff_path, sep="\t", comment="#", header=None,
                         names=['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'])
    
    all_genes = gff_df[gff_df['type'] == 'gene'].copy()

    all_cds = gff_df[gff_df['type'] == 'CDS'].copy()

    def extract_biotype(attr_string):
        """Extract biotype from GFF attributes."""
        if pd.isna(attr_string):
            return 'unknown'
        
        # Look for gene_biotype or gbkey
        for pair in attr_string.split(';'):
            if 'gene_biotype=' in pair:
                return pair.split('=')[1]
            elif 'gbkey=' in pair:
                return pair.split('=')[1]
        return 'unknown'
    
    all_genes['biotype'] = all_genes['attributes'].apply(extract_biotype)
    biotype_counts = all_genes['biotype'].value_counts()

    protein_coding = len(all_cds)
    non_coding = len(all_genes) - protein_coding
    
    print(f"\nTotal genes in annotation: {len(all_genes):,}")
    print(f"  • Protein-coding genes (with CDS): {protein_coding:,} ({protein_coding/len(all_genes)*100:.1f}%)")
    print(f"  • Non-coding genes (RNA, etc.): {non_coding:,} ({non_coding/len(all_genes)*100:.1f}%)")
    
    print(f"\n INTERPRETATION:")
    print(f"   The {non_coding:,} genes without CDS are:")
    print(f"   - Transfer RNA (tRNA)")
    print(f"   - Ribosomal RNA (rRNA)")
    print(f"   - Non-coding RNA (ncRNA)")
    print(f"   - Pseudogenes (degraded genes that don't produce proteins)")
    print(f"   - Other regulatory RNA elements")
    print()

    print("="*100)
    
    return {
        'total_genes': len(all_genes),
        'protein_coding': protein_coding,
        'non_coding': non_coding,
        'biotype_counts': biotype_counts.to_dict()
    }

def compare_codon_usage(
    genome_id: str,
    codon_model: Dict,
    background_codon_model: Dict,
    title: str = None,
    figsize: tuple = (14, 6)
) -> None:
    """
    Compare and visualize codon usage between coding and background regions.
    
    Creates a side-by-side bar chart showing codon frequency differences between
    training set (coding) and intergenic regions (background).
    
    Parameters:
    -----------
    genome_id : str
        Genome identifier for plot title
    codon_model : Dict
        Codon frequencies from training set (coding regions)
        Format: {codon: frequency}
    background_codon_model : Dict
        Codon frequencies from intergenic regions
        Format: {codon: frequency}
    title : str, optional
        Custom plot title. If None, uses default title with genome_id
    figsize : tuple, default=(14, 6)
        Figure size (width, height) in inches
        
    Returns:
    --------
    None
        Displays the plot
        
    Example:
    --------
    >>> from src.traditional_methods import build_codon_freq_model
    >>> 
    >>> # Build models from training and intergenic sets
    >>> codon_model = build_codon_freq_model(training_set)
    >>> background_model = build_codon_freq_model(intergenic_set)
    >>> 
    >>> # Compare and visualize
    >>> compare_codon_usage('NC_000913.3', codon_model, background_model)
    """
    import matplotlib.pyplot as plt
    
    # Get all unique codons from both models
    all_codons = sorted(set(codon_model.keys()) | set(background_codon_model.keys()))
    
    # Prepare frequencies (fill missing codons with 0)
    coding_freqs = [codon_model.get(c, 0) for c in all_codons]
    background_freqs = [background_codon_model.get(c, 0) for c in all_codons]
    differences = [c - b for c, b in zip(coding_freqs, background_freqs)]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Bar chart: coding vs background
    x = range(len(all_codons))
    plt.bar(x, coding_freqs, width=0.4, label='Coding', color='steelblue', alpha=0.8)
    plt.bar([i + 0.4 for i in x], background_freqs, width=0.4, label='Background', color='orange', alpha=0.8)
    
    # Formatting
    plt.xticks([i + 0.2 for i in x], all_codons, rotation=90, fontsize=8)
    plt.ylabel("Codon Frequency", fontsize=12)
    
    if title is None:
        title = f"Codon Usage: Coding vs Background for {genome_id}"
    plt.title(title, fontsize=14, pad=20)
    
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CODON USAGE SUMMARY")
    print("="*80)
    print(f"Total codons analyzed: {len(all_codons)}")
    print(f"Average coding frequency: {sum(coding_freqs)/len(coding_freqs):.6f}")
    print(f"Average background frequency: {sum(background_freqs)/len(background_freqs):.6f}")
    
    # Find codons with biggest differences
    codon_diffs = list(zip(all_codons, differences))
    codon_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop 10 most enriched in coding regions:")
    print("-"*80)
    for codon, diff in codon_diffs[:10]:
        if diff > 0:
            print(f"  {codon}  +{diff:.6f}  (Coding: {codon_model.get(codon, 0):.6f}, "
                  f"Background: {background_codon_model.get(codon, 0):.6f})")
    
    print(f"\nTop 10 most depleted in coding regions:")
    print("-"*80)
    for codon, diff in reversed(codon_diffs[-10:]):
        if diff < 0:
            print(f"  {codon}  {diff:.6f}  (Coding: {codon_model.get(codon, 0):.6f}, "
                  f"Background: {background_codon_model.get(codon, 0):.6f})")
    
    print("="*80 + "\n")
