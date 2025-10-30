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

import math
import time
from Bio.Seq import Seq
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
from .config import KNOWN_RBS_MOTIFS, START_CODONS, STOP_CODONS,  MIN_ORF_LENGTH, LENGTH_REFERENCE_BP
from .config import SCORE_WEIGHTS, START_CODON_WEIGHTS, START_SELECTION_WEIGHTS, FIRST_FILTER_THRESHOLD, SECOND_FILTER_THRESHOLD
from functools import lru_cache

_GLOBAL_CODING_IMM = None
_GLOBAL_NONCODING_IMM = None


from functools import lru_cache

# =============================================================================
# RBS (RIBOSOME BINDING SITE) PREDICTION
# =============================================================================

def find_purine_rich_regions(
    sequence: str, 
    min_length: int = 4, 
    min_purine_content: float = 0.6
) -> List[Dict]:
    """Find purine-rich regions using sliding window optimization."""
    purine_regions = []
    seq_len = len(sequence)
    
    if seq_len < min_length:
        return purine_regions
    
    is_purine = [1 if base in 'AG' else 0 for base in sequence]
    
    for start in range(seq_len):
        max_length = min(9, seq_len - start + 1)
        
        if max_length > min_length:
            purine_count = sum(is_purine[start:start + min_length])
            
            length = min_length
            if length <= seq_len - start:
                purine_fraction = purine_count / length
                if purine_fraction >= min_purine_content:
                    purine_regions.append({
                        'sequence': sequence[start:start + length],
                        'start': start,
                        'end': start + length,
                        'purine_content': purine_fraction,
                        'length': length
                    })
            
            for length in range(min_length + 1, max_length):
                if start + length > seq_len:
                    break
                
                purine_count += is_purine[start + length - 1]
                
                purine_fraction = purine_count / length
                if purine_fraction >= min_purine_content:
                    purine_regions.append({
                        'sequence': sequence[start:start + length],
                        'start': start,
                        'end': start + length,
                        'purine_content': purine_fraction,
                        'length': length
                    })
    
    return purine_regions


@lru_cache(maxsize=100000)
def score_motif_similarity(sequence: str) -> Tuple[float, str]:
    """Score sequence similarity to known RBS motifs."""
    best_score = 0.0
    best_motif = None
    
    for motif in KNOWN_RBS_MOTIFS:
        for offset in range(max(len(sequence), len(motif))):
            matches = 0
            total_positions = 0
            
            for i in range(len(sequence)):
                motif_pos = i + offset
                if 0 <= motif_pos < len(motif):
                    total_positions += 1
                    if sequence[i] == motif[motif_pos]:
                        matches += 1
            
            if total_positions > 0:
                similarity = matches / total_positions
                
                overlap_length = total_positions
                motif_weight = len(motif) / 6.0  
                
                score = similarity * overlap_length * motif_weight
                
                if score > best_score:
                    best_score = score
                    best_motif = motif
    
    return best_score, best_motif


def predict_rbs_simple(sequence: str, orf: Dict, upstream_length: int = 20) -> Dict:
    """Predict RBS using purine content, spacing, and motif similarity."""
    start_pos = orf['start']
    
    if start_pos < upstream_length:
        return {
            'rbs_score': -5.0,
            'spacing_score': 0.0,
            'motif_score': 0.0,
            'best_sequence': None,
            'best_motif': None,
            'spacing': 0,
            'position': 0
        }

    upstream_start = start_pos - upstream_length
    upstream_seq = sequence[upstream_start:start_pos]
    
    purine_regions = find_purine_rich_regions(upstream_seq, min_length=4, min_purine_content=0.6)
    
    best_score = -5.0
    best_prediction = None
    
    for region in purine_regions:
        sd_candidate = region['sequence']
        spacing = len(upstream_seq) - region['end']
        
        if spacing < 4 or spacing > 12:
            continue
        elif 6 <= spacing <= 8:
            spacing_score= 3.0  # Optimal
        elif 5 <= spacing <= 10:
            spacing_score= 2.5  # good
        elif 4 <= spacing <= 12:
            spacing_score= 1.5  # ok
        
        motif_score, best_motif = score_motif_similarity(sd_candidate)
        purine_bonus = (region['purine_content'] - 0.6) * 2.0
        

        combined_score = (
            spacing_score * 2.0 +    
            motif_score * 1.5 +      
            purine_bonus             
        )
        
        if combined_score > best_score:
            best_score = combined_score
            best_prediction = {
                'rbs_score': combined_score,
                'spacing_score': spacing_score,
                'motif_score': motif_score,
                'best_sequence': sd_candidate,
                'best_motif': best_motif,
                'spacing': spacing,
                'position': region['start'],
                'purine_content': region['purine_content'],
                'length': region['length']
            }
    
    return best_prediction or {
        'rbs_score': -5.0,
        'spacing_score': 0.0,
        'motif_score': 0.0,
        'best_sequence': None,
        'best_motif': None,
        'spacing': 0,
        'position': 0
    }


# =============================================================================
# ORF DETECTION
# =============================================================================

def find_orfs_candidates(sequence: str, min_length: int = 100) -> List[Dict]:
    """Detect all ORF candidates with dual coordinates and RBS scores."""
    
    if hasattr(score_motif_similarity, 'cache_clear'):
        score_motif_similarity.cache_clear()
    
    orfs = []
    
    reverse_seq = str(Seq(sequence).reverse_complement())
    
    sequences = [
        ('forward', sequence),
        ('reverse', reverse_seq)
    ]
    seq_len = len(sequence)

    print("Detecting ORFs and calculating RBS...")

    for strand_name, seq in sequences:
        for frame in range(3):
            active_starts = [] 
            
            for i in range(frame, len(seq) - 2, 3):
                codon = seq[i:i+3]
                
                if len(codon) != 3:
                    break
                
                if codon in START_CODONS:
                    active_starts.append((i, codon))
                    
                elif codon in STOP_CODONS and active_starts:
                    for start_pos, start_codon in active_starts:
                        orf_length = i + 3 - start_pos
                        
                        if orf_length >= min_length:
                            # Create ORF with dual coordinates
                            if strand_name == 'forward':
                                orf = {
                                    'start': start_pos + 1,
                                    'end': i + 3,
                                    'genome_start': start_pos + 1,
                                    'genome_end': i + 3,
                                    'length': orf_length,
                                    'frame': frame,
                                    'strand': 'forward',
                                    'start_codon': start_codon,
                                    'sequence': seq[start_pos:i+3]
                                }
                            else:  # reverse strand
                                orf = {
                                    'start': start_pos + 1,
                                    'end': i + 3,
                                    'genome_start': seq_len - (i + 3) + 1,
                                    'genome_end': seq_len - start_pos,
                                    'length': orf_length,
                                    'frame': frame,
                                    'strand': 'reverse',
                                    'start_codon': start_codon,
                                    'sequence': seq[start_pos:i+3]
                                }
                            
                            # Calculate RBS for this ORF
                            rbs_result = predict_rbs_simple(seq, orf, upstream_length=20)
                            orf['rbs_score'] = rbs_result['rbs_score']
                            orf['rbs_motif'] = rbs_result.get('best_motif')
                            orf['rbs_spacing'] = rbs_result.get('spacing', 0)
                            orf['rbs_sequence'] = rbs_result.get('best_sequence')
                            
                            orfs.append(orf)
                    
                    active_starts = []
    
    print(f"Complete: {len(orfs):,} ORFs detected with RBS scores")
    return orfs

# =============================================================================
# TRAINING SET SELECTION
# =============================================================================

def select_training_glimmer(
    all_orfs: List[Dict], 
    min_length: int = 300, 
    max_training_size: int = 2000
) -> List[Dict]:
    """GLIMMER Pure - select long, non-overlapping ORFs."""
    long_orfs = [orf for orf in all_orfs if orf['length'] >= min_length]
    long_orfs.sort(key=lambda x: x['length'], reverse=True)
    
    training_set = []
    covered_intervals = []
    
    for orf in long_orfs:
        start = orf.get('genome_start', orf['start'])
        end = orf.get('genome_end', orf['end'])
        if start > end:
            start, end = end, start
        
        overlaps = False
        for cov_start, cov_end in covered_intervals:
            if not (end < cov_start or start > cov_end):
                overlaps = True
                break
        
        if not overlaps:
            training_set.append(orf)
            covered_intervals.append((start, end))
            if max_training_size is not None and len(training_set) >= max_training_size:
                break
    
    return training_set


def select_training_flexible(
    all_orfs: List[Dict], 
    target_size: int = 500, 
    min_length: int = 300, 
    max_length: int = 2400, 
    max_overlap_fraction: float = 0.3,
    prefer_atg: bool = True
) -> List[Dict]:
    """Flexible training set selection with controlled overlap."""

    filtered = [orf for orf in all_orfs 
                if min_length <= orf['length'] <= max_length]
    
    if prefer_atg:
        atg_orfs = [orf for orf in filtered if orf.get('start_codon') == 'ATG']
        non_atg_orfs = [orf for orf in filtered if orf.get('start_codon') != 'ATG']
        
        atg_orfs.sort(key=lambda x: x['length'], reverse=True)
        non_atg_orfs.sort(key=lambda x: x['length'], reverse=True)
        
        candidates = atg_orfs + non_atg_orfs
    else:
        candidates = sorted(filtered, key=lambda x: x['length'], reverse=True)
    
    selected = []
    
    for orf in candidates:
        orf_start = orf.get('genome_start', orf['start'])
        orf_end = orf.get('genome_end', orf['end'])
        if orf_start > orf_end:
            orf_start, orf_end = orf_end, orf_start
        
        orf_strand = orf.get('strand', 'forward')
        orf_length = orf['length']
        
        max_overlap = 0.0
        for sel in selected:
            if sel.get('strand', 'forward') != orf_strand:
                continue
            
            sel_start = sel.get('genome_start', sel['start'])
            sel_end = sel.get('genome_end', sel['end'])
            if sel_start > sel_end:
                sel_start, sel_end = sel_end, sel_start
            
            overlap_bp = max(0, min(orf_end, sel_end) - max(orf_start, sel_start) + 1)
            overlap_frac = overlap_bp / orf_length
            max_overlap = max(max_overlap, overlap_frac)
        
        if max_overlap <= max_overlap_fraction:
            selected.append(orf)
            
            if len(selected) >= target_size:
                break
    
    return selected


# =============================================================================
# INTERGENIC REGION EXTRACTION
# =============================================================================

def extract_intergenic_regions(
    sequence: str, 
    training_orfs: List[Dict], 
    buffer: int = 50, 
    min_length: int = 150
) -> Tuple[str, List[Tuple[int, int]]]:
    """Extract intergenic regions using high-confidence genes."""
    gene_regions = []
    for orf in training_orfs:
        start = orf.get('genome_start', orf['start'])
        end = orf.get('genome_end', orf['end'])
        if start > end:
            start, end = end, start
        gene_regions.append((max(1, start-buffer), min(len(sequence), end+buffer)))
    
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
            intergenic_coords.append((last_end, s-1))
            intergenic_seqs.append(sequence[last_end-1:s-1])
        last_end = e + 1
    if len(sequence) - last_end + 1 >= min_length:
        intergenic_coords.append((last_end, len(sequence)))
        intergenic_seqs.append(sequence[last_end-1:])
    
    concatenated = ''.join(intergenic_seqs)
    return concatenated, intergenic_coords

def extract_non_orf_regions_conservative(
    sequence: str, 
    all_orfs: List[Dict], 
    min_rbs_threshold: float = 3.0, 
    min_length: int = 150
) -> Tuple[str, List[Tuple[int, int]]]:
    """Extract non-ORF regions using RBS-filtering."""
    filtered = [orf for orf in all_orfs if orf.get('rbs_score', 0) >= min_rbs_threshold]
    occupied = []
    for orf in filtered:
        start = orf.get('genome_start', orf['start'])
        end = orf.get('genome_end', orf['end'])
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
            non_orf_coords.append((last_end, s-1))
            non_orf_seqs.append(sequence[last_end-1:s-1])
        last_end = e + 1
    if len(sequence) - last_end + 1 >= min_length:
        non_orf_coords.append((last_end, len(sequence)))
        non_orf_seqs.append(sequence[last_end-1:])
    
    concatenated = ''.join(non_orf_seqs)
    return concatenated, non_orf_coords

def extract_all_non_orf_regions(
    sequence: str, 
    all_orfs: List[Dict], 
    min_length: int = 150
) -> Tuple[str, List[Tuple[int, int]]]:
    """Extract all non-ORF regions (no filtering)."""
    occupied = []
    for orf in all_orfs:
        start = orf.get('genome_start', orf['start'])
        end = orf.get('genome_end', orf['end'])
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
            non_orf_coords.append((last_end, s-1))
            non_orf_seqs.append(sequence[last_end-1:s-1])
        last_end = e + 1
    if len(sequence) - last_end + 1 >= min_length:
        non_orf_coords.append((last_end, len(sequence)))
        non_orf_seqs.append(sequence[last_end-1:])
    
    concatenated = ''.join(non_orf_seqs)
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
    flexible_target_size: int = 2000
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
    # Mode detection
    if sequence is not None and all_orfs is not None:
        # LIVE MODE - working with new genome
        pass
    elif genome_id is not None and cached_data is not None:
        # CACHED MODE - working with catalog genome
        genome_data = cached_data.get(genome_id)
        if genome_data is None:
            raise ValueError(f"No precomputed ORFs found for genome_id {genome_id}")
        sequence = genome_data['sequence']
        all_orfs = genome_data['orfs']
    else:
        raise ValueError(
            "Must provide either (sequence + all_orfs) for live mode "
            "or (genome_id + cached_data) for cached mode"
        )
    
    # Now we have sequence and all_orfs regardless of mode
    
    # Select using Glimmer method
    glimmer_set = select_training_glimmer(
        all_orfs, 
        min_length=300, 
        max_training_size=glimmer_max_size
    )
    
    # Select using Flexible method
    flexible_set = select_training_flexible(
        all_orfs, 
        target_size=flexible_target_size, 
        min_length=300, 
        max_length=20000,
        max_overlap_fraction=0.3, 
        prefer_atg=True
    )
    
    # Find intersection
    glimmer_coords = set((orf.get('genome_start', orf['start']),
                          orf.get('genome_end', orf['end'])) for orf in glimmer_set)
    flexible_coords = set((orf.get('genome_start', orf['start']),
                           orf.get('genome_end', orf['end'])) for orf in flexible_set)
    
    intersection_coords = glimmer_coords & flexible_coords
    
    intersection_orfs = [orf for orf in all_orfs
                         if (orf.get('genome_start', orf['start']),
                             orf.get('genome_end', orf['end'])) in intersection_coords]
    
    return intersection_orfs


def create_intergenic_set(
    sequence: str = None,
    all_orfs: List[Dict] = None,
    genome_id: str = None,
    cached_data: Dict = None,
    buffer: int = 50,
    min_length: int = 150,
    min_rbs_threshold: float = 3.0
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
    # Mode detection
    if sequence is not None and all_orfs is not None:
        # LIVE MODE - working with new genome
        pass
    elif genome_id is not None and cached_data is not None:
        # CACHED MODE - working with catalog genome
        genome_data = cached_data.get(genome_id)
        if genome_data is None:
            raise ValueError(f"No precomputed ORFs found for genome {genome_id}")
        sequence = genome_data['sequence']
        all_orfs = genome_data['orfs']
    else:
        raise ValueError(
            "Must provide either (sequence + all_orfs) for live mode "
            "or (genome_id + cached_data) for cached mode"
        )
    
    # Now we have sequence and all_orfs regardless of mode
    
    # Step 1: Identify likely coding regions
    likely_genes = [orf for orf in all_orfs if orf['length'] >= 200]
    
    # Step 2: Extract intergenic regions from multiple strategies
    _, intergenic_coords_1 = extract_intergenic_regions(
        sequence, likely_genes, buffer=buffer, min_length=min_length
    )
    _, intergenic_coords_2 = extract_non_orf_regions_conservative(
        sequence, all_orfs, min_rbs_threshold=min_rbs_threshold, min_length=min_length
    )
    _, intergenic_coords_3 = extract_all_non_orf_regions(
        sequence, all_orfs, min_length=min_length
    )
    
    # Step 3: Merge all coordinates
    all_union_coords = merge_intervals(
        intergenic_coords_1 + intergenic_coords_2 + intergenic_coords_3
    )
    
    # Step 4: Build dictionary objects
    intergenic_regions = []
    for start, end in all_union_coords:
        seq = sequence[start-1:end]
        intergenic_regions.append({
            'start': start,
            'end': end,
            'length': len(seq),
            'sequence': seq,
            'type': 'intergenic'
        })
    
    return intergenic_regions


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def create_training_set_live(
    sequence: str,
    all_orfs: List[Dict],
    glimmer_max_size: int = 2000,
    flexible_target_size: int = 2000
) -> List[Dict]:
    """
    Convenience wrapper for live mode (new genomes).
    
    Use this when analyzing NCBI downloads or user FASTA files.
    """
    return create_training_set(
        sequence=sequence,
        all_orfs=all_orfs,
        glimmer_max_size=glimmer_max_size,
        flexible_target_size=flexible_target_size
    )


def create_training_set_cached(
    genome_id: str,
    cached_data: Dict,
    glimmer_max_size: int = 2000,
    flexible_target_size: int = 2000
) -> List[Dict]:
    """
    Convenience wrapper for cached mode (catalog genomes).
    
    Use this when working with pre-analyzed genomes from the catalog.
    """
    return create_training_set(
        genome_id=genome_id,
        cached_data=cached_data,
        glimmer_max_size=glimmer_max_size,
        flexible_target_size=flexible_target_size
    )


def create_intergenic_set_live(
    sequence: str,
    all_orfs: List[Dict],
    buffer: int = 50,
    min_length: int = 150,
    min_rbs_threshold: float = 3.0
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
        min_rbs_threshold=min_rbs_threshold
    )


def create_intergenic_set_cached(
    genome_id: str,
    cached_data: Dict,
    buffer: int = 50,
    min_length: int = 150,
    min_rbs_threshold: float = 3.0
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
        min_rbs_threshold=min_rbs_threshold
    )


# =============================================================================
# CODON USAGE MODELS
# =============================================================================

def build_codon_model(sequences: List[Dict]) -> Dict[str, float]:
    """Build species-specific codon frequency model."""
    codon_counts = Counter()
    total_codons = 0

    for seq in sequences:
        sequence = seq['sequence']
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if len(codon) == 3 and 'N' not in codon:
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
    training_sequences: List[str], 
    max_order: int, 
    min_observations: int = 10
) -> List[Dict]:
    """Build frame-aware IMM (3 position-specific models)."""
    # Initialize 3 models, one for each codon position
    position_models = [
        defaultdict(lambda: defaultdict(int)),  # Position 0 (1st base of codon)
        defaultdict(lambda: defaultdict(int)),  # Position 1 (2nd base of codon)
        defaultdict(lambda: defaultdict(int))   # Position 2 (3rd base of codon)
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
                    context = sequence[i-order:i]
                
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
    imm_type: str,   
    fallback_prob: float = 0.25
) -> float:

    global _GLOBAL_CODING_IMM, _GLOBAL_NONCODING_IMM  
    
    probabilities = _GLOBAL_CODING_IMM if imm_type == 'coding' else _GLOBAL_NONCODING_IMM
    
    
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
    sequence: str, 
    coding_imm: List[Dict], 
    noncoding_imm: List[Dict], 
    max_order: int
) -> float:
    """Score sequence using frame-aware IMM log-likelihood ratio."""
    global _GLOBAL_CODING_IMM, _GLOBAL_NONCODING_IMM 
    
    _GLOBAL_CODING_IMM = coding_imm
    _GLOBAL_NONCODING_IMM = noncoding_imm
    
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
                nucleotide, context, codon_position, 'coding'
            )
            noncoding_prob = get_interpolated_probability(
                nucleotide, context, codon_position, 'noncoding'
            )
        else:
            coding_prob = get_interpolated_probability(
                nucleotide, context, 0, 'coding'
            )
            noncoding_prob = get_interpolated_probability(
                nucleotide, context, 0, 'noncoding'
            )
        
        coding_prob = max(coding_prob, EPSILON)
        noncoding_prob = max(noncoding_prob, EPSILON)
        
        coding_log_prob += math.log(coding_prob)
        noncoding_log_prob += math.log(noncoding_prob)
    
    return (coding_log_prob - noncoding_log_prob) / len(sequence)

def score_codon_bias_ratio(
    orf_sequence: str, 
    codon_model: Dict[str, float], 
    background_codon_model: Dict[str, float]
) -> float:
    """Score ORF by comparing coding vs background codon usage."""
    if len(orf_sequence) < 3:
        return 0.0
    
    coding_score = 0.0
    background_score = 0.0
    codon_count = 0
    
    for i in range(0, len(orf_sequence) - 2, 3):
        codon = orf_sequence[i:i+3]
        if len(codon) == 3 and 'N' not in codon:
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
    """Clear the LRU cache for IMM scoring."""
    get_interpolated_probability.cache_clear()

def build_all_scoring_models(
    training_set: List[Dict], 
    intergenic_set: List[Dict], 
    min_observations: int = 10
) -> Dict:
    """Build all traditional scoring models from training data."""
    print("Building traditional scoring models...")
    start_time = time.time()
    
    clear_imm_cache()
    
    print("  Building codon usage models...")
    codon_model = build_codon_model(training_set)
    background_codon_model = build_codon_model(intergenic_set)

    print("  Building IMM models...")
    training_seqs = [orf['sequence'] for orf in training_set]
    intergenic_seqs = [orf['sequence'] for orf in intergenic_set]
    
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
    noncoding_imm = build_interpolated_markov_model(intergenic_seqs, estimated_order, min_observations)
    
    print(f"✓ All models built in {time.time() - start_time:.1f}s")
    print(f"  IMM order: {estimated_order}")
    print(f"  Training sequences: {len(training_seqs)} ({n_training:,} bp)")
    print(f"  Intergenic sequences: {len(intergenic_seqs)} ({n_intergenic:,} bp)")
    
    return {
        'codon_model': codon_model,
        'background_codon_model': background_codon_model,
        'coding_imm': coding_imm,
        'noncoding_imm': noncoding_imm,
        'max_order': estimated_order
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

def normalize_all_orf_scores(scored_orfs: List[Dict]) -> List[Dict]:
    """
    Normalize all score components across all ORFs.
    Adds normalized versions: *_score_norm fields.
    """
    print(f"\nNormalizing {len(scored_orfs):,} ORF scores...")
    
    codon_scores = np.array([orf['codon_score'] for orf in scored_orfs])
    imm_scores = np.array([orf['imm_score'] for orf in scored_orfs])
    rbs_scores = np.array([orf['rbs_score'] for orf in scored_orfs])
    length_scores = np.array([orf['length_score'] for orf in scored_orfs])
    start_scores = np.array([orf['start_score'] for orf in scored_orfs])
    
    # Normalize each component
    codon_norm = normalize_scores_zscore(codon_scores)
    imm_norm = normalize_scores_zscore(imm_scores)
    rbs_norm = normalize_scores_zscore(rbs_scores)
    length_norm = normalize_scores_zscore(length_scores)
    start_norm = normalize_scores_zscore(start_scores)
    
    # Add normalized scores to ORFs
    for i, orf in enumerate(scored_orfs):
        orf['codon_score_norm'] = float(codon_norm[i])
        orf['imm_score_norm'] = float(imm_norm[i])
        orf['rbs_score_norm'] = float(rbs_norm[i])
        orf['length_score_norm'] = float(length_norm[i])
        orf['start_score_norm'] = float(start_norm[i])
    
    print("✓ Normalization complete")
    return scored_orfs

def calculate_combined_score(orf: Dict, weights: Dict = None) -> float:
    """Calculate weighted combined score from normalized components."""
    if weights is None:
        weights = SCORE_WEIGHTS
    
    combined = (
        orf['codon_score_norm'] * weights['codon'] +
        orf['imm_score_norm'] * weights['imm'] +
        orf['rbs_score_norm'] * weights['rbs'] +
        orf['length_score_norm'] * weights['length'] +
        orf['start_score_norm'] * weights['start']
    )
    
    return float(combined)

def add_combined_scores(scored_orfs: List[Dict], weights: Dict = None) -> List[Dict]:
    """Add combined score to all ORFs."""
    if weights is None:
        weights = SCORE_WEIGHTS
    
    print(f"\nCalculating combined scores...")
    
    for orf in scored_orfs:
        orf['combined_score'] = calculate_combined_score(orf, weights)
    
    print("Combined scores added")
    return scored_orfs

def score_all_orfs(
    all_orfs: List[Dict], 
    models: Dict,
    normalize: bool = True,
    add_combined: bool = True,
    weights: Dict = None
) -> List[Dict]:
    """
    Score all ORFs using pre-built traditional models.
    Optionally normalize and calculate combined scores.
    """
    print(f"Scoring {len(all_orfs):,} ORFs with traditional methods...")
    start_time = time.time()
    
    codon_model = models['codon_model']
    background_codon_model = models['background_codon_model']
    coding_imm = models['coding_imm']
    noncoding_imm = models['noncoding_imm']
    max_order = models['max_order']
    
    for i, orf in enumerate(all_orfs):
        if i % 25000 == 0 and i > 0:
            print(f"  {i:,}...")
        
        orf['codon_score'] = score_codon_bias_ratio(
            orf['sequence'], codon_model, background_codon_model
        )
        
        orf['imm_score'] = score_imm_ratio(
            orf['sequence'], coding_imm, noncoding_imm, max_order
        )
        
        if 'rbs_score' not in orf:
            orf['rbs_score'] = 0.0
        
        orf['length_score'] = score_orf_length(orf['length'])
        orf['start_score'] = score_start_codon(orf.get('start_codon', 'ATG'))
    
    print(f"✓ Scoring complete in {(time.time() - start_time)/60:.1f} minutes")
    
    # Normalize scores
    if normalize:
        all_orfs = normalize_all_orf_scores(all_orfs)
    
    # Add combined score
    if add_combined:
        all_orfs = add_combined_scores(all_orfs, weights)
    
    return all_orfs


#  =============================================================================
# FILTERING
# =============================================================================
def filter_candidates(
    all_orfs: List[Dict],
    codon_threshold: float = 0,
    imm_threshold: float = 0,
    length_threshold: float = 0,
    combined_threshold: float = 0
) -> List[Dict]:
    """
    Removes ORFs if:
    - all three scores (codon, imm, length) are below thresholds, OR combined_score is below threshold
    """
    filtered_orfs = []
    
    for orf in all_orfs:
        length_score = orf.get('length_score', 0)
        codon_score = orf.get('codon_score', 0)
        imm_score = orf.get('imm_score', 0)
        combined_score = orf.get('combined_score', 0)
        
        # Remove if ALL THREE are below thresholds OR combined is below
        all_three_below = (
            length_score < length_threshold and 
            codon_score < codon_threshold and 
            imm_score < imm_threshold
        )
        combined_below = combined_score < combined_threshold
        
        if not (all_three_below or combined_below):
            filtered_orfs.append(orf)
    
    removed = len(all_orfs) - len(filtered_orfs)
    print(f"Filtered: {len(filtered_orfs):,} kept, {removed:,} removed")
    
    return filtered_orfs

# =============================================================================
# ORF GROUPING AND START SELECTION
# =============================================================================

def organize_nested_orfs(all_orfs: List[Dict]) -> Dict:
    """Group ORFs by stop codon, sort by start position within each group."""
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for orf in all_orfs:
        key = (orf['strand'], orf['end'])
        groups[key].append(orf)
    
    for key in groups:
        groups[key].sort(key=lambda x: x['start'])
    
    return groups

def select_best_starts(nested_groups: Dict, weights: Dict = None) -> List[Dict]:
    """Select best start position for each stop codon using multi-factor scoring."""
    if weights is None:
        from .config import START_SELECTION_WEIGHTS
        weights = START_SELECTION_WEIGHTS
    
    print(f"\nSelecting best start for {len(nested_groups):,} groups")
    
    selected_orfs = []
    single_option = 0
    multiple_options = 0
    
    selection_reasons = {
        'rbs_winner': 0,
        'imm_winner': 0,
        'codon_winner': 0,
        'length_winner': 0
    }
    
    for (strand, end), orfs in nested_groups.items():
        if len(orfs) == 1:
            selected_orfs.append(orfs[0])
            single_option += 1
        else:
            # Recalculate score using start-selection weights
            for orf in orfs:
                orf['start_selection_score'] = (
                    orf['codon_score_norm'] * weights['codon'] +
                    orf['imm_score_norm'] * weights['imm'] +
                    orf['rbs_score_norm'] * weights['rbs'] +
                    orf['length_score_norm'] * weights['length'] +
                    orf['start_score_norm'] * weights['start']
                )
            
            # Select based on the new score
            best_orf = max(orfs, key=lambda x: x['start_selection_score'])
            selected_orfs.append(best_orf)
            multiple_options += 1
            
            # Track selection reasons
            components = ['rbs_score_norm', 'imm_score_norm', 'codon_score_norm', 'length_score_norm']
            component_names = ['rbs_winner', 'imm_winner', 'codon_winner', 'length_winner']
            
            best_component_value = -999
            best_component = None
            for comp, name in zip(components, component_names):
                if best_orf[comp] > best_component_value:
                    best_component_value = best_orf[comp]
                    best_component = name
            
            if best_component:
                selection_reasons[best_component] += 1
    
    print(f"  Single option groups: {single_option:,}")
    print(f"  Multiple option groups: {multiple_options:,}")
    
    return selected_orfs
    
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
    all_orfs = genome_data['orfs']
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