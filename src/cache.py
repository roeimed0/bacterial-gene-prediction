"""
Caching Module for ORF Candidate Preprocessing

Handles caching of expensive ORF detection operations to speed up analysis.
Uses data_management for I/O and traditional_methods for ORF detection.
"""

import os
import pickle
from typing import Dict, List

from .config import CACHE_FILENAME, NCBI_EMAIL, MIN_ORF_LENGTH

from .data_management import (
    download_genome_and_reference,
    load_genome_sequence,
    get_data_dir
)

from .traditional_methods import find_orfs_candidates


def get_cache_file() -> str:
    """Get cache file path."""
    return os.path.join(get_data_dir("processed"), CACHE_FILENAME)


def load_cache() -> Dict:
    """Load existing cache or return empty dict."""
    cache_file = get_cache_file()
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Loaded {len(cached_data)} cached genomes")
        return cached_data
    else:
        print("No cache found, starting fresh")
        return {}


def save_cache(cached_data: Dict) -> None:
    """Save cache to disk."""
    cache_file = get_cache_file()
    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    print(f"Saved {len(cached_data)} genomes to {cache_file}")


def add_genome_to_cache(genome_id: str, email: str, cached_data: Dict, min_length: int = None) -> bool:
    """Add genome to cache with ORF detection (skips if already cached)."""
    if min_length is None:
        min_length = MIN_ORF_LENGTH
    
    if genome_id in cached_data:
        print(f"[SKIP] {genome_id} already in cache")
        return False
    
    print(f"\n[PROCESSING] {genome_id}...")
    
    fasta_path, gff_path = download_genome_and_reference(genome_id, email=email)
    genome_data = load_genome_sequence(fasta_path)
    genome_sequence = genome_data['sequence']
    
    print(f"  Finding ORF candidates...")
    orfs = find_orfs_candidates(genome_sequence, min_length=min_length)
    print(f"  Found {len(orfs):,} ORF candidates")
    
    cached_data[genome_id] = {
        'sequence': genome_sequence,
        'orfs': orfs
    }
    
    print(f"[DONE] Added {genome_id} to cache")
    return True


def precompute_genomes(genome_ids: List[str], email: str = NCBI_EMAIL, cached_data: Dict = None, min_length: int = None) -> Dict:
    """Precompute ORF candidates for genomes with incremental caching."""
    if cached_data is None:
        cached_data = load_cache()
    
    new_genomes = 0
    skipped_genomes = 0
    
    print(f"\nProcessing {len(genome_ids)} genomes...")
    print("="*60)
    
    for i, genome_id in enumerate(genome_ids, 1):
        print(f"\n[{i}/{len(genome_ids)}] {genome_id}")
        
        was_added = add_genome_to_cache(genome_id, email, cached_data, min_length)
        
        if was_added:
            new_genomes += 1
            save_cache(cached_data)
        else:
            skipped_genomes += 1
    
    print("\n" + "="*60)
    print(f"Precomputation complete!")
    print(f"  New genomes processed: {new_genomes}")
    print(f"  Already cached: {skipped_genomes}")
    print(f"  Total in cache: {len(cached_data)}")
    print("="*60)
    
    return cached_data


def get_cached_genome(genome_id: str, cached_data: Dict = None) -> Dict:
    """Get cached genome data (loads cache if needed)."""
    if cached_data is None:
        cached_data = load_cache()
    
    if genome_id not in cached_data:
        raise KeyError(f"Genome {genome_id} not in cache. Run precompute_genomes() first.")
    
    return cached_data[genome_id]


def clear_cache() -> None:
    """Delete cache file."""
    cache_file = get_cache_file()
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Cache deleted: {cache_file}")
    else:
        print("No cache file to delete")


def cache_stats(cached_data: Dict = None) -> None:
    """Print cache statistics."""
    if cached_data is None:
        cached_data = load_cache()
    
    if not cached_data:
        print("Cache is empty")
        return
    
    print(f"\n=== CACHE STATISTICS ===")
    print(f"Total genomes cached: {len(cached_data)}")
    
    total_orfs = sum(len(data['orfs']) for data in cached_data.values())
    total_sequence_bp = sum(len(data['sequence']) for data in cached_data.values())
    
    print(f"Total ORFs: {total_orfs:,}")
    print(f"Total sequence: {total_sequence_bp:,} bp")
    print(f"Average ORFs per genome: {total_orfs / len(cached_data):.0f}")
    
    cache_file = get_cache_file()
    if os.path.exists(cache_file):
        size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"Cache file size: {size_mb:.1f} MB")
    
    print(f"\nCached genomes:")
    for genome_id in sorted(cached_data.keys()):
        num_orfs = len(cached_data[genome_id]['orfs'])
        print(f"  {genome_id}: {num_orfs:,} ORFs")