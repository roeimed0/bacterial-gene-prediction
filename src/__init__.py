"""
Bacterial Gene Prediction Package

A hybrid machine learning and traditional algorithm approach
to bacterial gene prediction and comparative genomics.
"""

__version__ = "0.1.1"
__author__ = "Roy Medina"

# Import from cache
from .cache import (
    cache_stats,
    get_cached_genome,
    load_cache,
    precompute_genomes,
    save_cache,
)

# Import from comparative_analysis
from .comparative_analysis import (
    analyze_and_plot_scores,
    compare_orfs_to_reference,
)

# Import from config
from .config import (
    MIN_ORF_LENGTH,
    NCBI_EMAIL,
    START_CODONS,
    STOP_CODONS,
    TEST_GENOMES,
)

# Import from data_management
from .data_management import (  # Download functions; Loading functions; Utility functions
    download_all_test_genomes,
    download_genome_and_reference,
    get_data_dir,
    get_fasta_path,
    get_gff_path,
    get_project_root,
    get_reference_orfs_from_gff,
    load_genome_sequence,
    load_reference_genes_from_gff,
    print_dna_sequence,
)

# Import from traditional_methods
from .traditional_methods import (
    find_orfs_candidates,
    process_genome,
)

# Define exports
__all__ = [
    # Data management
    "download_genome_and_reference",
    "download_all_test_genomes",
    "load_genome_sequence",
    "load_reference_genes_from_gff",
    "get_reference_orfs_from_gff",
    "print_dna_sequence",
    "get_project_root",
    "get_data_dir",
    "get_gff_path",
    "get_fasta_path",
    # Cache
    "load_cache",
    "save_cache",
    "precompute_genomes",
    "get_cached_genome",
    "cache_stats",
    # Traditional methods
    "find_orfs_candidates",
    "process_genome",
    # Comparative analysis
    "analyze_and_plot_scores",
    "compare_orfs_to_reference",
    # Config
    "TEST_GENOMES",
    "NCBI_EMAIL",
    "START_CODONS",
    "STOP_CODONS",
    "MIN_ORF_LENGTH",
]
