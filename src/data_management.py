"""
Data Management Module for Bacterial Genome Analysis

This module handles the complete data lifecycle for bacterial genome data:

1. **Collection**: Download genomes and annotations from NCBI
2. **Storage**: Save to data/full_dataset with automatic caching
3. **Loading**: Read genome sequences and reference data from disk
4. **Parsing**: Extract genes, ORFs, and sequences from files
5. **Cleanup**: Remove downloaded genomes and generated results

Key Functions:
- download_genome_and_reference(): Download FASTA + GFF from NCBI
- download_all_test_genomes(): Batch download all test genomes
- load_genome_sequence(): Load genome sequence from FASTA
- load_reference_genes_from_gff(): Load reference gene positions
- get_reference_orfs_from_gff(): Extract ORF details from annotations
- cleanup_generated_files(): Clean up downloads and results
- get_project_root(): Get project root directory (portable)
- get_data_dir(): Get data subdirectory path (portable)

All downloads are cached - re-running will skip existing files.
"""


import os
from pathlib import Path
from Bio import Entrez, SeqIO
import pandas as pd
from typing import Tuple, Set, List, Dict

from .config import NCBI_EMAIL, TEST_GENOMES


def get_project_root() -> Path:
    """
    Get project root directory (works from anywhere).
    Returns the directory containing 'src' and 'data' folders.
    
    Returns:
        Path: Project root directory
    """
    # Start from this file's location (src/data_management.py)
    current = Path(__file__).resolve().parent  # src/
    project_root = current.parent              # project root/
    return project_root


def get_data_dir(subdir: str = "full_dataset") -> str:
    """
    Get data directory path (works from anywhere, any platform).
    
    Args:
        subdir: Subdirectory within data/ ('full_dataset', 'processed', etc.)
    
    Returns:
        str: Absolute path to data subdirectory
    """
    project_root = get_project_root()
    data_dir = project_root / "data" / subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def download_genome_and_reference(
    genome_id: str,
    output_dir: str = None,
    email: str = NCBI_EMAIL
) -> Tuple[str, str]:
    """Download genome FASTA and reference GFF from NCBI with auto-caching."""
    
    # Use default data directory if not specified
    if output_dir is None:
        output_dir = get_data_dir("full_dataset")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    Entrez.email = email
    
    # Define file paths
    fasta_path = os.path.join(output_dir, f"{genome_id}.fasta")
    gff_path = os.path.join(output_dir, f"{genome_id}.gff")
    
    # Download FASTA (genome sequence)
    if os.path.exists(fasta_path):
        print(f"[SKIP] Genome file {genome_id}.fasta already exists")
    else:
        print(f"Downloading genome {genome_id}...")
        try:
            handle = Entrez.efetch(
                db="nuccore", 
                id=genome_id, 
                rettype="fasta", 
                retmode="text"
            )
            genome_record = SeqIO.read(handle, "fasta")
            handle.close()
            
            SeqIO.write(genome_record, fasta_path, "fasta")
            print(f"[DONE] Saved to {fasta_path}")
            print(f"  {genome_record.description}")
            print(f"  Length: {len(genome_record.seq):,} bp")
        except Exception as e:
            print(f"[ERROR] Error downloading genome {genome_id}: {e}")
            return None, None
    
    # Download GFF (reference annotations)
    if os.path.exists(gff_path):
        print(f"[SKIP] Reference file {genome_id}.gff already exists")
    else:
        print(f"Downloading reference GFF...")
        try:
            handle = Entrez.efetch(
                db="nuccore", 
                id=genome_id, 
                rettype="gff3", 
                retmode="text"
            )
            with open(gff_path, "w") as f:
                f.write(handle.read())
            handle.close()
            print(f"[DONE] Saved to {gff_path}")
        except Exception as e:
            print(f"[ERROR] Error downloading GFF {genome_id}: {e}")
            return fasta_path, None
    
    return fasta_path, gff_path


def load_genome_sequence(fasta_path: str) -> Dict[str, any]:
    """Load genome sequence and metadata from FASTA file."""
    try:
        record = SeqIO.read(fasta_path, "fasta")
        
        genome_info = {
            'accession': record.id,
            'description': record.description,
            'length': len(record.seq),
            'sequence': str(record.seq).upper()
        }
        
        print(f"Loaded: {genome_info['description']}")
        print(f"Genome size: {genome_info['length']:,} bp")
        
        return genome_info
        
    except Exception as e:
        print(f"[ERROR] Error loading genome from {fasta_path}: {e}")
        return None


def load_reference_genes_from_gff(gff_path: str) -> Set[Tuple[int, int]]:
    """Load reference gene positions from GFF as (start, end) tuples."""
    print(f"Loading reference genes from {os.path.basename(gff_path)}...")
    
    try:
        # Read GFF file (tab-separated, skip comment lines starting with #)
        ref = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
        
        # GFF columns: seqid, source, type, start, end, score, strand, phase, attributes
        # We want columns 2 (type), 3 (start), 4 (end)
        
        # Filter for CDS features (protein-coding sequences)
        if (ref[2] == "CDS").sum() > 0:
            ref_genes = ref[ref[2] == "CDS"][[3, 4]].copy()
            print(f"  Using CDS features")
        elif (ref[2] == "gene").sum() > 0:
            ref_genes = ref[ref[2] == "gene"][[3, 4]].copy()
            print(f"  Using gene features")
        else:
            ref_genes = ref[[3, 4]].copy()
            print(f"  Using all features")
        
        # Rename columns and remove duplicates
        ref_genes.columns = ["start", "end"]
        ref_genes = ref_genes.drop_duplicates()
        
        # Convert to set of tuples for O(1) lookup
        ref_gene_set = set(zip(ref_genes['start'], ref_genes['end']))
        
        print(f"  [DONE] Loaded {len(ref_gene_set):,} reference genes")
        return ref_gene_set
        
    except Exception as e:
        print(f"[ERROR] Error loading reference genes from {gff_path}: {e}")
        return set()


def get_reference_orfs_from_gff(gff_path: str) -> List[Dict[str, any]]:
    """Extract reference ORF details (start, end, strand, length) from GFF."""
    try:
        gff = pd.read_csv(gff_path, sep="\t", comment="#", header=None)
        cds = gff[gff[2] == "CDS"]
        
        reference_orfs = []
        for _, row in cds.iterrows():
            start = int(row[3])
            end = int(row[4])
            reference_orfs.append({
                'start': start,
                'end': end,
                'strand': row[6],
                'length': end - start + 1
            })
        
        print(f"[DONE] Extracted {len(reference_orfs):,} reference ORFs from GFF")
        return reference_orfs
        
    except Exception as e:
        print(f"[ERROR] Error extracting ORFs from {gff_path}: {e}")
        return []


def get_gff_path(genome_id: str) -> str:
    """
    Get GFF path for a genome from default location.
    
    Args:
        genome_id: NCBI accession ID
    
    Returns:
        str: Path to GFF file
    """
    return os.path.join(get_data_dir("full_dataset"), f"{genome_id}.gff")

def get_fasta_path(genome_id: str) -> str:
    """
    Get FASTA path for a genome from default location.
    
    Args:
        genome_id: NCBI accession ID
    
    Returns:
        str: Path to FASTA file
    """
    return os.path.join(get_data_dir("full_dataset"), f"{genome_id}.fasta")

def print_dna_sequence(
    genome_seq: str,
    start: int,
    end: int,
    line_width: int = 60,
    full: bool = False
):
    """Print DNA sequence with position labels (1-indexed, inclusive)."""
    sequence = genome_seq[start-1:end]
    
    print(f"Position: {start:,} - {end:,}")
    print(f"Length: {len(sequence)} bp")
    print(f"Sequence:")
    
    if full:
        print(sequence)
    else:
        for i in range(0, len(sequence), line_width):
            pos = start + i
            line = sequence[i:i+line_width]
            print(f"{pos:>10,}  {line}")
    print()


def download_all_test_genomes(
    output_dir: str = None,
    email: str = NCBI_EMAIL
) -> Dict[str, Dict[str, str]]:
    """Download all genomes in TEST_GENOMES list with progress tracking."""
    
    # Use default data directory if not specified
    if output_dir is None:
        output_dir = get_data_dir("full_dataset")
    
    print("="*60)
    print(f"Downloading {len(TEST_GENOMES)} genomes (10 Bacteria + 5 Archaea)")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    results = {}
    
    for i, genome_id in enumerate(TEST_GENOMES, 1):
        print(f"\n[{i}/{len(TEST_GENOMES)}] Processing: {genome_id}")
        print("-"*60)
        
        fasta_path, gff_path = download_genome_and_reference(
            genome_id, 
            output_dir, 
            email
        )
        
        if fasta_path and gff_path:
            results[genome_id] = {
                'fasta': fasta_path,
                'gff': gff_path
            }
            print(f"[SUCCESS]")
        else:
            print(f"[FAILED] Could not download {genome_id}")
    
    print("\n" + "="*60)
    print(f"Download complete: {len(results)}/{len(TEST_GENOMES)} genomes")
    print("="*60)
    
    return results

def cleanup_generated_files(interactive: bool = True) -> Dict[str, int]:
    """
    Clean up downloaded genomes and prediction results.
    
    Args:
        interactive: If True, ask for confirmation. If False, delete silently.
    
    Returns:
        Dict with counts of deleted files by category
    """
    project_root = get_project_root()
    
    # Define directories and patterns to clean
    targets = {
        'downloads': {
            'dir': project_root / 'data' / 'full_dataset',
            'patterns': ['*.fasta', '*.gff', '*.gbk']
        },
        'results': {
            'dir': project_root / 'results',
            'patterns': ['*.gff', '*.txt', '*.log']
        }
    }
    
    # Collect all files to clean
    files_by_category = {}
    total_files = 0
    
    for category, info in targets.items():
        if not info['dir'].exists():
            continue
        
        files = []
        for pattern in info['patterns']:
            files.extend(info['dir'].glob(pattern))
        
        if files:
            files_by_category[category] = files
            total_files += len(files)
    
    if total_files == 0:
        if interactive:
            print("[+] No generated files found to clean.")
        return {'downloads': 0, 'results': 0}
    
    # Interactive mode: show files and ask for confirmation
    if interactive:
        print(f"\nFound {total_files} file(s) to clean:\n")
        
        for category, files in files_by_category.items():
            print(f"{category.upper()} ({targets[category]['dir']}):")
            for i, file in enumerate(files, 1):
                size_kb = file.stat().st_size / 1024
                print(f"  [{i}] {file.name:<40} ({size_kb:>8.1f} KB)")
            print()
        
        confirm = input(f"Delete all {total_files} file(s)? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Cleanup cancelled.")
            return {'downloads': 0, 'results': 0}
    
    # Delete files
    deleted_counts = {'downloads': 0, 'results': 0}
    
    for category, files in files_by_category.items():
        for file in files:
            try:
                file.unlink()
                deleted_counts[category] += 1
                if interactive:
                    print(f"  [+] Deleted: {file.name}")
            except Exception as e:
                if interactive:
                    print(f"  [!] Error deleting {file.name}: {e}")
    
    if interactive:
        print(f"\n[+] Cleanup complete!")
        print(f"    Downloads: {deleted_counts['downloads']} files")
        print(f"    Results:   {deleted_counts['results']} files")
    
    return deleted_counts


def main():
    """Download all test genomes and display summary."""
    print("Bacterial Gene Prediction - Data Management")
    print("="*60)
    
    results = download_all_test_genomes()
    
    if results:
        print("\nDownloaded files:")
        for genome_id, paths in results.items():
            print(f"\n{genome_id}:")
            print(f"  FASTA: {paths['fasta']}")
            print(f"  GFF:   {paths['gff']}")
    else:
        print("\n[ERROR] No genomes were downloaded successfully")


if __name__ == "__main__":
    main()
