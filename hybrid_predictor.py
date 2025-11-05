"""
Hybrid Gene Predictor - Main Pipeline

Supports modes:
1. Browse Catalog - Choose from 100 well-studied genomes
2. NCBI Download - Enter any NCBI accession number
3. Your Own FASTA - Analyze your own genome file
4. Validate Results - Compare predictions to reference
5. Cleanup Files - Delete downloaded genomes and results
6. Exit
"""

import sys
import argparse
import os
from pathlib import Path
import re
from typing import List, Dict


script_dir = Path(__file__).parent.resolve()
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

# Now import with proper error handling
try:
    from src import config  # type: ignore
    from src.data_management import cleanup_generated_files  # type: ignore
    from src.validation import validate_from_results_directory  # type: ignore
except ImportError:
    try:
        import config  # type: ignore
        from data_management import cleanup_generated_files  # type: ignore
        from validation import validate_from_results_directory  # type: ignore
    except ImportError as e:
        print(f"[!] Import error: {e}", file=sys.stderr)
        sys.exit(1)


# Extract config variables
NCBI_EMAIL = config.NCBI_EMAIL
GENOME_CATALOG = config.GENOME_CATALOG
get_genome_by_id = config.get_genome_by_id
get_genome_by_accession = config.get_genome_by_accession


def print_banner():
    """Print a nice welcome banner"""
    print("\n" + "="*80)
    print("            HYBRID BACTERIAL GENE PREDICTOR")
    print("            Interactive Mode")
    print("="*80 + "\n")


def interactive_menu():
    """Show interactive menu and get user choice"""
    print_banner()
    
    print("How would you like to predict genes?\n")
    print("  [1] Browse Catalog    - Choose from 100 well-studied genomes")
    print("  [2] NCBI Download     - Enter any NCBI accession number")
    print("  [3] Your Own FASTA    - Analyze your own genome file")
    print("  [4] Validate Results  - Compare predictions to reference")
    print("  [5] Cleanup Files     - Delete downloaded genomes and results")
    print("  [6] Exit")
    print()
    
    while True:
        choice = input("Enter your choice [1-6]: ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")


def browse_catalog():
    """Interactive catalog browser"""
    print("\n" + "="*80)
    print("GENOME CATALOG - 100 Well-Studied Organisms")
    print("="*80 + "\n")
    
    # Show groups
    groups = {}
    for genome in GENOME_CATALOG:
        group = genome['group']
        if group not in groups:
            groups[group] = []
        groups[group].append(genome)
    
    print("Available taxonomic groups:")
    for i, group in enumerate(groups.keys(), 1):
        print(f"  [{i}] {group} ({len(groups[group])} genomes)")
    print(f"  [A] Show All ({len(GENOME_CATALOG)} genomes)")
    print()
    
    # Get group choice
    while True:
        choice = input("Choose a group [1-4 or A]: ").strip().upper()
        
        if choice == 'A':
            genomes_to_show = GENOME_CATALOG
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(groups):
            group_name = list(groups.keys())[int(choice) - 1]
            genomes_to_show = groups[group_name]
            break
        print("Invalid choice. Try again.")
    
    # Display genomes
    print("\n" + "-"*80)
    current_group = None
    for genome in genomes_to_show:
        if genome['group'] != current_group:
            current_group = genome['group']
            print(f"\n{current_group}:")
            print("-" * 80)
        print(f"  {genome['id']:3d}. {genome['accession']:<15} {genome['name']}")
    print("-"*80 + "\n")
    
    # Get genome choice
    while True:
        choice = input(f"Enter genome number [1-{len(GENOME_CATALOG)}] or 'B' to go back: ").strip().upper()
        
        if choice == 'B':
            return None
        
        if choice.isdigit():
            genome_id = int(choice)
            genome = get_genome_by_id(genome_id)
            if genome:
                print(f"\n[+] Selected: {genome['name']}")
                print(f"    Accession: {genome['accession']}")
                confirm = input("\nProceed with this genome? [Y/n]: ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    return ('ncbi', genome['accession'], genome['name'])
            else:
                print(f"Genome #{genome_id} not found in catalog.")
        else:
            print("Please enter a valid number or 'B'.")


def ncbi_download():
    """Interactive NCBI accession input"""
    print("\n" + "="*80)
    print("NCBI DOWNLOAD")
    print("="*80 + "\n")
    
    print("Enter an NCBI accession number (e.g., NC_000913.3)")
    print("Format: XX_XXXXXX.X where XX is 2 letters (NC, GC, etc.)")
    print("Type 'B' to go back\n")
    
    ncbi_pattern = r'^[A-Z]{2}_\d{6,}\.\d+$'
    
    while True:
        accession = input("Accession: ").strip()
        
        if accession.upper() == 'B':
            return None
        
        if re.match(ncbi_pattern, accession):
            # Check if it's in catalog
            genome = get_genome_by_accession(accession)
            if genome:
                print(f"\n[+] Found in catalog: {genome['name']}")
            else:
                print(f"\n[+] Valid accession format: {accession}")
                print("    (Not in our catalog, but will download from NCBI)")
            
            # Get email
            print("\nNCBI requires an email address for downloads.")
            email = input("Your email: ").strip()
            
            if email and '@' in email:
                confirm = input(f"\nDownload {accession}? [Y/n]: ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    return ('ncbi', accession, email)
            else:
                print("Invalid email address.")
        else:
            print("Invalid accession format. Must be like: NC_000913.3")


def fasta_upload():
    """Interactive FASTA file selection"""
    print("\n" + "="*80)
    print("YOUR OWN FASTA FILE")
    print("="*80 + "\n")
    
    print("Provide the path to your FASTA file")
    print("Supported formats: .fasta, .fa, .fna, .fasta.gz")
    print("Results will be saved to: ./results/")
    print("Type 'B' to go back\n")
    
    while True:
        file_path = input("File path: ").strip().strip('"').strip("'")
        
        if file_path.upper() == 'B':
            return None
        
        path = Path(file_path)
        
        if not path.exists():
            print(f"[!] File not found: {file_path}")
            print("    Make sure the path is correct and try again.")
            continue
        
        if path.suffix.lower() not in ['.fasta', '.fa', '.fna', '.gz']:
            print(f"[!] Unsupported file type: {path.suffix}")
            print("    Must be .fasta, .fa, .fna, or .fasta.gz")
            continue
        
        # File is valid
        print(f"\n[+] Found: {path.name}")
        print(f"    Size: {path.stat().st_size / 1024:.1f} KB")
        
        # Automatically use results folder
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        output = str(results_dir / path.with_suffix('.gff').name)
        print(f"    Output will be saved to: {output}")
        
        confirm = input(f"\nAnalyze {path.name}? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            return ('fasta', str(path), output)


def validate_menu():
    """Interactive validation menu"""
    print("\n" + "="*80)
    print("VALIDATE PREDICTIONS")
    print("="*80 + "\n")
    
    print("How would you like to select the prediction file?\n")
    print("  [1] Enter genome ID manually")
    print("  [2] Choose from results/ directory (NCBI genomes only)")
    print("  [B] Go back")
    print()
    
    while True:
        choice = input("Choose [1/2/B]: ").strip().upper()
        
        if choice == 'B':
            return None
        
        elif choice == '1':
            # Manual genome ID entry
            print("\nEnter the genome ID (e.g., NC_000913.3):")
            genome_id = input("Genome ID: ").strip()
            
            if not genome_id:
                continue
            
            # Check if results file exists
            results_dir = Path('results')
            pred_file = results_dir / f'{genome_id}_predictions.gff'
            
            if not pred_file.exists():
                print(f"[!] Prediction file not found: {pred_file}")
                print("    Make sure you've run predictions for this genome first.")
                continue
            
            return run_validation(genome_id)
        
        elif choice == '2':
            # Browse results directory for NCBI genomes
            results_dir = Path('results')
            
            if not results_dir.exists():
                print("[!] results/ directory doesn't exist yet.")
                print("    Run predictions first!")
                input("\nPress Enter to continue...")
                return None
            
            # Find all GFF files with NCBI accessions
            all_gff = sorted(results_dir.glob('*_predictions.gff'))
            ncbi_files = []
            
            for file in all_gff:
                # Extract genome ID from filename
                genome_id = file.stem.replace('_predictions', '')
                # Check if it looks like NCBI accession
                if re.match(r'^[A-Z]{2}_\d{6,}\.\d+$', genome_id):
                    ncbi_files.append((file, genome_id))
            
            if not ncbi_files:
                print("[!] No NCBI prediction files found in results/")
                print("    Only files with NCBI accessions (NC_XXXXXX.X) can be auto-validated.")
                print("    Use option [1] for manual validation.")
                input("\nPress Enter to continue...")
                return None
            
            # Display files
            print("\nAvailable NCBI prediction files:\n")
            for i, (file, genome_id) in enumerate(ncbi_files, 1):
                size_kb = file.stat().st_size / 1024
                genome = get_genome_by_accession(genome_id)
                
                if genome:
                    print(f"  [{i}] {file.name:<45} ({size_kb:>6.1f} KB) - {genome['name']}")
                else:
                    print(f"  [{i}] {file.name:<45} ({size_kb:>6.1f} KB) - {genome_id}")
            
            print()
            
            while True:
                file_choice = input(f"Choose file [1-{len(ncbi_files)}] or 'B' to go back: ").strip().upper()
                
                if file_choice == 'B':
                    return None
                
                if file_choice.isdigit() and 1 <= int(file_choice) <= len(ncbi_files):
                    selected_file, genome_id = ncbi_files[int(file_choice) - 1]
                    print(f"\n[+] Selected: {selected_file.name}")
                    print(f"[+] Genome ID: {genome_id}")
                    return run_validation(genome_id)
                
                print("Invalid choice. Try again.")
        
        else:
            print("Invalid choice. Please enter 1, 2, or B.")


def run_validation(genome_id: str):
    """Run validation and save report"""
    try:
        # Call the validation function
        metrics = validate_from_results_directory(genome_id)
        
        # Save report
        results_dir = Path('results')
        report_path = results_dir / f'{genome_id}_validation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VALIDATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Genome ID:   {genome_id}\n")
            f.write(f"Prediction:  {metrics['results_file']}\n")
            f.write(f"Reference:   {metrics['reference_file']}\n")
            f.write("\n")
            f.write(f"Reference genes:       {metrics['reference_count']:,}\n")
            f.write(f"Predicted genes:       {metrics['predicted_count']:,}\n")
            f.write("\n")
            f.write(f"True Positives (TP):   {metrics['true_positives']:,}\n")
            f.write(f"False Positives (FP):  {metrics['false_positives']:,}\n")
            f.write(f"False Negatives (FN):  {metrics['false_negatives']:,}\n")
            f.write("\n")
            f.write(f"Sensitivity (Recall):  {metrics['sensitivity']:.4f}\n")
            f.write(f"Precision:             {metrics['precision']:.4f}\n")
            f.write(f"F1 Score:              {metrics['f1_score']:.4f}\n")
            f.write("="*80 + "\n")
        
        print(f"\n[+] Report saved to: {report_path}")
        input("\nPress Enter to continue...")
        
    except FileNotFoundError as e:
        print(f"\n[!] File not found: {e}")
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\n[!] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to continue...")


def cleanup_files():
    """Interactive cleanup - delegates to data_management module"""
    print("\n" + "="*80)
    print("CLEANUP FILES")
    print("="*80 + "\n")
    
    print("This will clean up:")
    print("  - Downloaded genomes (data/full_dataset/)")
    print("  - Prediction results (results/)")
    print()
    
    # Call the data_management cleanup function
    cleanup_generated_files(interactive=True)
    
    input("\nPress Enter to continue...")


def detect_input_mode(input_str: str) -> tuple:
    """
    Detect which mode to use based on input string.
    (CLI mode - skips menu)
    """
    # Check if it's a catalog number (1-100)
    try:
        catalog_id = int(input_str)
        if 1 <= catalog_id <= 100:
            genome = get_genome_by_id(catalog_id)
            if genome:
                print(f"[+] Catalog #{catalog_id}: {genome['name']}")
                return 'ncbi', genome['accession']
    except ValueError:
        pass
    
    # Check if it looks like NCBI accession
    ncbi_pattern = r'^[A-Z]{2}_\d{6,}\.\d+$'
    if re.match(ncbi_pattern, input_str):
        genome = get_genome_by_accession(input_str)
        if genome:
            print(f"[+] Found in catalog: {genome['name']}")
        return 'ncbi', input_str
    
    # Check if it's a file path
    path = Path(input_str)
    if path.exists():
        if path.suffix.lower() in ['.fasta', '.fa', '.fna', '.fasta.gz']:
            return 'fasta', str(path)
        else:
            raise ValueError(f"File exists but not recognized as FASTA: {input_str}")
    
    raise ValueError(
        f"Could not determine input type for: {input_str}\n"
        f"  - Not a catalog number (1-100)\n"
        f"  - Doesn't match NCBI accession pattern (NC_XXXXXX.X)\n"
        f"  - File doesn't exist\n"
        f"  Use --list to see available genomes"
    )


def predict_ncbi_genome(
    accession: str, 
    email: str = None,
    use_ml: bool = True,
    ml_threshold: float = 0.1,
    use_final_filtration_ml: bool = True,
    final_ml_threshold: float = 0.12
):
    """Download genome from NCBI and predict genes"""
    from Bio import Entrez
    import gzip
    
    print(f"\n{'='*80}")
    print(f"MODE: NCBI DOWNLOAD")
    print(f"{'='*80}")
    print(f"Accession: {accession}")
    
    if email is None:
        raise ValueError("Email required for NCBI downloads")
    
    print(f"Email: {email}")
    
    # Set up Entrez
    Entrez.email = email
    
    # Create data/full_dataset directory if it doesn't exist
    downloads_dir = Path('data') / 'full_dataset'
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the genome
    fasta_path = downloads_dir / f'{accession}.fasta'
    
    print(f"\nDownloading from NCBI...")
    print(f"  Target: {fasta_path}")
    
    try:
        # Search for the nucleotide record
        print("  Fetching sequence...")
        handle = Entrez.efetch(
            db="nucleotide",
            id=accession,
            rettype="fasta",
            retmode="text"
        )
        
        # Save to file
        with open(fasta_path, 'w') as f:
            fasta_content = handle.read()
            f.write(fasta_content)
        handle.close()
        
        # Verify download
        file_size = fasta_path.stat().st_size
        print(f"  Downloaded: {file_size:,} bytes")
        
        if file_size < 100:
            raise ValueError("Downloaded file is too small, download may have failed")
        
        print(f"  ✓ Genome downloaded successfully\n")
        
    except Exception as e:
        print(f"\n[!] Download failed: {e}", file=sys.stderr)
        raise
    
    # Now predict genes using the downloaded file
    print(f"{'='*80}")
    print(f"RUNNING GENE PREDICTION")
    print(f"{'='*80}\n")
    
    # Set output path using genome_id for results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f'{accession}_predictions.gff'
    
    # Call predict_fasta_file with the downloaded genome
    try:
        predictions = predict_fasta_file(
            fasta_path=str(fasta_path),
            use_ml=use_ml,
            ml_threshold=ml_threshold,
            use_final_filtration_ml=use_final_filtration_ml,
            final_ml_threshold=final_ml_threshold
        )
        
        print(f"\n{'='*80}")
        print(f"NCBI PREDICTION COMPLETE!")
        print(f"{'='*80}")
        print(f"Genome:      {accession}")
        print(f"Downloaded:  {fasta_path}")
        print(f"Predictions: {output_path}")
        print(f"Total genes: {len(predictions):,}")
        print(f"{'='*80}\n")
        
        return predictions
        
    except Exception as e:
        print(f"\n[!] Prediction failed: {e}", file=sys.stderr)
        raise


def predict_fasta_file(
    fasta_path: str, 
    use_ml: bool = True, 
    ml_threshold: float = 0.1,
    use_final_filtration_ml: bool = True,
    final_ml_threshold: float = 0.12
):
    """Predict genes from FASTA file."""
    print(f"\n{'='*80}")
    print(f"MODE: RAW FASTA FILE")
    print(f"{'='*80}")
    print(f"Input: {fasta_path}")
    
    # Always results folder
    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(exist_ok=True)
    filename = Path(fasta_path).stem + '_predictions.gff'
    output_path = str(results_dir / filename)
    
    print(f"Output: {output_path}")
    print(f"ML group filtering: {'Enabled' if use_ml else 'Disabled'}")
    if use_ml:
        print(f"  Group ML threshold: {ml_threshold}")
    print(f"Final ML filtration: {'Enabled' if use_final_filtration_ml else 'Disabled'}")
    if use_final_filtration_ml:
        print(f"  Final ML threshold: {final_ml_threshold}")
    
    try:
        from src.data_management import load_genome_sequence
        from src.traditional_methods import (
            find_orfs_candidates, create_training_set, create_intergenic_set,
            build_all_scoring_models, score_all_orfs, filter_candidates,
            organize_nested_orfs, select_best_starts,
            FIRST_FILTER_THRESHOLD, SECOND_FILTER_THRESHOLD, START_SELECTION_WEIGHTS 
        )    
        from src.ml_models import OrfGroupClassifier, HybridGeneFilter
        
        print(f"\n{'='*80}\nSTEP 1: LOAD FASTA FILE\n{'='*80}")
        genome_info = load_genome_sequence(fasta_path)
        sequence = genome_info['sequence']
        print(f"Sequence loaded: {len(sequence):,} bp")
        
        print(f"\n{'='*80}\nSTEP 2: FIND ALL ORFs\n{'='*80}")
        all_orfs = find_orfs_candidates(sequence, min_length=100)
        print(f"ORFs found: {len(all_orfs):,}")
        
        print(f"\n{'='*80}\nSTEP 3: CREATE TRAINING SET\n{'='*80}")
        training_set = create_training_set(sequence=sequence, all_orfs=all_orfs)
        print(f"Training set: {len(training_set):,} ORFs")
        
        print(f"\n{'='*80}\nSTEP 4: CREATE INTERGENIC REGIONS\n{'='*80}")
        intergenic_set = create_intergenic_set(sequence=sequence, all_orfs=all_orfs)
        print(f"Intergenic regions: {len(intergenic_set):,}")
        
        print(f"\n{'='*80}\nSTEP 5: BUILD SCORING MODELS\n{'='*80}")
        models = build_all_scoring_models(training_set, intergenic_set)
        print(f"Models built: {len(models)} models")
        
        print(f"\n{'='*80}\nSTEP 6: SCORE ALL ORFs\n{'='*80}")
        scored_orfs = score_all_orfs(all_orfs, models)
        print(f"ORFs scored: {len(scored_orfs):,}")
        
        print(f"\n{'='*80}\nSTEP 7: INITIAL FILTERING\n{'='*80}")
        candidates = filter_candidates(scored_orfs, **FIRST_FILTER_THRESHOLD)
        print(f"Candidates: {len(candidates):,}")
        
        print(f"\n{'='*80}\nSTEP 8: GROUP NESTED ORFs\n{'='*80}")
        grouped_orfs = organize_nested_orfs(candidates)
        print(f"Groups: {len(grouped_orfs):,}")
        
        if use_ml:
            print(f"\n{'='*80}\nSTEP 9: ML GROUP FILTERING\n{'='*80}")
            try:
                classifier = OrfGroupClassifier()
                model_path = Path(__file__).parent / 'models' / 'orf_classifier_lgb.pkl'
                if model_path.exists():
                    classifier.load(str(model_path))
                    pre_filter_count = len(grouped_orfs)
                    grouped_orfs = classifier.filter_groups(
                        groups=grouped_orfs, genome_id="user_genome",
                        weights=START_SELECTION_WEIGHTS, threshold=ml_threshold
                    )
                    post_filter_count = len(grouped_orfs)
                    print(f"Groups before ML: {pre_filter_count:,}")
                    print(f"Groups after ML:  {post_filter_count:,}")
                    print(f"Groups removed:   {pre_filter_count - post_filter_count:,}")
                else:
                    print(f"[!] Model not found, skipping ML...")
            except Exception as e:
                print(f"[!] ML error: {e}, skipping...")
        
        print(f"\n{'='*80}\nSTEP 10: SELECT BEST START CODONS\n{'='*80}")
        top_candidates = select_best_starts(grouped_orfs, START_SELECTION_WEIGHTS)
        print(f"Top candidates: {len(top_candidates):,}")
        
        print(f"\n{'='*80}\nSTEP 11: FINAL FILTERING\n{'='*80}")
        final_predictions = filter_candidates(top_candidates, **SECOND_FILTER_THRESHOLD)
        print(f"Final predictions: {len(final_predictions):,}")
        
        # NEW: FINAL ML FILTRATION (STEP 11.5 - AFTER traditional filter)
        if use_final_filtration_ml:
            print(f"\n{'='*80}\nSTEP 11.5: HYBRID ML FILTRATION\n{'='*80}")
            try:
                hybrid_filter = HybridGeneFilter()
                model_path = Path(__file__).parent / 'models' / 'hybrid_best_model.pkl'
                if model_path.exists():
                    hybrid_filter.load(str(model_path))
                    pre_filter_count = len(final_predictions)
                    
                    final_predictions = hybrid_filter.filter_candidates(
                        candidates=final_predictions,
                        genome_id="user_genome",
                        threshold=final_ml_threshold,
                        batch_size=32  # Process 32 candidates at a time
                    )
                    post_filter_count = len(final_predictions)
                    print(f"Candidates before hybrid ML: {pre_filter_count:,}")
                    print(f"Candidates after hybrid ML:  {post_filter_count:,}")
                    print(f"Candidates removed:          {pre_filter_count - post_filter_count:,}")
                else:
                    print(f"[!] Hybrid model not found at {model_path}, skipping final ML filtration...")
            except Exception as e:
                print(f"[!] Hybrid ML error: {e}, skipping final filtration...")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}\nSTEP 12: WRITE OUTPUT\n{'='*80}")
        write_gff(final_predictions, output_path, sequence_id=Path(fasta_path).stem)
        
        print(f"\n{'='*80}\nPREDICTION COMPLETE!\n{'='*80}")
        print(f"Input:  {fasta_path}")
        print(f"Output: {output_path}")
        print(f"Size:   {len(sequence):,} bp")
        print(f"Genes:  {len(final_predictions):,}")
        print(f"{'='*80}\n")
        
        return final_predictions
        
    except Exception as e:
        print(f"\n[!] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise


def write_gff(predictions: List[Dict], output_path: str, sequence_id: str = "sequence"):
    """Write predictions to GFF3 format."""
    with open(output_path, 'w') as f:
        f.write("##gff-version 3\n")
        for i, pred in enumerate(predictions, 1):
            start = pred.get('genome_start', pred.get('start'))
            end = pred.get('genome_end', pred.get('end'))
            strand = '+' if pred.get('strand') == 'forward' else '-'
            score = pred.get('combined_score', 0.0)
            rbs = pred.get('rbs_score', 0.0)
            attrs = f"ID=gene_{i};rbs_score={rbs:.2f};combined_score={score:.2f}"
            f.write(f"{sequence_id}\tHybridPredictor\tCDS\t{start}\t{end}\t{score:.3f}\t{strand}\t0\t{attrs}\n")
    print(f"[+] Wrote {len(predictions)} predictions to {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Hybrid bacterial gene predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (no arguments)
  python hybrid_predictor.py
  
  # Command-line mode (skips menu)
  python hybrid_predictor.py --list
  python hybrid_predictor.py 1
  python hybrid_predictor.py NC_000913.3 --email you@example.com
  python hybrid_predictor.py mygenome.fasta
  
  # With ML options
  python hybrid_predictor.py mygenome.fasta --group-threshold 0.15 --final-threshold 0.2
  python hybrid_predictor.py mygenome.fasta --no-group-ml --no-final-ml
        """
    )
    
    parser.add_argument('input', nargs='?', help='Catalog number, NCBI accession, or FASTA file')
    parser.add_argument('--email', help='Email for NCBI (required for downloads)')
    
    # ML filtering options
    parser.add_argument('--group-threshold', type=float, default=0.1,
                       help='ML group filtering threshold (default: 0.1)')
    parser.add_argument('--final-threshold', type=float, default=0.12,
                       help='Final hybrid ML filtration threshold (default: 0.12)')
    parser.add_argument('--no-group-ml', action='store_true',
                       help='Disable ML group filtering')
    parser.add_argument('--no-final-ml', action='store_true',
                       help='Disable final hybrid ML filtration')
    
    parser.add_argument('-l', '--list', action='store_true',
                       help='List available genomes in catalog')
    parser.add_argument('--group', choices=['Proteobacteria', 'Firmicutes', 'Actinobacteria', 'Archaea'],
                       help='Filter list by taxonomic group')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Force interactive mode')
    
    args = parser.parse_args()
    
    # Handle --list (command-line mode - SKIPS MENU)
    if args.list:
        print("\n" + "="*80)
        print("GENOME CATALOG - 100 Well-Studied Organisms")
        print("="*80)
        
        if args.group:
            genomes = [g for g in GENOME_CATALOG if g['group'] == args.group]
            print(f"\nShowing: {args.group} ({len(genomes)} genomes)")
        else:
            genomes = GENOME_CATALOG
            print("\nUse --group to filter by: Proteobacteria, Firmicutes, Actinobacteria, Archaea")
        
        print()
        current_group = None
        for genome in genomes:
            if genome['group'] != current_group:
                current_group = genome['group']
                print(f"\n{current_group}:")
                print("-" * 80)
            print(f"  {genome['id']:3d}. {genome['accession']:<15} {genome['name']}")
        
        print("\n" + "="*80)
        print("Usage: python hybrid_predictor.py <number>")
        print("="*80 + "\n")
        sys.exit(0)
    
    # If input provided, use COMMAND-LINE MODE (SKIPS MENU)
    if args.input and not args.interactive:
        try:
            mode, resolved_input = detect_input_mode(args.input)
            print(f"\n[+] Detected mode: {mode.upper()}")
        except ValueError as e:
            print(f"\n[!] Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        try:
            if mode == 'ncbi':
                predict_ncbi_genome(
                    resolved_input, 
                    args.email,
                    use_ml=not args.no_group_ml,
                    ml_threshold=args.group_threshold,
                    use_final_filtration_ml=not args.no_final_ml,
                    final_ml_threshold=args.final_threshold
                )
            elif mode == 'fasta':
                predict_fasta_file(
                    resolved_input,
                    use_ml=not args.no_group_ml,
                    ml_threshold=args.group_threshold,
                    use_final_filtration_ml=not args.no_final_ml,
                    final_ml_threshold=args.final_threshold
                )
            
            print(f"\n[+] Success!")
            
        except NotImplementedError as e:
            print(f"\n[!] {e}", file=sys.stderr)
            print("This mode is under construction.")
            sys.exit(1)
        except Exception as e:
            print(f"\n[!] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        sys.exit(0)
    
    # INTERACTIVE MODE (no arguments provided, or --interactive flag)
    while True:
        choice = interactive_menu()
        
        if choice == '1':
            # Browse catalog
            result = browse_catalog()
            if result:
                mode, accession, name = result
                try:
                    predict_ncbi_genome(accession, args.email or NCBI_EMAIL)
                    print(f"\n[+] Success!")
                    break
                except Exception as e:
                    print(f"\n[!] Error: {e}")
                    input("\nPress Enter to continue...")
        
        elif choice == '2':
            # NCBI download
            result = ncbi_download()
            if result:
                mode, accession, email = result
                try:
                    predict_ncbi_genome(accession, email)
                    print(f"\n[+] Success!")
                    break
                except Exception as e:
                    print(f"\n[!] Error: {e}")
                    input("\nPress Enter to continue...")
        
        elif choice == '3':
            # FASTA file
            result = fasta_upload()
            if result:
                mode, fasta_path, _ = result 
                try:
                    predict_fasta_file(fasta_path)
                    print(f"\n[+] Success!")
                    break
                except Exception as e:
                    print(f"\n[!] Error: {e}")
                    input("\nPress Enter to continue...")
        
        elif choice == '4':
            # Validation
            validate_menu()
        
        elif choice == '5':
            # Cleanup
            cleanup_files()
        
        elif choice == '6':
            # Exit
            print("\nGoodbye!\n")
            sys.exit(0)


if __name__ == '__main__':
    main()