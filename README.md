# Bacterial Gene Prediction Tool

A hybrid machine learning and traditional algorithm approach to bacterial gene prediction. This tool combines classical bioinformatics methods (codon usage bias, Interpolated Markov Models, Ribosome Binding Site detection) with supervised learning to predict genes in bacterial and archaeal genomes.

## Overview

This tool provides both a **web interface** and **command-line interface** for predicting genes in bacterial genomes using four different modes:

1. **Catalog Mode** - Choose from 100 pre-selected well-studied genomes
2. **NCBI Download Mode** - Download and analyze any genome from NCBI by accession number
3. **FASTA File Mode** - Analyze your own genome FASTA file
4. **Validation Mode** - Compare predictions against reference annotations

The tool performs de novo gene prediction without requiring pre-downloaded training data. Each genome is analyzed independently using self-training on its own sequence characteristics.

**Key Features:**
- Self-training on target genome (no external training data required)
- Optional ML nested ORF filtering trained on 27 diverse prokaryotes (4 taxonomic families)
- Optional deep learning model for precision enhancement
- Works on any bacterial or archaeal genome
- Web interface and command-line modes
- Automatic validation against NCBI reference annotations
- Outputs standard GFF3 format

## Performance

Evaluated on 15 diverse bacterial and archaeal genomes:

**Without Deep Learning:**

| Metric | Average Score |
|--------|---------------|
| Sensitivity (Recall)| ~75% |
| Precision | ~65% |
| F1 Score | ~69% |

**With Deep Learning:**

| Metric | Average Score |
|--------|---------------|
| Sensitivity (Recall)| ~73% |
| Precision | ~80% |
| F1 Score | ~76% |

**ML Models Training:**
The optional LightGBM classifier and CNN+Dense models were trained on 27 prokaryotic genomes spanning 4 major taxonomic groups (Proteobacteria, Firmicutes, Actinobacteria, Archaea) to ensure generalization across different bacterial families with varying GC content, genome sizes, and codon usage patterns.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/roeimed0/bacterial-gene-prediction.git
cd bacterial-gene-prediction

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd gene-prediction-frontend
npm install
cd ..
```

## Usage Options

This tool provides **two ways** to interact with the system:

### Option 1: Web Application (Recommended)

The web interface provides an intuitive way to upload genomes, run predictions, and download results.

**Launch the web interface:**

**Windows - Using Batch File:**
```bat
launch_app.bat
```

This will automatically:
- Start the backend server (minimized terminal)
- Start the frontend server (minimized terminal)
- Open your browser to the application

**Manual Launch (Any Platform):**
```bash
# Terminal 1 - Start backend
conda activate gene_prediction  # If using conda
python -m uvicorn api.main:app --reload

# Terminal 2 - Start frontend
cd gene-prediction-frontend
npm run dev
```

**Access:**
- Application: http://localhost:5173

**For Developers:**
- Backend API runs on port 8000 (internal)
- API Documentation: http://localhost:8000/docs

**Web Interface Features:**
- Upload FASTA files via drag-and-drop or file browser
- Download genomes directly from NCBI by accession number
- Browse catalog of 100 pre-selected genomes organized by taxonomy
- Interactive validation against reference annotations
- Download predictions as GFF3 files
- View prediction statistics and quality metrics

**Web Workflow Example:**

1. **Run `launch_app.bat`** - Browser opens automatically
2. **Choose prediction method:**
   - Upload your own FASTA file
   - Enter an NCBI accession number (e.g., NC_000913.3)
   - Browse the genome catalog
3. **Configure options:**
   - Enable/disable ML filtering
   - Adjust ML thresholds if needed
4. **Run prediction** - Results appear in a few minutes
5. **Download GFF3 file** - Standard format compatible with genome browsers
6. **Validate** (optional) - Compare against reference for NCBI genomes

### Option 2: Command-Line Tool

The CLI provides scriptable access for batch processing and automation.

**Run the interactive tool:**

```bash
python hybrid_predictor.py
```

You'll see a menu:

```
================================================================================
            HYBRID BACTERIAL GENE PREDICTOR
            Interactive Mode
================================================================================

How would you like to predict genes?

  [1] Browse Catalog    - Choose from 100 well-studied genomes
  [2] NCBI Download     - Enter any NCBI accession number
  [3] Your Own FASTA    - Analyze your own genome file
  [4] Validate Results  - Compare predictions to reference
  [5] Cleanup Files     - Delete downloaded genomes and results
  [6] Exit

Enter your choice [1-6]:
```

## CLI Usage Modes

### Mode 1: Catalog Browser

Browse and select from 100 curated bacterial and archaeal genomes organized by taxonomy:

```bash
python hybrid_predictor.py
# Select option 1
# Choose a taxonomic group (Proteobacteria, Firmicutes, Actinobacteria, Archaea)
# Select a genome by number
```

Or use command-line mode to list genomes:

```bash
# List all genomes
python hybrid_predictor.py --list

# List genomes from specific group
python hybrid_predictor.py --list --group Proteobacteria

# Predict directly by catalog number
python hybrid_predictor.py 1
```

### Mode 2: NCBI Download

Download and analyze any bacterial genome from NCBI:

```bash
# Interactive mode
python hybrid_predictor.py
# Select option 2
# Enter accession (e.g., NC_000913.3)
# Enter your email (required by NCBI)

# Command-line mode
python hybrid_predictor.py NC_000913.3 --email your.email@example.com
```

The tool will:
1. Download the genome FASTA from NCBI to `data/full_dataset/`
2. Download the reference GFF annotation (for validation)
3. Run gene prediction
4. Save predictions to `results/{accession}_predictions.gff`

### Mode 3: Your Own FASTA File

Analyze your own bacterial genome:

```bash
# Interactive mode
python hybrid_predictor.py
# Select option 3
# Enter path to your FASTA file

# Command-line mode
python hybrid_predictor.py mygenome.fasta
```

**Supported formats:**
- `.fasta`, `.fa`, `.fna`
- Gzipped: `.fasta.gz`

**Output:**
- GFF3 file with predicted gene coordinates
- Automatically saves to `results/{filename}_predictions.gff`
- Output directory is always `results/` (cannot be changed)

### Mode 4: Validation

Compare your predictions against NCBI reference annotations:

```bash
# Interactive mode
python hybrid_predictor.py
# Select option 4
# Choose [1] to enter genome ID manually (e.g., NC_000913.3)
# OR choose [2] to browse available predictions in results/
```

**Example validation output:**
```
================================================================================
VALIDATION RESULTS
================================================================================

Genome ID: NC_000913.3
Reference genes:       4,340
Predicted genes:       4,933

True Positives (TP):   3,486
False Positives (FP):  1,447
False Negatives (FN):  854

Sensitivity (Recall):  80.32%
Precision:             70.67%
F1 Score:              0.7519
================================================================================
```

**Validation is only available for:**
- Genomes downloaded from NCBI (have reference annotations)
- Files with NCBI accession in filename (e.g., NC_000913.3_predictions.gff)
- Cannot validate custom FASTA files without reference

## Command-Line Options

```bash
# Show help
python hybrid_predictor.py --help

# List available genomes
python hybrid_predictor.py --list
python hybrid_predictor.py --list --group Proteobacteria

# Predict from catalog genome (by number)
python hybrid_predictor.py 1

# Predict from NCBI accession
python hybrid_predictor.py NC_000913.3 --email your@email.com

# Predict from FASTA file
python hybrid_predictor.py genome.fasta

# Adjust group ML threshold (default: 0.1, range: 0.0-1.0)
python hybrid_predictor.py genome.fasta --ml-threshold 0.2

# Adjust hybrid ML threshold (default: 0.12, range: 0.0-1.0)
python hybrid_predictor.py genome.fasta --final-ml-threshold 0.2

# Disable group ML filtering
python hybrid_predictor.py genome.fasta --no-group-ml

# Disable final ML filtering
python hybrid_predictor.py genome.fasta --no-final-ml

# Force interactive mode even with arguments
python hybrid_predictor.py --interactive
```

## Output Format

Results are saved in GFF3 format (compatible with genome browsers like IGV, Artemis, JBrowse):

```gff
##gff-version 3
sequence_id  HybridPredictor  CDS  start  end  score  strand  0  ID=gene_1;rbs_score=0.85;combined_score=2.45
```

**Columns:**
- `start`, `end`: Gene coordinates (1-indexed, inclusive)
- `strand`: `+` (forward) or `-` (reverse)
- `score`: Combined prediction confidence score
- `ID`: Unique gene identifier
- `rbs_score`: Ribosome Binding Site detection score
- `combined_score`: Weighted combination of all scoring features

## Algorithm Pipeline

### Step 1: ORF Detection (Optimized)
- Scans both DNA strands (forward and reverse complement)
- Identifies all Open Reading Frames with start codons: ATG, GTG, TTG
- Minimum length: 100 bp (configurable in `src/config.py`)
- Optimized with LRU caching for ~3x speedup on RBS motif scoring

### Step 2: RBS Detection
- Searches upstream regions (-20 to -5 bp) for Shine-Dalgarno motifs
- Identifies purine-rich regions (AGGAGGU consensus)
- Scores potential ribosome binding sites

### Step 3: Self-Training
- Identifies high-confidence genes (long ORFs with strong signals)
- Builds genome-specific models from the target genome itself:
  - Codon usage frequency tables (coding vs non-coding)
  - Interpolated Markov Models - order is modified to fit genome length (context-dependent scoring)
- No external training data required

### Step 4: Traditional Scoring
Each ORF is scored using five features:
- **Codon Bias**: Likelihood ratio of codon usage (coding vs non-coding patterns)
- **IMM Score**: Interpolated Markov Model score (sequence context modeling)
- **Length Score**: Logarithmic scaling favoring longer ORFs
- **RBS Score**: Ribosome binding site strength
- **Start Codon Type**: Weight preference: ATG > GTG > TTG

### Step 5: Initial Filtering
- Threshold-based filtering
- Thresholds optimized on training data from diverse genomes
- Removes low-scoring ORF candidates

### Step 6: Grouping & Overlap Resolution
- Groups nested ORFs (same stop codon, different start codons)
- Identifies overlapping gene predictions
- Prepares for start site selection

### Step 7: ML-Based Refinement (Optional)
- LightGBM binary classifier filters false positive gene groups
- Extracts 31 aggregate features per group:
  - Score distributions (mean, std, min, max)
  - Feature margins between top candidates
  - Entropy measures
- **Training data**: 27 diverse prokaryotic genomes from 4 major taxonomic families:
  - Proteobacteria (Gram-negative bacteria)
  - Firmicutes (Gram-positive bacteria)
  - Actinobacteria (high GC content)
  - Archaea
- Trained to generalize across different:
  - GC content ranges (25%-75%)
  - Genome sizes (1.5-10 Mbp)
  - Codon usage patterns
- Requires: `models/orf_classifier_lgb.pkl` (optional - prediction works without it)
- Default threshold: 0.1 (adjustable via `--ml-threshold`)

### Step 8: Start Site Selection
- Selects optimal start codon for each gene
- Weighted scoring using optimized weights:
  - codon bias: 4.86
  - length: 7.44
  - IMM: 1.01
  - RBS: 0.64
  - start type: 0.28

### Step 9: Traditional Second Filtering
- Threshold-based filtering
- Thresholds optimized on training data from diverse genomes
- Removes low-scoring ORF candidates

### Step 10: Hybrid ML Filtration (Optional)
- Deep-learning–based binary classifier that refines the final candidate genes after traditional filtering and start-site selection
- Combines sequence-based embeddings (via CNN) with traditional features such as codon bias, IMM score, RBS score, and ORF length
- Reduces residual false positives that remain after the main LightGBM group filter
- Model file: `models/hybrid_best_model.pkl`
- Default threshold: 0.12 (adjustable via `--final-ml-threshold`)
- Can be disabled with `--no-final-ml`

## Project Structure

```
bacterial-gene-prediction/
├── launch_app.bat              # Web app launcher (Windows)
├── hybrid_predictor.py         # Command-line interface
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── LICENSE
├── README.md
│
├── api/                        # FastAPI backend
│   ├── main.py                # API entry point with all routes
│   ├── models.py              # Pydantic models for API
│   └── __init__.py
│
├── gene-prediction-frontend/   # React web interface
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout/
│   │   │   │   ├── Header.jsx          # Top navigation bar
│   │   │   │   ├── MainContent.jsx     # Main content wrapper
│   │   │   │   └── ModeBar.jsx         # Mode selection tabs
│   │   │   ├── Modes/
│   │   │   │   ├── CatalogMode.jsx     # Genome catalog browser
│   │   │   │   ├── FastaMode.jsx       # File upload interface
│   │   │   │   ├── NcbiMode.jsx        # NCBI download interface
│   │   │   │   └── ValidateMode.jsx    # Validation interface
│   │   │   ├── PipelineVisualization/
│   │   │   │   ├── index.jsx           # Main pipeline component
│   │   │   │   ├── PipelineSteps.jsx   # Step-by-step visualization
│   │   │   │   ├── PipelineStep.jsx    # Individual step card
│   │   │   │   ├── KeyFeatures.jsx     # Feature highlights
│   │   │   │   ├── FeatureCard.jsx     # Individual feature card
│   │   │   │   ├── MLToggle.jsx        # ML options toggle
│   │   │   │   ├── ModeSelector.jsx    # Mode selection component
│   │   │   │   └── pipelineData.js     # Pipeline configuration data
│   │   │   ├── Results/
│   │   │   │   ├── ResultsView.jsx     # Main results display
│   │   │   │   ├── ResultsTable.jsx    # Prediction table
│   │   │   │   └── GenomeViewer.jsx    # Genome visualization
│   │   │   ├── FileManager.jsx         # File management utilities
│   │   │   └── ProcessManager.jsx      # Process status management
│   │   ├── services/
│   │   │   └── api.js         # API client functions
│   │   ├── App.jsx            # Main application component
│   │   ├── App.css            # Application styles
│   │   ├── main.jsx           # React entry point
│   │   └── index.css          # Global styles
│   ├── public/
│   ├── package.json           # Node dependencies
│   ├── vite.config.js         # Vite build configuration
│   ├── tailwind.config.js     # Tailwind CSS configuration
│   ├── postcss.config.js      # PostCSS configuration
│   └── eslint.config.js       # ESLint configuration
│
├── src/                        # Core prediction algorithms
│   ├── config.py              # Configuration & genome catalog (100 genomes)
│   ├── data_management.py     # NCBI download & file I/O
│   ├── traditional_methods.py # ORF detection & scoring algorithms
│   ├── comparative_analysis.py# Validation metrics computation
│   ├── validation.py          # Validation wrapper functions
│   ├── ml_models.py           # Optional ML classifiers
│   ├── cache.py               # Caching utilities
│   └── __init__.py
│
├── models/                     # Trained ML models (optional)
│   ├── orf_classifier_lgb.pkl # LightGBM group filter
│   ├── hybrid_best_model.pkl  # Deep learning final filter
│   └── feature_names.pkl      # Feature name mappings
│
├── notebooks/                  # Jupyter notebooks (research & development)
│   ├── 01_data_collection.ipynb
│   ├── 02_orf_detection.ipynb
│   ├── 03_ml_models.ipynb
│   ├── hybrid_dl_classifier.ipynb
│   └── orf_detection_optimized.ipynb
│
├── data/                       # Downloaded genomes (auto-created)
│   ├── full_dataset/          # NCBI genome downloads
│   ├── hybrid_dl_training/    # Training data for deep learning model
│   └── processed/             # Processed genome data
│
└── results/                    # Output directory (auto-created)
    ├── *_predictions.gff      # Prediction files (GFF3 format)
    └── *_validation_report.txt# Validation reports
```

## Configuration

Edit `src/config.py` to customize:

```python
# NCBI email (required for downloads)
NCBI_EMAIL = "your.email@example.com"

# Minimum ORF length
MIN_ORF_LENGTH = 100  # base pairs

# Start and stop codons
START_CODONS = {'ATG', 'GTG', 'TTG'}
STOP_CODONS = {'TAA', 'TAG', 'TGA'}

# Scoring weights (generalized over 15 genomes)
START_SELECTION_WEIGHTS = {
    'codon': 4.8562,
    'imm': 1.0107,
    'rbs': 0.6383,
    'length': 7.4367,
    'start': 0.2755
}

# ML thresholds (adjustable)
ML_THRESHOLD = 0.1        # Group filter
FINAL_ML_THRESHOLD = 0.12 # Hybrid filter
```

## Examples

### Web Interface Examples

**Example 1: Upload and analyze your genome**
1. Run `launch_app.bat`
2. Click "Upload FASTA File"
3. Drag and drop `my_genome.fasta`
4. Click "Predict Genes"
5. Wait for analysis to complete
6. Download `my_genome_predictions.gff`

**Example 2: Download from NCBI and validate**
1. Click "NCBI Download"
2. Enter accession: `NC_000913.3`
3. Enter email: `your@email.com`
4. Click "Download & Predict"
5. After completion, click "Validate Results"
6. View validation metrics in browser

**Example 3: Browse catalog**
1. Click "Browse Catalog"
2. Select "Proteobacteria"
3. Choose "Escherichia coli K-12 MG1655"
4. Click "Predict Genes"
5. Download results

### Command-Line Examples

**Example 1: Analyze E. coli from catalog**

```bash
python hybrid_predictor.py
# Choose [1] Browse Catalog
# Choose [1] Proteobacteria
# Choose 1 (E. coli K-12 MG1655 - NC_000913.3)

# Output: results/NC_000913.3_predictions.gff
```

Then validate:
```bash
python hybrid_predictor.py
# Choose [4] Validate Results
# Choose [2] Browse files
# Select NC_000913.3_predictions.gff

# Creates: results/NC_000913.3_validation_report.txt
```

**Example 2: Download and predict from NCBI**

```bash
python hybrid_predictor.py NC_002516.2 --email your@email.com

# Downloads to: data/full_dataset/NC_002516.2.fasta
# Predictions: results/NC_002516.2_predictions.gff
```

**Example 3: Analyze your own genome**

```bash
python hybrid_predictor.py my_bacterial_genome.fasta

# Results: results/my_bacterial_genome_predictions.gff
# Note: Cannot validate without reference annotation
```

**Example 4: Adjust ML thresholds**

```bash
# More lenient group filter (more predictions)
python hybrid_predictor.py NC_000913.3 --email user@email.com --ml-threshold 0.05

# More strict final filter (higher precision)
python hybrid_predictor.py NC_000913.3 --email user@email.com --final-ml-threshold 0.2
```

**Example 5: Command-line workflow**

```bash
# List Actinobacteria genomes
python hybrid_predictor.py --list --group Actinobacteria

# Predict from catalog #51 (Mycobacterium tuberculosis)
python hybrid_predictor.py 51

# Results in: results/NC_000962.3_predictions.gff
```

## Performance Considerations

**Recent Optimizations (v1.1):**
- RBS detection optimized with LRU caching and sliding window algorithm
- Average 73% reduction in ORF detection time across diverse genomes
- No changes to prediction accuracy - results identical to previous version

### Memory Usage
- Small genome (1-2 Mbp): ~200-300 MB RAM
- Typical genome (4-5 Mbp): ~500 MB RAM
- Large genome (10+ Mbp): 1-2 GB RAM

### Processing Time
Per genome on typical hardware (4-core CPU):
- ORF detection: ~8-10 seconds
- Self-training (IMM, codon tables): ~1-2 minutes
- Scoring all ORFs: ~1-2 minutes
- ML inference (optional): <1 second
- **Total: 2-4 minutes per genome**

### Storage
- Each downloaded genome: 1-5 MB (FASTA + GFF)
- Prediction results: 50-500 KB per genome
- No large model files cached (only optional 5 MB ML models)
- Use cleanup to remove downloaded files

## File Cleanup

The tool can clean up generated files:

**Via Web Interface:**
- Navigate to Settings or Tools menu
- Click "Cleanup Files"
- Review files to be deleted
- Confirm deletion

**Via CLI:**
```bash
python hybrid_predictor.py
# Select [5] Cleanup Files

# This removes:
# - Downloaded FASTA files (data/full_dataset/*.fasta)
# - Downloaded GFF files (data/full_dataset/*.gff)
# - Prediction result files (results/*.gff)
# - Validation reports (results/*.txt)
```

The cleanup shows you all files before deletion and asks for confirmation.

## Requirements

### Backend (Python)
- **Python 3.7 or higher**

**Dependencies:**
```
biopython>=1.79
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
```

Install via:
```bash
pip install -r requirements.txt
```

### Frontend (Web Interface)
- **Node.js 18+ and npm**

**Technologies:**
- React 18.3
- Vite (build tool)
- Tailwind CSS
- Lucide React (icons)

Install via:
```bash
cd gene-prediction-frontend
npm install
```

### System Requirements
- 4GB+ RAM (8GB recommended for large genomes)
- ~500 MB disk space for data/results
- Internet connection (for NCBI downloads only)

## Jupyter Notebooks

The `notebooks/` directory contains research notebooks showing the development process:

### 01_data_collection.ipynb
- Downloading genomes from NCBI using BioPython
- Parsing FASTA and GFF file formats
- Extracting reference gene annotations
- Building test genome catalog

### 02_orf_detection.ipynb
- ORF detection algorithm implementation
- Traditional scoring methods (codon bias, IMM, RBS)
- Weight optimization using grid search
- Threshold tuning on validation set
- Performance analysis

### 03_ml_models.ipynb
- Feature engineering for ORF groups
- LightGBM classifier training
- Cross-validation across taxonomic groups
- Performance evaluation and metrics
- Feature importance analysis

**Note:** Notebooks are for development/research. The production tools are the web interface and `hybrid_predictor.py`.

## Methods & References

This implementation is inspired by established gene prediction tools:

**GLIMMER** (Interpolated Markov Models)
- Salzberg et al. (1998) "Microbial gene identification using interpolated Markov models" *Nucleic Acids Research*
- Delcher et al. (2007) "Identifying bacterial genes and endosymbiont DNA with Glimmer" *Bioinformatics*

**Prodigal** (Self-training and RBS detection)
- Hyatt et al. (2010) "Prodigal: prokaryotic gene recognition and translation initiation site identification" *BMC Bioinformatics*

**GeneMark** (Self-training methodology)
- Besemer & Borodovsky (1999) "Heuristic approach to deriving models for gene finding" *Nucleic Acids Research*

## Known Limitations

- **No frameshift detection**: Does not handle programmed frameshifts or ribosomal slippage
- **Not optimized for viruses**: Best suited for cellular organisms (bacteria/archaea)
- **No RNA genes**: Does not predict tRNA, rRNA, or ncRNA genes
- **Atypical genomes**: May underperform on organisms with unusual codon usage (e.g., some symbionts)
- **No gene function**: Predicts gene locations only (no functional annotation)
- **Validation limited**: Can only validate against NCBI genomes with reference annotations
- **Single contig**: Best performance on complete genomes (not optimized for draft assemblies with many contigs)

## Troubleshooting

### Web Application Issues

**Error: "Cannot connect to backend"**
```bash
# Make sure backend is running
python -m uvicorn api.main:app --reload

# Check if port 8000 is available
# Windows: netstat -ano | findstr :8000
# Linux/Mac: lsof -i :8000
```

**Error: "Frontend won't start"**
```bash
# Clear npm cache and reinstall
cd gene-prediction-frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**Error: "Port already in use"**
```bash
# Kill processes using ports 5173 and 8000
# Windows: 
#   netstat -ano | findstr :5173
#   taskkill /PID <PID> /F
# Linux/Mac:
#   kill -9 $(lsof -t -i:5173)
```

### Import Errors

**Error: "No module named 'src'"**
```bash
# Make sure you're running from the project root directory
cd bacterial-gene-prediction
python hybrid_predictor.py
```

**Error: "cannot import name 'compare_results_file_to_reference'"**
```bash
# Make sure src/comparative_analysis.py exists and contains the function
# Reinstall or check file integrity
```

### NCBI Download Issues

**Error: "NCBI email required"**
```bash
# Option 1: Set in config.py
# Edit src/config.py and set NCBI_EMAIL = "your@email.com"

# Option 2: Provide via command line
python hybrid_predictor.py NC_000913.3 --email your@email.com

# Option 3: Enter in web interface
```

**Error: "HTTP Error 429: Too Many Requests"**
```bash
# NCBI rate limits requests. Wait 1-2 minutes and try again
# Or download file manually and place in data/full_dataset/
```

### File Issues

**Error: "File not found"**
```bash
# Use absolute path
python hybrid_predictor.py /full/path/to/genome.fasta

# Or navigate to file directory first
cd /path/to/genome/
python /path/to/hybrid_predictor.py genome.fasta
```

**Error: "results/ directory not writable"**
```bash
# Check permissions
chmod 755 results/

# Or run from a directory where you have write permissions
```

### Prediction Quality Issues

**Predictions seem inaccurate (low F1 score)**
- Verify FASTA file contains a bacterial/archaeal genome
- Check if genome has unusual characteristics:
  - Very high or low GC content (<30% or >75%)
  - Unusually small (<1 Mbp) or large (>15 Mbp)
  - Draft assembly with many short contigs
- Try adjusting ML thresholds:
  ```bash
  # More lenient (may increase sensitivity)
  python hybrid_predictor.py genome.fasta --ml-threshold 0.05
  
  # More strict (may increase precision)
  python hybrid_predictor.py genome.fasta --final-ml-threshold 0.2
  ```

**ML model not found**
```bash
# The tool works without ML models (falls back to traditional scoring)
# To use ML filtering, ensure models exist:
#   - models/orf_classifier_lgb.pkl
#   - models/hybrid_best_model.pkl
# Or disable ML with --no-group-ml and --no-final-ml
```

### Validation Issues

**Error: "Cannot validate - no reference available"**
- Validation only works for NCBI genomes with reference annotations
- Custom FASTA files cannot be validated without providing a reference GFF
- Make sure genome ID matches NCBI accession format (e.g., NC_000913.3)

## Contributing

Contributions are welcome! Areas for improvement:
- Support for draft genomes with multiple contigs
- RNA gene prediction (tRNA, rRNA)
- Functional annotation integration
- Additional visualization options in web interface
- Batch processing capabilities
- Additional ML models

Please open an issue or pull request on GitHub.

## License

MIT License - see LICENSE file for details.

## Author

Roy Medina  
GitHub: [@roeimed0](https://github.com/roeimed0)

## Citation

If you use this tool in your research, please cite:

```
Medina, R. (2025). Bacterial Gene Prediction: A Hybrid ML and Traditional
Algorithm Approach. GitHub repository.
https://github.com/roeimed0/bacterial-gene-prediction
```

## Acknowledgments

- **NCBI** for genome data and Entrez API access
- **BioPython** community for excellent bioinformatics tools
- **LightGBM** developers for the ML framework
- **FastAPI** and **React** communities for web framework tools
- Authors of **GLIMMER**, **GeneMark**, and **Prodigal** for algorithmic inspiration
- Training genome data from publicly available bacterial genome projects

---

**Disclaimer:** This is a research and educational tool developed for learning purposes. For production genome annotation, consider using established tools like:
- **Prokka** - Rapid prokaryotic genome annotation
- **NCBI PGAP** - NCBI Prokaryotic Genome Annotation Pipeline
- **Bakta** - Rapid & standardized annotation toolkit
- **Prodigal** - Industry-standard gene prediction