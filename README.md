# Bacterial Gene Prediction Pipeline

[![CI](https://github.com/roeimed0/bacterial-gene-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/roeimed0/bacterial-gene-prediction/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A seven-stage hybrid pipeline for *de novo* bacterial gene prediction, combining classical bioinformatics scoring with three trained machine learning models. Achieves **F1 = 79.01%** on a 20-genome holdout spanning four phyla at **~18.8 Mbp/min** on a modern workstation with GPU.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Stage 1 — ORF Detection](#2-stage-1--orf-detection)
3. [Stage 2 — Per-Genome Scoring Models](#3-stage-2--per-genome-scoring-models)
4. [Stage 3 — Filter 1 (combined_score gate)](#4-stage-3--filter-1)
5. [Stage 4 — LGB Group Filter](#5-stage-4--lgb-group-filter)
6. [Stage 5 — Start Codon Selection](#6-stage-5--start-codon-selection)
7. [Stage 6 — Filter 2](#7-stage-6--filter-2)
8. [Stage 7 — Hybrid Filter](#8-stage-7--hybrid-filter)
9. [Feature Reference Tables](#9-feature-reference-tables)
10. [Model Files and Configuration](#10-model-files-and-configuration)
11. [Training Pipeline](#11-training-pipeline)
12. [Benchmark and Evaluation](#12-benchmark-and-evaluation)
13. [Performance Engineering](#13-performance-engineering)
14. [Known Limitations](#14-known-limitations)
15. [Quick Start](#15-quick-start)

---

## 1. Pipeline Overview

```
Genome FASTA
     │
     ▼
[Stage 1] ORF Detection              find_orfs_candidates()
     │    ATG/GTG/TTG start codons    Numba JIT scanner
     │    ≥ 100 bp ORFs               ~176K ORFs for E. coli
     ▼
[Stage 2] Per-Genome Scoring         build_all_scoring_models()
     │    Codon bias, IMM, RBS,       Self-trained on this genome
     │    length, start codon         score_all_orfs()
     ▼
[Stage 3] Filter 1                   filter_candidates()
     │    combined_score threshold    -4.0pp sensitivity loss
     ▼
[Stage 4] LGB Group Filter           OrfGroupClassifier.filter_groups()
     │    30 group-level features     -2.8pp sensitivity loss
     │    LightGBM classifier
     ▼
[Stage 5] Start Codon Selection      StartSelectionClassifier.select_best_starts()
     │    30 pairwise features        -12.7pp sensitivity loss (biggest stage)
     │    Contested pair classifier
     ▼
[Stage 6] Filter 2                   filter_candidates()
     │    Tighter combined_score      -0.2pp sensitivity loss
     ▼
[Stage 7] Hybrid Filter              HybridGeneFilter.predict()
     │    CNN (DNA sequence) +        -2.9pp sensitivity loss
     │    Dense (26 tabular feats)
     ▼
GFF3 Output
```

**Stage-by-stage sensitivity losses** (mean across 20 holdout genomes):

| Stage | Proteo | Firmicutes | Actino | Archaea | Overall |
|---|---|---|---|---|---|
| 1. ORF detection | -1.5 | -1.1 | -1.4 | -1.6 | **-1.4 pp** |
| 2. Filter 1 | -4.6 | -2.5 | -4.4 | -4.5 | **-4.0 pp** |
| 3. LGB filter | -3.0 | -1.2 | -4.6 | -2.5 | **-2.8 pp** |
| 4. **Start selection** | **-11.6** | **-6.6** | **-16.6** | **-16.1** | **-12.7 pp** |
| 5. Filter 2 | -0.1 | -0.3 | -0.1 | -0.3 | **-0.2 pp** |
| 6. Hybrid filter | -2.4 | -0.7 | -4.2 | -4.1 | **-2.9 pp** |

Start codon selection is the dominant loss. High-GC Actinobacteria and Archaea (often using leaderless transcription) are most affected.

---

## 2. Stage 1 — ORF Detection

**File:** `src/traditional_methods.py` → `find_orfs_candidates()`

### What it does
Scans both strands of the genome for every possible Open Reading Frame (ORF) — a stretch of codons beginning with a start codon and ending at the next in-frame stop codon, of at least `min_length` bases (default 100 bp / ~33 amino acids).

### Biology
Bacteria use three start codons: ATG (most common, ~80%), GTG (~15%), and TTG (~5%). All are legitimate translation initiation sites. A minimum length of 100 bp (~33 aa) filters the vast number of short random ORFs while retaining essentially all real genes (>98% of annotated bacterial genes are ≥100 bp).

### Implementation — Numba JIT scanner
The scan uses a Numba-compiled inner loop (`_scan_orfs_numba`) for speed:

1. **Codon encoding**: Each base is encoded as a 2-bit integer (A=0, C=1, G=2, T=3). A triplet `(a, b, c)` becomes the integer `a×16 + b×4 + c`. This encoding maps:
   - ATG → 0×16 + 3×4 + 2 = **14**
   - GTG → 2×16 + 3×4 + 2 = **46**  
   - TTG → 3×16 + 3×4 + 2 = **62**
   - TAA → 48, TAG → 50, TGA → 56 (stop codons)

2. **Active-starts buffer**: The scanner maintains a list of active start codon positions. Each stop codon closes all active starts, emitting one ORF record per valid start-stop pair.

3. **Dynamic buffer**: The buffer auto-resizes via `_scan_orfs_safe()` if it fills (happens for very high-GC genomes like *Pseudomonas putida* which previously suffered a silent 25% gene loss bug from buffer overflow).

4. **RBS scoring**: Immediately after detection, each ORF is scored for Shine-Dalgarno signal in the 20 bp upstream region using a Numba-batch scorer (`_score_rbs_batch`).

### Historical note
A silent Numba buffer overflow bug (wrong `max_results` formula) caused ~25% of genes to be silently dropped for high-GC Proteobacteria and Actinobacteria. The fix increased the buffer from `n//50` to a dynamic doubling scheme, recovering e.g. P.putida from 75% → 99.9% gene detection.

---

## 3. Stage 2 — Per-Genome Scoring Models

**File:** `src/traditional_methods.py` → `build_all_scoring_models()`, `score_all_orfs()`

### Why per-genome?
Each genome is scored against models built **from that genome only**. This avoids inter-species bias: a codon usage pattern that looks "gene-like" in *E. coli* may look random in *Streptomyces* (GC=71%). Self-training ensures all scores are genome-relative.

### Five scoring components

#### 1. Codon Bias Score
**Biology**: Real coding sequences exhibit non-uniform codon usage (some synonymous codons are preferred, especially at GC-rich third positions). Random intergenic sequence does not.

**Calculation**: Log-odds ratio of the ORF's codon frequencies vs. a background codon model trained on intergenic regions:
```
score = Σ log(P_coding(codon) / P_background(codon)) / n_codons
```

**Implementation**: Codon log-ratio table is precomputed (`build_codon_log_ratio_table()`), then applied via Numba JIT in batch for all ORFs simultaneously.

#### 2. IMM Score (Interpolated Markov Model)
**Biology**: Real genes have position-specific nucleotide dependencies due to codon structure. An Interpolated Markov Model of order 8 captures dependencies of up to 9 consecutive bases.

**Calculation**: Frame-aware log-likelihood ratio between a coding model (trained on likely real ORFs) and a non-coding model (trained on intergenic regions):
```
score = (1/n) × Σᵢ [log P_coding(nᵢ | context) - log P_noncoding(nᵢ | context)]
```
where context = up to 8 preceding bases in the same codon frame.

**Implementation**: Uses integer-indexed numpy tables (`build_numba_log_table()`) for O(1) lookup per base. Applied as a vectorised numpy sliding-window operation in batch.

#### 3. RBS Score (Ribosome Binding Site)
**Biology**: The Shine-Dalgarno sequence, typically 4–12 bp upstream of the start codon, base-pairs with the 3' end of 16S rRNA to position the ribosome. Strong SD signal (AGGAGG) with optimal spacing (6–8 bp) predicts real genes.

**Formula** (simplified):
```
score = spacing_score × 2.0 + motif_score × 1.5 + (purine_fraction - 0.6) × 2.0
```
where spacing_score is 3.0 for 6–8 bp, 2.5 for 5–10 bp, 1.5 otherwise.

**GC correlation**: In high-GC genomes, SD signals are naturally weaker because purines (A, G) are less common. RBS score has AUC=0.62 for FP/TP discrimination vs 0.79 for combined_score. This is a fundamental biological limitation (not a model deficiency).

#### 4. Length Score
**Biology**: Real genes are substantially longer than random ORFs. A log-normal length distribution characterises real genes.

**Calculation**: `score = log(max(length, 100) / 300)` where 300 bp is the reference length.

#### 5. Start Codon Score
**Biology**: ATG is strongly preferred (Kozak analog in bacteria); GTG and TTG are used less often and require stronger other signals.

**Values**: ATG=1.0, GTG=0.7, TTG=0.4.

### Combined Score
All five scores are z-score normalised per genome, then combined with equal weights (since they are already on the same scale after normalisation):
```
combined_score = codon_norm + imm_norm + rbs_norm + length_norm + start_norm
```

---

## 4. Stage 3 — Filter 1

**File:** `src/traditional_methods.py` → `filter_candidates()`

Removes ORFs with `combined_score < threshold` (threshold from `FIRST_FILTER_THRESHOLD` in config).

### Design choice
The original implementation used an AND condition on all individual raw scores, which caused 2–6 pp extra sensitivity loss in high-GC genomes. Because raw scores are genome-specific (a "good" IMM score in *E. coli* is numerically different from a "good" IMM score in *Streptomyces*), applying raw thresholds cross-genome is unsound. The combined_score, after z-normalisation, is genome-invariant and sufficient as a single gate.

---

## 5. Stage 4 — LGB Group Filter

**File:** `src/ml_models.py` → `OrfGroupClassifier`

### The ORF grouping problem
After Filter 1, many ORFs in the same genomic region share the same stop codon but differ in where they start — forming a "nested ORF group." Most real genes have exactly one correct start codon; the others are random ATG/GTG upstream artifacts. The LGB model classifies each **group** (not individual ORF) as "contains a real gene" vs. "entirely spurious."

### Features (30 total)
Each group feature summarises the ORF ensemble within the group:

| Category | Features |
|---|---|
| Score aggregates | `combined_mean`, `combined_max`, `combined_std` |
| Relative scores | `rel_combined_max`, `rel_rbs_max`, `rel_codon_max` |
| Score fractions | `frac_above_0.8_combined`, `frac_above_0.95_combined` |
| Length statistics | `length_max`, `length_min`, `length_ratio_max_min` |
| Group size | `n_orfs`, `entropy_combined` |
| RBS | `rbs_max`, `rbs_mean`, `rbs_dominance` (rbs_max/rbs_mean) |
| Strand | `strand_plus_frac`, `strand_minus_frac` |
| Weighted sum | `start_select_score_max` (combined via START_SELECTION_WEIGHTS) |
| GC gate | `genome_gc_high = max(0, genome_gc - 0.65)` |

**`genome_gc_high`**: At GC > 65%, codon bias and RBS signals are systematically weaker (fewer stop codons → longer ORFs; fewer A/G → weaker SD). This gated feature lets the model adjust its thresholds for high-GC genomes without conflating the genome-wide GC with any single ORF's properties. The floor of 0.65 was determined empirically — below 65% GC, the raw signals are sufficient; above, explicit adjustment is needed.

### Training
- 68 genomes from GENOME_CATALOG (4 phyla × ~17 each)
- LightGBM classifier, 30 features, threshold calibrated on held-out validation set
- **Production threshold**: 0.15 (intentionally aggressive — the Hybrid filter downstream handles precision)
- The training threshold used during downstream model data generation is 0.05 (lower, to give more candidate examples for the start selector and hybrid to learn from)

---

## 6. Stage 5 — Start Codon Selection

**File:** `src/ml_models.py` → `StartSelectionClassifier`

### Why this is the hardest stage
After the LGB filter, each kept group still contains multiple candidate start codons. Selecting the correct one is the pipeline's **dominant accuracy bottleneck** (-12.7 pp sensitivity overall). The difficulty is that competing starts often differ by only tens of base pairs and have very similar signal profiles.

### Two-stage decision
1. **Uncontested groups**: If the baseline weighted-sum score gap between the #1 and #2 candidates exceeds `contest_t = 1.0`, take the #1 without involving the ML classifier.
2. **Contested groups**: If gap < 1.0, the pair is scored by a LightGBM pairwise classifier.

### Pairwise features (30 total)
All features are computed as **differences** (top-1 minus top-2) plus absolute values for the top candidate:

| Feature | Biology | Formula |
|---|---|---|
| `d_baseline` | Baseline score gap | `score_1 - score_2` |
| `d_rbs` | RBS signal advantage | `rbs_norm_1 - rbs_norm_2` |
| `d_start` | Start codon preference | `start_norm_1 - start_norm_2` |
| `d_codon` | Codon bias advantage | `codon_norm_1 - codon_norm_2` |
| `d_imm` | Coding potential advantage | `imm_norm_1 - imm_norm_2` |
| `d_length` | Length difference (bp) | `len_1 - len_2` |
| `d_f4` | SD motif strength delta | `f4_spacer(up_1) - f4_spacer(up_2)` |
| `d_f5` | Codon-position GC bias delta | `f5_gc_bias(up_1) - f5_gc_bias(up_2)` |
| `d_up_imm` | Upstream coding potential | IMM ratio on 25bp upstream |
| `d_genome_rbs` | Per-genome RBS PWM score | Score against genome-built PWM |
| `d_anti_sd` | Anti-SD complementarity | 16S 3'-tail base-pairing score |
| `anti_sd_top1/2` | Absolute anti-SD scores | — |
| `ext_codon` | Extension region codon bias | Codon bias of region between starts |
| `d_any_stop_dist` | Nearest upstream stop delta | Closer stop → more isolated ORF |
| `d_len_zscore` | Z-scored length vs genome | Normalised length advantage |
| `d_post_start` | Post-start codon bias | Codon bias of first 5 codons |
| `d_ctx_pwm` | Context PWM score | 13bp window around start codon |
| `gap`, `rel_gap`, `score_range` | Group-level contest context | — |
| `n_near_ties`, `n_orfs`, `frac_atg` | Group statistics | — |
| `gc_pct` | Genome GC fraction | — |

### Key feature: Anti-SD complementarity (`d_anti_sd`)
**Biology**: The Shine-Dalgarno mechanism. The 3' tail of 16S rRNA has the sequence 5'-GAUCACCUCCUUA-3'. During translation initiation, this tail base-pairs antiparallel with the mRNA's upstream region. The more complementary bases (Watson-Crick: A-U, G-C; wobble: G-U), and the more optimal the spacing (4–14 bp from the SD end to the start codon), the stronger the ribosome binding.

**Implementation**: For each 25bp upstream window, slide the ANTI_SD string across all valid positions (spacer 4–14 bp from start), score each position as `matches / len(ANTI_SD)` (1.0 per Watson-Crick, 0.5 per wobble pair), return the maximum score.

**Vectorised batch**: The entire anti-SD scoring for all N contested pairs uses `_anti_sd_scores_batch()` — a numpy sliding-window operation on a `(N, 25)` byte matrix against a pre-built `(13, 5)` score lookup table. This replaces 4N Python function calls with a single numpy pass.

### Classifier output
- `flip_t = 0.80`: The classifier only changes the selection if it is **≥80% confident** the alternative is better. This conservative threshold prevents noise from degrading good baseline calls.

---

## 7. Stage 6 — Filter 2

Same mechanism as Filter 1 but with a **tighter threshold** (`SECOND_FILTER_THRESHOLD`). After start codon selection, the per-ORF combined_score has changed (the selected ORF may be shorter or longer than before), so a second pass removes newly-below-threshold candidates.

---

## 8. Stage 7 — Hybrid Filter

**File:** `src/ml_models.py` → `HybridGeneFilter`

### Architecture
A dual-branch neural network combining raw sequence content with tabular biological features:

```
DNA sequence (up to 1500 bp, one-hot encoded)
    │
    ▼
[CNN Branch]
Conv1D(4→64, k=7) → BN → MaxPool(2) →
Conv1D(64→128, k=5) → BN → MaxPool(2) →
Conv1D(128→128, k=3) → BN → GlobalMaxPool →
Dense(128→64) → Dropout(0.3)
    │
    ├──────────────────────┐
    │                      │
26 tabular features    [Dense Branch]
    │                  Dense(26→64) →
    │                  Dense(64→64) → Dropout(0.3)
    │                      │
    └──────────┬───────────┘
               ▼
         [Fusion Layer]
         Dense(128→64) → Dense(64→1) → Sigmoid
```

**Why CNN + Dense?** The CNN learns motif patterns in the raw sequence (codon periodicity, GC composition gradients, structural signals). The dense branch uses pre-computed biological features that require expert knowledge to compute (IMM, RBS, amino acid properties). Fusion lets the model weight sequence evidence against computed evidence.

### 26 tabular features

| Feature | Biology |
|---|---|
| `codon_score_norm` | Codon usage log-odds (z-normalised) |
| `imm_score_norm` | Interpolated Markov Model score (z-norm) |
| `rbs_score_norm` | Shine-Dalgarno signal strength (z-norm) |
| `length_score_norm` | log(length/300) (z-norm) |
| `start_score_norm` | Start codon preference ATG=1, GTG=0.7, TTG=0.4 (z-norm) |
| `combined_score` | Weighted sum of all normalised scores |
| `length_bp` | Raw gene length in base pairs |
| `length_log` | log(length), natural scale for neural net |
| `start_codon_type` | Categorical: ATG=0, GTG=1, TTG=2 |
| `stop_codon_type` | Categorical: TAA=0, TAG=1, TGA=2 |
| `gc_content` | GC fraction of the ORF |
| `gc_deviation` | ORF GC minus genome GC (gene-specific bias) |
| `gc3_content` | GC at 3rd codon position (wobble position) |
| `gc_skew` | (G-C)/(G+C) — strand usage asymmetry |
| `at_skew` | (A-T)/(A+T) — strand usage asymmetry |
| `purine_content` | (A+G)/length — related to RBS and coding |
| `effective_num_codons` | ENC statistic: 20 (strong bias) to 61 (no bias) |
| `codon_bias_index` | CBI: fraction preferred codons |
| `has_hairpin_near_stop` | Stem-loop near stop (transcription termination signal) |
| `minus10_box_score` | Prokaryotic promoter -10 box (TATAAT) score |
| `hydrophobicity_mean` | Mean Kyte-Doolittle hydrophobicity of translated protein |
| `hydrophobicity_std` | Std of hydrophobicity (structural variation) |
| `charge_mean` | Mean charge of amino acids (electrostatics) |
| `aromatic_fraction` | Fraction of F/Y/W residues |
| `small_fraction` | Fraction of G/A/S/T residues |
| `polar_fraction` | Fraction of Q/N/H/S/T/Y/C residues |

### Training
The hybrid is trained on pipeline output from **LGB-filtered + ML start-selected** candidates, labelled against reference GFF annotations. This ensures the model sees exactly the input distribution it will encounter at inference time. Training on rule-based start selection output (the old approach) created a distribution mismatch that cost ~0.3 pp F1.

**GPU acceleration**: Inference uses CUDA (RTX 4070 Ti in our setup) with batch_size=512. The GPU gives 3.6× speedup over CPU inference.

---

## 9. Feature Reference Tables

### Start Selector Pairwise Features (complete)

| Feature | Type | Biology | Range |
|---|---|---|---|
| `d_baseline` | float | Baseline gap between candidates | (-∞, +∞) |
| `d_rbs` | float | Δ RBS signal strength | (-∞, +∞) |
| `d_start` | float | Δ start codon preference | ~(-1, +1) |
| `d_codon` | float | Δ codon bias score | ~(-3, +3) |
| `d_imm` | float | Δ IMM coding potential | ~(-3, +3) |
| `d_length` | float | Δ length (bp) | (-1000, +1000) |
| `d_f4` | float | Δ SD motif spacer score (0–7 nt motif) | (-1, +1) |
| `d_f5` | float | Δ codon-pos GC asymmetry \|GC3−(GC1+GC2)/2\| | (-0.5, +0.5) |
| `d_up_imm` | float | Δ upstream region IMM score | (-∞, +∞) |
| `d_genome_rbs` | float | Δ per-genome RBS PWM score | (-∞, +∞) |
| `d_anti_sd` | float | Δ 16S anti-SD complementarity | (-1, +1) |
| `anti_sd_top1` | float | Absolute anti-SD score, candidate 1 | (0, 1) |
| `anti_sd_top2` | float | Absolute anti-SD score, candidate 2 | (0, 1) |
| `ext_codon` | float | Codon bias in region between starts | ~(-3, +3) |
| `d_any_stop_dist` | int | Δ distance to nearest upstream stop (bp) | (-300, +300) |
| `any_stop_top1` | int | Distance to nearest upstream stop, cand. 1 | (0, 300) |
| `any_stop_top2` | int | Distance to nearest upstream stop, cand. 2 | (0, 300) |
| `d_len_zscore` | float | Δ z-scored length vs genome | ~(-3, +3) |
| `len_zscore_top1` | float | Z-scored length, candidate 1 | ~(-3, +3) |
| `len_zscore_top2` | float | Z-scored length, candidate 2 | ~(-3, +3) |
| `d_post_start` | float | Δ first-5-codon codon bias | ~(-3, +3) |
| `post_start_top1` | float | Post-start codon bias, candidate 1 | ~(-3, +3) |
| `post_start_top2` | float | Post-start codon bias, candidate 2 | ~(-3, +3) |
| `d_ctx_pwm` | float | Δ 13bp context PWM score | ~(-5, +5) |
| `ctx_pwm_top1` | float | Context PWM score, candidate 1 | ~(-5, +5) |
| `ctx_pwm_top2` | float | Context PWM score, candidate 2 | ~(-5, +5) |
| `gap` | float | Baseline gap (same as d_baseline) | — |
| `score_range` | float | Range of group scores | (0, +∞) |
| `rel_gap` | float | gap / score_range | (0, 1) |
| `n_near_ties` | int | Candidates within 0.5 of top-1 | (0, N) |
| `n_orfs` | int | Total ORFs in group | (1, ~50) |
| `top1_len_rank` | float | Rank of top-1 by length | (0, 1) |
| `group_len_cv` | float | Coefficient of variation of lengths | (0, +∞) |
| `frac_longer` | float | Fraction of group longer than top-1 | (0, 1) |
| `grp_rbs_mean` | float | Mean RBS score across group | ~(-5, +5) |
| `grp_rbs_range` | float | RBS score range across group | (0, +∞) |
| `frac_atg` | float | Fraction of group using ATG | (0, 1) |
| `top1_rbs_rank` | float | Rank of top-1 by RBS score | (0, 1) |
| `gc_pct` | float | Genome GC fraction | (0, 1) |
| `both_atg` | int | Both candidates use ATG | 0 or 1 |

---

## 10. Model Files and Configuration

### `models/` directory

| File | Contents |
|---|---|
| `orf_classifier_lgb.pkl` | LightGBM group classifier (joblib format) |
| `orf_classifier_lgb_feature_names.pkl` | Feature name list for LGB |
| `orf_classifier_lgb_meta.pkl` | `{"gc_floor": 0.65}` — GC gate floor |
| `start_selector.pkl` | StartSelectionClassifier bundle (pickle) |
| `hybrid_best_model.pkl` | HybridGeneFilter weights + threshold (pickle) |
| `thresholds.json` | Production thresholds for LGB and Hybrid |

### `models/thresholds.json`
```json
{
  "orf_classifier_lgb": {
    "threshold": 0.15,
    "note": "Calibrated on 20-genome holdout. Higher threshold = more precision, less recall."
  },
  "hybrid_best_model": {
    "threshold": 0.398,
    "note": "Calibrated threshold (plateau 0.35–0.42). Val_f1=0.9117 (v4)."
  }
}
```

### `src/config.py` — key constants
| Constant | Value | Meaning |
|---|---|---|
| `MIN_ORF_LENGTH` | 100 | Minimum ORF length (bp) scanned |
| `RBS_UPSTREAM_LENGTH` | 20 | Window for RBS detection (bp) |
| `LGB_GC_FLOOR_DEFAULT` | 0.55 | Legacy fallback for old models without meta |
| `LGB_TRAINING_THRESHOLD` | 0.05 | LGB threshold used during training data generation (lower than production to give more candidates) |
| `HF_MAX_SEQ_LEN` | 1500 | Max sequence length input to Hybrid CNN |
| `HF_MAX_EPOCHS` | 50 | Hybrid training epoch ceiling |
| `HF_EARLY_STOP_PATIENCE` | 10 | Epochs without improvement before stopping |
| `CODON_INT_ATG` | 14 | Base-4 encoding: A×16+T×4+G = 0×16+3×4+2 |

---

## 11. Training Pipeline

### Correct training order

The three models must be trained **sequentially** because each downstream model depends on the previous one's output distribution:

```
Step 1: Train LGB
   python scripts/training/train_lgb.py --seed 42 [--gc-floor 0.65]
   → Saves models/orf_classifier_lgb_v2.pkl

Step 2: Train Start Selector  (uses new LGB)
   python scripts/training/train_start_classifier.py \
       --lgb-path models/orf_classifier_lgb_v2.pkl \
       --out-model models/start_selector_v2.pkl --seed 42
   → Saves models/start_selector_v2.pkl

Step 3: Train Hybrid  (uses new LGB + new Start Selector)
   python scripts/training/train_hybrid.py \
       --lgb-path models/orf_classifier_lgb_v2.pkl \
       --start-selector models/start_selector_v2.pkl \
       --seed 42
   → Saves models/hybrid_best_model_v4.pkl
```

**Why this order matters**: The Hybrid is trained on the output of the ML Start Selector (not the simpler rule-based version). Training the Hybrid on rule-based start selection creates a distribution mismatch because the Hybrid sees different candidates at inference than it was trained on. This mismatch costs ~0.3 pp F1.

### Key training parameters
| Script | Key flag | Default | Notes |
|---|---|---|---|
| `train_lgb.py` | `--gc-floor` | 0.55 | Set to 0.65 for production models |
| `train_lgb.py` | `--seed` | system entropy | Set for reproducibility |
| `train_start_classifier.py` | `--lgb-path` | production model | Always use new LGB for clean training |
| `train_hybrid.py` | `--start-selector` | rule-based | Always pass ML start selector path |
| `train_hybrid.py` | `--epochs` | 50 | Usually stops early (~25–35 epochs) |

### Benchmarking after training
```bash
# After training, always run benchmark before promoting
python scripts/evaluation/benchmark.py \
    --lgb-path models/orf_classifier_lgb_v2.pkl \
    --hf-path models/hybrid_best_model_v4.pkl \
    --start-selector models/start_selector_v2.pkl \
    --lgb-threshold 0.15 --hf-threshold 0.398 \
    --save "description-of-change"

# Only promote to production if F1 improves and no regressions
```

---

## 12. Benchmark and Evaluation

### Holdout set
20 genomes from `TEST_GENOMES` in `src/config.py` — never used in any training or validation run. 5 genomes per taxonomic group (Proteobacteria, Firmicutes, Actinobacteria, Archaea), spanning GC 31–71%.

### Current performance

| Group | F1 | Sensitivity | Precision |
|---|---|---|---|
| Actinobacteria | 73.2% | 68.8% | 78.5% |
| Archaea | 74.2% | 72.0% | 76.5% |
| Firmicutes | 88.5% | 87.2% | 89.8% |
| Proteobacteria | 80.2% | 77.4% | 83.1% |
| **Overall** | **79.01%** | **76.41%** | **81.91%** |

### Running the benchmark
```bash
python scripts/evaluation/benchmark.py --save "my-experiment"
```
Results are appended to `experiments/log.json`. Compare with:
```bash
python -c "
import json
log = json.load(open('experiments/log.json'))
for e in log[-5:]:
    print(f\"[{log.index(e)+1}] {e['description'][:40]:40} F1={e['overall']['f1']:.2f}\")
"
```

---

## 13. Performance Engineering

### Vectorised batch operations

The most important performance work was replacing Python loops with numpy/Numba batch operations. The key insight: numpy operations call optimised C/BLAS code with **zero Python overhead per element**, while Python loops have ~100–1000ns overhead per iteration regardless of element complexity.

**Example: Anti-SD batch scoring**

*Before (per-pair Python loop)*:
```python
for gid, grp_df in groups.items():
    up1 = self._get_upstream(seq, t1d, 25)
    score = self._anti_sd_score(up1)   # Python function, nested loops
    # ... called ~21,476 times for E. coli
```

*After (numpy batch)*:
```python
asd_1 = self._anti_sd_scores_batch(up1)  # One numpy call, all N pairs
```
Inside `_anti_sd_scores_batch`, all N upstream sequences are encoded into a single `(N, 25)` integer matrix. The ANTI_SD score table `(13, 5)` is indexed via numpy advanced indexing: `score_table[position_indices, window_characters]`. The entire computation for 5,000 pairs runs in one numpy operation.

**Speedup achieved**: 4 Python calls per pair → 1 numpy pass. E. coli: 9.6s → 4.2s (2.3×).

**Example: Stop-codon distance batch**

*Before*: Scan 300 bp upstream in 3 frames per ORF — Python loops, ~4 calls per pair.

*After* (`_dist_any_stop_batch`): All N upstream regions stacked into `(N, 300)` uint8 matrix. Three numpy matrix operations (one per reading frame) check all N ORFs simultaneously for TAA/TAG/TGA via bitwise comparisons: `(c1==T) & (c2==A) & (c3==A)`.

The vectorisation trick for finding the **last** stop codon (closest to start): instead of `argmax` (finds first True), multiply `is_stop * (codon_positions + 1)` and take `argmax` — returns index of the highest-position stop codon.

### GPU acceleration (HybridGeneFilter)
The Hybrid's CNN+Dense model runs on CUDA when available:
- Model loaded directly to GPU (`model.to(device)` in `load()`)
- Batch size 512 processes all genes in 2–5 batches per genome
- No per-batch `empty_cache()` — this forces CPU-GPU sync every batch (3.6× overhead)
- Single `empty_cache()` after all batches

### Pipeline throughput
| Stage | Time (E. coli 4.6 Mbp) | % of total |
|---|---|---|
| Start selection | 4.4s | 30% |
| Hybrid (GPU) | 3.3s | 22% |
| ORF detection | 1.6s | 11% |
| Scoring models build | 1.2s | 8% |
| Group organisation | 1.1s | 7% |
| LGB filter | 1.1s | 7% |
| **Total** | **~14.8s** | **18.8 Mbp/min** |

---

## 14. Known Limitations

### *Bordetella pertussis* FP problem
The new pipeline (post-ORF-buffer-fix retrain) introduces ~110 new false positives for *B. pertussis* (GC=67.7%), causing -0.83 pp F1 regression. Root cause: the Hybrid was trained on a dataset that now includes many more background ORFs from high-GC Proteobacteria (previously missing due to the buffer overflow bug). These background ORFs have high codon bias (expected in high-GC sequence) and the Hybrid cannot distinguish them from real genes. The 211 hard FPs have Hybrid probability 0.7–0.9 — they are not outliers in the feature distribution. Fix requires targeted negative training examples for high-GC Proteobacteria.

### Archaea — leaderless transcription
~40–50% of archaeal genes use leaderless transcription (no Shine-Dalgarno). The pipeline's RBS scoring and Anti-SD features are designed for SD-based genes. For leaderless genes, these features are near-zero for both the correct and incorrect start candidates, giving the classifier no signal. This is a biological limitation, not an algorithmic deficiency. Current F1 for Archaea is 62–77%.

### High-GC genomes (>65% GC)
The RBS signal is genuinely weaker in high-GC organisms (lower purine content → shorter/weaker SD). Start selection loses -16.6 pp for Actinobacteria vs -6.6 pp for Firmicutes. This is a fundamental biology issue partially mitigated by the `genome_gc_high` gating feature.

### Short genes (<150 bp)
The `min_length=100` cutoff misses genuinely short genes. A very small fraction of bacterial genes are <100 bp. Lowering the cutoff dramatically increases the search space (more random ORFs per genome).

---

## 15. Quick Start

### Installation
```bash
conda create -n gene_prediction python=3.9
conda activate gene_prediction
pip install -r requirements.txt
```

### Predict genes in a genome
```bash
python hybrid_predictor.py path/to/genome.fasta --output predictions.gff
```

### Batch prediction
```bash
python scripts/training/predict_batch.py \
    --output-dir results/ \
    genome1.fasta genome2.fasta genome3.fasta
```

### Validate against reference annotations
```bash
python scripts/evaluation/benchmark.py \
    --save "my-genome-validation"
```

### Directory structure
```
src/
  ml_models.py          — OrfGroupClassifier, StartSelectionClassifier, HybridGeneFilter
  traditional_methods.py — ORF detection, scoring, filtering (Numba/numpy optimised)
  config.py              — All constants and hyperparameters
  data_management.py     — Genome loading, GFF parsing, caching
models/
  orf_classifier_lgb.pkl     — Production LGB model
  start_selector.pkl         — Production start selector
  hybrid_best_model.pkl      — Production hybrid model
  thresholds.json            — Production thresholds
scripts/
  training/                  — train_lgb.py, train_hybrid.py, train_start_classifier.py
  evaluation/                — benchmark.py, benchmark_start_classifier.py
  experiments/               — Analysis and profiling scripts
experiments/log.json         — Benchmark history (all runs)
tests/                       — 467 unit and integration tests
```
