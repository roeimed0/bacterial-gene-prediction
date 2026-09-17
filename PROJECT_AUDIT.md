# Project Audit & Improvement Roadmap

**Date:** 2026-09-17  
**Branch audited:** `fix/filter1-combined-only` (9 commits ahead of `main`)  
**Benchmark baseline:** F1=79.01%, Sens=76.41%, Prec=81.91% (20-genome holdout, pseudogene-excluded)  
**Audited dimensions:** Code organization, ML science, Performance engineering

---

## Branch & Git Status

| Item | Status |
|---|---|
| Current branch | `fix/filter1-combined-only` |
| Commits ahead of `main` | 9 — all benchmarked and production-ready |
| `main` / `origin/main` | Identical at `b000339` — 9 commits behind |
| Working tree | Clean (models restored to HEAD) |
| Production models | `orf_classifier_lgb.pkl` (LGB, 30 feats, gc_floor=0.65, t=0.15), `hybrid_best_model.pkl` (CNN+Dense, 26 feats, t=0.398), `start_selector.pkl` (pairwise LGB, flip_t=0.75) |

**Recommended next action:** Open a PR from `fix/filter1-combined-only` → `main` to land all 9 commits. Then create new feature branches off `main` for the improvements below.

---

## Issue Registry

Each issue has a severity rating, effort estimate, and a concrete fix description suitable for converting directly into a GitHub issue.

**Severity scale:** `CRITICAL` (correctness/data loss) | `HIGH` (significant quality impact) | `MEDIUM` (moderate impact or maintainability) | `LOW` (polish/nice-to-have)  
**Effort scale:** `S` (<1 day) | `M` (1-3 days) | `L` (3-7 days) | `XL` (>1 week)

---

### CATEGORY A — ML Science Correctness

---

#### A1 — ENC implementation is scientifically incorrect; CBI renamed ✓

**Severity:** HIGH (was CRITICAL — CBI rename resolved the misrepresentation issue)  
**Effort:** M  
**Files:** `src/ml_models.py` — `_calculate_enc()` (line ~830)

**Resolved (2026-09-17):**  
`_calculate_cbi()` renamed to `_calculate_codon_entropy()` and feature key renamed `codon_bias_index` → `codon_entropy` throughout. The Shannon entropy calculation is a valid de novo feature; it was only the name that was wrong. CAI (Sharp & Li 1987) was evaluated but rejected — CAI requires a reference set of known highly-expressed genes, creating circularity in de novo prediction. No retraining needed (pure rename, same calculation, same feature position in model).

**Remaining problem:**  
`_calculate_enc()` does not compute Wright (1990) Effective Number of Codons. It computes a simple codon diversity ratio (unique codons / 61). ENC' (Novembre 2002, *Genetics* 163:2097) is preferred over Wright ENC for this project because it corrects for GC background — critical given training genomes span 30–70% GC. ENC' works fully de novo (no reference set needed).

**Fix:**  
1. Implement ENC' (Novembre 2002): compute per-degeneracy-class homozygosity F adjusted for expected GC-based frequencies, apply `ENC' = 2 + 9/F2' + 1/F3' + 5/F4' + 3/F6'`.
2. Update feature name `effective_num_codons` → `enc_prime` and README.
3. Re-run benchmark after fix (no retraining required if feature name in model artifact is updated).
4. **Requires retraining HybridGeneFilter** if ENC' values differ substantially from the current ratio (they will — current range is [0,1], ENC' range is [20,61]).

---

#### A2 — Training data is dominated by near-clonal strains

**Severity:** HIGH  
**Effort:** L  
**Files:** `src/config.py` — `GENOME_CATALOG`, `scripts/training/train_lgb.py`, `scripts/training/train_hybrid.py`

**Problem:**  
~17 of 25 Proteobacteria training genomes are E. coli / Shigella strains. ~10 of 25 Actinobacteria training genomes are M. tuberculosis strains. Near-clonal replicates inflate apparent training diversity without providing generalization signal. This is the most likely root cause of the Actinobacteria F1 gap (73.2% vs 88.5% for Firmicutes) and the -12.7pp sensitivity loss on problem genomes identified in the stage-by-stage analysis.

**Fix:**  
1. Audit current training genomes by genus (not species) in `GENOME_CATALOG`. Target max 2 strains per genus.
2. Replace ~12 E. coli / Shigella entries with diverse Proteobacteria genera: Pseudomonas aeruginosa, Caulobacter crescentus, Helicobacter pylori, Vibrio cholerae, Campylobacter jejuni, Burkholderia cepacia.
3. Replace ~8 M. tuberculosis entries with diverse Actinobacteria genera: Streptomyces coelicolor, Corynebacterium glutamicum, Nocardia farcinica, Bifidobacterium longum.
4. No code changes required — only genome catalog entries + retrain all three models.
5. Full 41-genome benchmark required before PR (rule: no PR before benchmark).

---

#### A3 — `gap` and `d_baseline` are duplicate features in StartSelectionClassifier

**Severity:** MEDIUM  
**Effort:** S  
**Files:** `src/ml_models.py` — `_build_feature_row()` in `StartSelectionClassifier`

**Problem:**  
Both `gap` (position difference between start candidates) and `d_baseline` (baseline distance measure) encode the same underlying quantity (inter-candidate distance). Including both inflates feature importance for distance, may mislead SHAP analysis, and wastes a model degree of freedom.

**Fix:**  
1. Run SHAP analysis on `start_selector.pkl` to confirm which of the two has higher importance.
2. Remove the lower-importance duplicate from `_build_feature_row()`.
3. Retrain `start_selector.pkl` with the reduced feature set.
4. Full benchmark required (rule: no PR before benchmark).

---

#### A4 — LGB threshold inconsistency between MODEL_LOG and thresholds.json

**Severity:** MEDIUM  
**Effort:** S  
**Files:** `MODEL_LOG.md`, `models/thresholds.json`

**Problem:**  
`MODEL_LOG.md` records LGB threshold=0.07 for the current production model. `models/thresholds.json` says 0.15. The benchmark note in `thresholds.json` explicitly confirms 0.15 gave the best F1=79.01%. The MODEL_LOG entry is stale/wrong and could mislead anyone tuning thresholds.

Also: the hybrid model entry in MODEL_LOG says threshold=0.25 but production is 0.398.

**Fix:**  
1. Update MODEL_LOG entry for `orf_classifier_lgb` v3: set threshold=0.15, F1=79.01%.
2. Update MODEL_LOG entry for `hybrid_best_model` v4: set threshold=0.398, note 26 features.
3. No code changes, no benchmark needed.

---

#### A5 — No comparison to Prodigal baseline

**Severity:** MEDIUM  
**Effort:** M  
**Files:** `scripts/evaluation/benchmark.py`, `README.md`

**Problem:**  
The 79.01% F1 figure has no scientific context. Without a Prodigal baseline on the same 20-genome holdout, it is impossible to know whether this model outperforms, matches, or underperforms the state-of-the-art tool. A peer reviewer would immediately ask for this comparison.

**Fix:**  
1. Add `--baseline prodigal` flag to `benchmark.py` that runs Prodigal on each FASTA and converts GFF output to the same coordinate format.
2. Run comparison on the 20-genome holdout. Report per-genome delta (F1, sensitivity, precision).
3. Add results table to README.

---

#### A6 — Start selector is architecturally capped at top-2 candidates

**Severity:** MEDIUM  
**Effort:** L  
**Files:** `src/ml_models.py` — `StartSelectionClassifier.select_best_starts()`

**Problem:**  
The pairwise design only contests top-1 vs top-2 candidates per group. Any gene whose true start is ranked 3rd or lower is unrecoverable by design. The stage-by-stage bottleneck analysis confirmed start selection causes -9 to -10pp sensitivity loss. A ranked-list approach (score all N candidates, pick argmax) would eliminate this architectural ceiling.

**Fix:**  
1. Change `select_best_starts` to score all candidates in each group individually (not pairwise).
2. Return the candidate with highest `start_select_score` directly (no flip logic needed).
3. Retrain `start_selector.pkl` with single-candidate feature vectors instead of difference vectors.
4. Full 41-genome benchmark required.

---

### CATEGORY B — Performance Engineering

---

#### B1 — `_one_hot_encode_dna` uses Python inner loop (7.5M iterations for E. coli)

**Severity:** HIGH  
**Effort:** S  
**Files:** `src/ml_models.py` — `HybridGeneFilter._one_hot_encode_dna()` (line ~1082)

**Problem:**  
The inner `for i, nt in enumerate(seq)` Python loop runs `num_candidates × max_len = 5000 × 1500 = 7.5M` iterations per genome. Estimated cost: 0.6–0.9s per genome, ~20% of total hybrid inference time. The pattern to fix this already exists in the same file (`_seq_to_int_fast`).

**Fix:**  
```python
# Replace the inner loop with:
_ONEHOT_TABLE = np.zeros((256, 4), dtype=np.float32)
for _i, _b in enumerate(b"ACGT"):
    _ONEHOT_TABLE[_b, _i] = 1.0

def _one_hot_encode_dna(self, sequences, max_len):
    N = len(sequences)
    result = np.zeros((N, max_len, 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        b = seq.upper().encode("ascii")[:max_len]
        result[i, :len(b)] = _ONEHOT_TABLE[np.frombuffer(b, dtype=np.uint8)]
    return result
```

Additionally, pre-encode all candidates once before batching (move outside the batch loop), matching what the training path already does at line 677. Switch array layout to `(N, 4, L)` channels-first to eliminate the `permute(0,2,1)` non-contiguous copy in `CNNBranch.forward` on every GPU pass.

---

#### B2 — Per-candidate BioPython `ProteinAnalysis` in HybridGeneFilter feature extraction

**Severity:** HIGH  
**Effort:** M  
**Files:** `src/ml_models.py` — `HybridGeneFilter._calculate_amino_acid_properties()` (line ~884), `extract_features()` (line ~1000)

**Problem:**  
`extract_features()` calls `Seq(sequence).translate(table=11)` and constructs `ProteinAnalysis(protein)` for every candidate in a Python loop. For 5000 E. coli candidates this is 5000 BioPython object instantiations. Estimated cost: 0.5–1.0s per genome (~15–30% of hybrid inference time). The GPU sits idle during this CPU bottleneck.

**Fix:**  
1. Build a 64-entry numpy array mapping codon index → amino acid index (same integer-table pattern as `_score_codon_bias_batch`).
2. Build per-property lookup arrays: `hydrophobicity[aa_idx]`, `charge[aa_idx]`, `aromaticity[aa_idx]` etc. — indexed by AA number, not character.
3. Translate each candidate via fancy indexing on the codon table (vectorized), then aggregate properties with numpy.sum / numpy.mean.
4. Eliminates all BioPython imports from the hot path.

---

#### B3 — `_upstream_imm` uses Python dict lru_cache instead of pre-built Numba integer table

**Severity:** HIGH  
**Effort:** S  
**Files:** `src/ml_models.py` — `StartSelectionClassifier._upstream_imm()` (line ~1760)

**Problem:**  
`_upstream_imm()` calls `score_imm_ratio()` → `get_interpolated_probability()` which is an `lru_cache` over Python dicts. The Numba integer table (`scoring_models["numba_coding_table"]`) is already built and passed into `select_best_starts()` — it is 10–20× faster than the Python dict path. For 5000 contested pairs × 25-base upstream windows: 125,000 Python dict lookups that should be Numba integer table lookups. Estimated cost: 0.3–0.4s per genome (~8% of start selection time).

**Fix:**  
Pass the upstream sequences as a batch to `_score_imm_batch` using the pre-built integer table already available in `scoring_models`. The batch operation replaces the per-pair Python call with one Numba dispatch.

---

#### B4 — Three separate O(G) passes over `groups` dict in `select_best_starts`

**Severity:** MEDIUM  
**Effort:** S  
**Files:** `src/ml_models.py` — `StartSelectionClassifier.select_best_starts()` (line ~1375)

**Problem:**  
`_build_rbs_pwm()`, `_build_ctx_pwm()`, and `_build_len_prior()` are called sequentially and each walks the entire groups dict independently. Three O(G) passes can be one. Additionally, per-group `sort_values(ascending=False)` inside the contested-group loop is a pandas sort per group; a single `groupby.rank` on the flat DataFrame would replace all per-group sorts.

**Fix:**  
Merge the three build functions into one `_build_priors(groups)` that returns all three in a single O(G) sweep. Replace per-group `sort_values` with `df.groupby("group_id")["_base"].rank(ascending=False, method="first")`.

---

#### B5 — `bi` dict rebuilt on every `_score_pwm` and `_score_ctx_pwm` call

**Severity:** LOW  
**Effort:** S  
**Files:** `src/ml_models.py` — `StartSelectionClassifier._score_pwm()` (line ~1636), `_score_ctx_pwm()` (line ~1757)

**Problem:**  
Both methods build `bi = {b: i for i, b in enumerate(self._BASES)}` on every call. For 5000 contested pairs in E. coli, this creates 10,000 dict objects. Zero ML cost to fix.

**Fix:**  
Add `_BASES_IDX: ClassVar[Dict[str, int]] = {b: i for i, b in enumerate(_BASES)}` as a class constant. Reference `self._BASES_IDX` in both methods.

---

#### B6 — `CNNBranch.forward` creates non-contiguous GPU tensor on every forward pass

**Severity:** LOW  
**Effort:** S  
**Files:** `src/ml_models.py` — `CNNBranch.forward()` (line ~503)

**Problem:**  
`x.permute(0, 2, 1)` produces a non-contiguous tensor. PyTorch inserts an implicit `.contiguous()` copy before `Conv1d`. Storing one-hot in `(N, 4, L)` channels-first layout (which fix B1 enables) would eliminate this copy entirely.

**Fix:**  
Change one-hot encoding to output `(N, 4, L)` directly (done in B1), then remove the `permute` from `CNNBranch.forward`. No change to model weights needed.

---

#### B7 — README CNNBranch architecture description doesn't match code

**Severity:** LOW  
**Effort:** S  
**Files:** `README.md` Section 8, `src/ml_models.py` `CNNBranch` (line ~494)

**Problem:**  
README says `Conv1D(128→128, k=3)` then `Dense(128→64)`. Code has `Conv1d(128, 256, 3)` — third conv doubles to 256 channels, then `nn.Linear(256→128)`. DenseBranch has three layers (26→64→128→128), not two.

**Fix:**  
Update README Section 8 architecture description to match code exactly.

---

### CATEGORY C — Code Organization & Software Engineering

---

#### C1 — `traditional_methods.py` is 2300+ lines mixing 5 unrelated concerns

**Severity:** HIGH  
**Effort:** L  
**Files:** `src/traditional_methods.py`

**Problem:**  
The file combines: (1) ORF scanning (Numba JIT, `_scan_orfs_numba`), (2) scoring models (IMM, codon bias, RBS — both Numba and Python fallback), (3) training set construction (`build_training_set`, IMM model training), (4) pipeline orchestration (`predict_genes`, `score_all_orfs`, `filter_candidates`), and (5) self-training utilities (`build_all_scoring_models`, `_select_imm_order`). This is the single biggest readability and maintainability problem in the project.

**Fix (mechanical refactor — no logic changes):**  
1. `src/orf_detection.py` — `_scan_orfs_numba`, `scan_all_orfs`, `_extract_upstream_windows`, `find_orfs`
2. `src/scoring_models.py` — all Numba/Python fallback scoring functions, `build_all_scoring_models`, `_select_imm_order`, `get_interpolated_probability`
3. `src/training.py` — `build_training_set`, IMM training, `_count_imm_kmers`
4. `src/traditional_methods.py` retains only `predict_genes`, `score_all_orfs`, `filter_candidates`, `select_best_starts` (the pipeline-level orchestration)
5. Update all imports in `ml_models.py`, `api.py`, test files.

---

#### C2 — `select_best_starts` exists in both `traditional_methods.py` and `ml_models.py` with different signatures

**Severity:** HIGH  
**Effort:** S  
**Files:** `src/traditional_methods.py`, `src/ml_models.py`

**Problem:**  
The name collision is a genuine maintenance trap. Callers calling the wrong version silently get different behavior. The rule-based version and the ML version have different signatures, different arguments, and different semantics — but identical function names.

**Fix:**  
Rename `traditional_methods.select_best_starts` → `rule_based_select_starts`. Update all callers. The ML version in `StartSelectionClassifier.select_best_starts` stays as-is (it's a method, so scope is clear).

---

#### C3 — RBS formula constants duplicated in 3 places

**Severity:** MEDIUM  
**Effort:** S  
**Files:** `src/traditional_methods.py` (`_score_rbs_batch`, Python fallback), `src/config.py`

**Problem:**  
Spacing penalties, motif weights, and score bounds for RBS scoring appear independently in the Numba batch function, the Python fallback, and `predict_rbs_simple`. If one is updated, the others must be manually kept in sync — a known source of silent drift.

**Fix:**  
Define named constants in `config.py` (`RBS_SPACING_OPTIMAL`, `RBS_WEIGHT_SPACING`, `RBS_MOTIFS`, etc.). Reference them in all three places. Numba functions can use module-level constants without issue.

---

#### C4 — FastAPI/uvicorn/pydantic in main `requirements.txt`

**Severity:** MEDIUM  
**Effort:** S  
**Files:** `requirements.txt`

**Problem:**  
FastAPI, uvicorn, pydantic, and httpx are API-serving dependencies. They have no role in the gene prediction pipeline and inflate the environment for users who only need the predictor. `httpx` appears twice.

**Fix:**  
1. Create `requirements-api.txt` with FastAPI, uvicorn, pydantic, httpx.
2. Remove them from `requirements.txt`.
3. Update `CONTRIBUTING.md` and `README.md` to document the split.

---

#### C5 — `GENOME_CATALOG` dict is 600+ lines of literals in `config.py`

**Severity:** MEDIUM  
**Effort:** S  
**Files:** `src/config.py`

**Problem:**  
`config.py` is supposed to hold numeric constants and thresholds. Embedding 600 lines of genome catalog entries (accession numbers, taxonomy, file paths) turns a constants file into a data file. Adding new genomes requires editing Python source.

**Fix:**  
1. Extract `GENOME_CATALOG` to `data/genome_catalog.json`.
2. Add a thin loader `load_genome_catalog() -> Dict` that reads JSON once.
3. Consumers call `load_genome_catalog()` instead of importing `GENOME_CATALOG` directly.
4. `config.py` drops to ~180 lines of actual numeric constants.

---

#### C6 — `get_genome_by_id()` returns `None` silently instead of raising

**Severity:** LOW  
**Effort:** S  
**Files:** `src/config.py` — `get_genome_by_id()`

**Problem:**  
When called with an unknown genome ID, the function returns `None`. Every caller must check for `None` explicitly or risks an AttributeError on the return value at an unrelated line. This pattern propagates the error far from its source.

**Fix:**  
```python
def get_genome_by_id(genome_id: str) -> Dict[str, Any]:
    if genome_id not in GENOME_CATALOG:
        raise KeyError(f"Unknown genome ID: {genome_id!r}")
    return GENOME_CATALOG[genome_id]
```
Update all callers that currently guard against `None`.

---

#### C7 — CI integration tests silently skipped when FASTA files are missing

**Severity:** MEDIUM  
**Effort:** S  
**Files:** `tests/` — integration test files

**Problem:**  
Integration tests that require genome FASTA files are silently skipped if the files are absent (e.g., in CI). This means CI can pass on a broken integration without any warning. There is also no test validating that Numba and Python fallback paths produce numerically equivalent results.

**Fix:**  
1. Change silent skip to `pytest.skip(reason="FASTA missing — set GENOME_DIR env var to run integration tests")` so the skip is visible in CI output.
2. Add a `test_numba_python_equivalence.py` that builds IMM/codon tables via both paths and asserts `np.allclose(numba_output, python_output, atol=1e-6)`.

---

#### C8 — `pyproject.toml` missing; `sys.path.insert` hacks in scripts

**Severity:** LOW  
**Effort:** S  
**Files:** `scripts/` (multiple), root

**Problem:**  
Absence of `pyproject.toml` forces every script to `sys.path.insert(0, ...)` to find `src/`. This is fragile and will break if a script is run from any directory other than the project root.

**Fix:**  
1. Add `pyproject.toml` with `[tool.setuptools.packages.find]` pointing at `src/`.
2. Add `pip install -e .` to `CONTRIBUTING.md` setup instructions.
3. Remove all `sys.path.insert` calls from scripts.

---

### CATEGORY D — Project Hygiene

---

#### D1 — Temp files and directories need cleanup + .gitignore update

**Severity:** LOW  
**Effort:** S  
**Files:** `logs/`, `lgb_attribution_results/`, `results/`, `__pycache__/`, `.coverage`, `gc3_test.py`, `scripts/experiments/full_bottleneck_analysis.py`

**Problem:**  
90 log files in `logs/`, 3 CSVs in `lgb_attribution_results/`, old GFF/txt in `results/`, and one-off experiment scripts (`gc3_test.py`) are tracked or untracked in the repo.

**Fix:**  
1. Add to `.gitignore`: `logs/`, `lgb_attribution_results/`, `results/`, `__pycache__/`, `.coverage`, `*.pyc`.
2. Move `gc3_test.py` to `scripts/experiments/gc3_analysis.py` with a proper header, or delete if analysis is complete.
3. Move `scripts/experiments/full_bottleneck_analysis.py` to its permanent home (already in the right folder, just untracked — commit or delete).

---

## Prioritized Roadmap

### Phase 1 — Merge existing work & correctness fixes (no retraining)

| # | Issue | Severity | Effort | Notes |
|---|---|---|---|---|
| 1 | Open PR: `fix/filter1-combined-only` → `main` | — | S | All 9 commits benchmarked |
| 2 | Fix MODEL_LOG inconsistency (A4) | MEDIUM | S | No code change |
| 3 | Fix `bi` dict class constant (B5) | LOW | S | Zero risk |
| 4 | Fix README CNN architecture description (B7) | LOW | S | Docs only |
| 5 | Rename `rule_based_select_starts` (C2) | HIGH | S | Pure rename |
| 6 | Fix silent `get_genome_by_id` (C6) | LOW | S | Raise instead of None |
| 7 | Cleanup temp files + .gitignore (D1) | LOW | S | Hygiene |
| 8 | Split requirements.txt (C4) | MEDIUM | S | No logic change |

### Phase 2 — Performance improvements (no ML retraining)

| # | Issue | Severity | Effort | Estimated gain |
|---|---|---|---|---|
| 9 | Vectorize `_one_hot_encode_dna` (B1) | HIGH | S | ~0.7s / genome |
| 10 | Fix `_upstream_imm` → Numba table (B3) | HIGH | S | ~0.35s / genome |
| 11 | Merge 3 group passes + groupby.rank (B4) | MEDIUM | S | ~0.25s / genome |
| 12 | Fix CNN permute → channels-first (B6) | LOW | S | ~0.1s / genome |
| 13 | Replace BioPython ProteinAnalysis (B2) | HIGH | M | ~0.75s / genome |

### Phase 3 — Code organization (mechanical refactors)

| # | Issue | Severity | Effort | Notes |
|---|---|---|---|---|
| 14 | Split `traditional_methods.py` into 3 files (C1) | HIGH | L | Zero logic changes |
| 15 | Move `GENOME_CATALOG` to JSON (C5) | MEDIUM | S | Data/code separation |
| 16 | Deduplicate RBS constants to config (C3) | MEDIUM | S | Drift prevention |
| 17 | Add `pyproject.toml` (C8) | LOW | S | Dev ergonomics |
| 18 | Fix CI test skip visibility + Numba equivalence test (C7) | MEDIUM | S | Test hygiene |

### Phase 4 — ML science improvements (require retraining + full benchmark)

| # | Issue | Severity | Effort | Notes |
|---|---|---|---|---|
| 19 | Fix ENC and CBI implementations (A1) | CRITICAL | S | Implement Wright 1990 ENC, CAI |
| 20 | Diversify training data (A2) | HIGH | L | Replace clonal E. coli / M. tb strains |
| 21 | Add Prodigal baseline to benchmark (A5) | MEDIUM | M | Scientific context |
| 22 | Remove duplicate `gap`/`d_baseline` feature (A3) | MEDIUM | S | SHAP first, then retrain |
| 23 | Ranked-list start selection (A6) | MEDIUM | L | Architectural change, retrain required |

---

## Open Questions Requiring Decisions

1. **ENC/CBI fix (A1):** Implement Wright ENC + CAI, or use a different metric? (Recommendation: Wright ENC + CAI are the most cited, easiest to validate against published tools)
2. **Training diversity (A2):** Which specific new genomes to add? Need NCBI accession numbers for 12 new Proteobacteria + 8 new Actinobacteria.
3. **Ranked-list start selection (A6):** Worth a full retrain? The pairwise model has been stable. This is the largest architectural change in Phase 4.
4. **`traditional_methods.py` split (C1):** Timing — do this before or after Phase 4 ML changes? (Recommendation: before, so ML changes are made in the clean structure)
5. **`gc3_test.py`:** Was this a one-off? Delete or commit to `scripts/experiments/`?
6. **`full_bottleneck_analysis.py`:** Commit or delete?
