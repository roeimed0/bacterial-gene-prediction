# Bacterial Gene Prediction — Agent Instructions

## Project Overview

Bacterial Gene Prediction is an open source, MIT-licensed tool for *ab initio* prokaryotic gene prediction. It implements a 10-step hybrid pipeline combining traditional bioinformatics (ORF detection, Shine-Dalgarno RBS scoring, Interpolated Markov Models, codon bias analysis) with optional ML refinement (LightGBM and CNN+Dense classifiers). A React web UI and FastAPI backend expose the pipeline for interactive use.

## Purpose

Assist contributors by suggesting code changes, tests, and documentation edits for this repository while preserving prediction accuracy, pipeline stability, and backward compatibility.

## Persona & Tone

Concise, neutral, code-focused. Prioritize biological correctness, code readability, and test coverage. When in doubt, prefer the simpler, more explicit implementation over a clever one.

---

## Project Guidelines

Follow all contributing guidelines specified in [`CONTRIBUTING.md`](CONTRIBUTING.md). The key local references to load into context before making changes are:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — workflow, branch naming, commit style, code style, docstrings, tests
- [`src/config.py`](src/config.py) — thresholds, weights, and genome catalog; change these only with benchmark evidence
- [`src/traditional_methods.py`](src/traditional_methods.py) — core scoring pipeline; changes here affect all predictions
- [`src/ml_models.py`](src/ml_models.py) — ML classifiers; retrain models when features change
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — CI must remain green before any merge

---

## Decision Heuristics

- **Every change starts with a GitHub issue.** Never open a PR without a linked issue.
- Favor small, focused changes. One issue → one branch → one PR.
- If a change alters prediction output (sensitivity, precision, F1), include a benchmark table in the PR.
- Prefer readability over micro-optimizations unless a performance benchmark justifies the trade-off.
- Changes to scoring weights (`START_SELECTION_WEIGHTS`) or thresholds (`FIRST_FILTER_THRESHOLD`, `SECOND_FILTER_THRESHOLD`) in `src/config.py` must be backed by validation results on at least 5 genomes.
- ML model changes require retraining and re-evaluating on the 15-genome benchmark set.
- Add tests for behavioral changes; update docs only after code changes are final.
- Do not add features, refactors, or cleanup beyond what the linked issue describes.

---

## Type Hints Guidance

- Use PEP 484 style throughout. All public functions must have type-annotated signatures.
- Use builtin generics (`list`, `dict`, `tuple`) over `typing.List`, `typing.Dict` (Python 3.9+).
- Avoid `typing.cast`; prefer refactors that convey types naturally to the type checker.
- Run `mypy src/ api/` before opening a PR. New type errors introduced by your change must be resolved.

```python
# Correct
def find_orfs_candidates(sequence: str, min_length: int = 100) -> list[dict]:

# Avoid
from typing import List, Dict
def find_orfs_candidates(sequence: str, min_length: int = 100) -> List[Dict]:
```

---

## Docstring Guidance

Follow **numpydoc** conventions used across the repo. Every public function and class must have:

- Short one-line summary (imperative mood)
- Extended summary if the logic is non-obvious
- `Parameters`, `Returns`, `Raises`, `See Also`, `Notes`, `Examples` sections as applicable

Rules:
- Triple double-quotes, no blank line before/after the docstring body
- Parameter format: `name : type\n    Description.`
- Examples must be deterministic and runnable as doctests
- Biological terminology should be precise (e.g., "ORF" not "gene candidate" when it is not yet validated)

```python
def score_codon_bias_ratio(sequence: str, coding_table: dict, noncoding_table: dict) -> float:
    """
    Compute the log-odds codon usage bias score for a candidate ORF.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence of the ORF. Length must be divisible by 3.
    coding_table : dict
        Codon frequency mapping estimated from high-confidence coding regions.
    noncoding_table : dict
        Codon frequency mapping estimated from intergenic regions.

    Returns
    -------
    float
        Log-odds score. Positive values indicate coding-like codon usage.

    Raises
    ------
    ValueError
        If ``len(sequence) % 3 != 0``.

    Examples
    --------
    >>> score_codon_bias_ratio("ATGAAATAA", coding_table, noncoding_table)
    1.43
    """
```

---

## Pull Requests

Pull request titles must include one of the following prefixes:

| Prefix | Use for |
|--------|---------|
| `ENH:` | New feature or enhancement |
| `BUG:` | Bug fix |
| `DOC:` | Documentation additions or updates |
| `TST:` | Test additions or updates |
| `BLD:` | Build, CI, or dependency changes |
| `PERF:` | Performance improvement |
| `REF:` | Code refactor (no behavior change) |
| `TYP:` | Type annotation additions or fixes |

PR descriptions must follow the pull request template and succinctly describe the change. A few sentences is enough. PRs resolving a GitHub issue must link the issue in the description (`Closes #N`).

**Do not add summaries or additional comments to individual commit messages. The single PR description is sufficient.**

---

## Tests

Tests live in `tests/`. Mirror the `src/` structure:

| Source file | Test file |
|-------------|-----------|
| `src/pipeline.py` | `tests/integration/test_pipeline_smoke.py` |
| `src/traditional_methods.py` | `tests/test_traditional_methods.py` |
| `src/ml_models.py` | `tests/test_ml_models.py` |
| `src/data_management.py` | `tests/test_data_management.py` |
| `src/comparative_analysis.py` | `tests/test_comparative_analysis.py` |
| `src/validation.py` | `tests/test_validation.py` |
| `api/main.py` | `tests/test_api.py` |

Rules:
- Use small synthetic sequences (≤ 500 bp) for unit tests — never download real genomes in tests.
- Mark slow integration tests with `@pytest.mark.slow`.
- New behavioral changes must include at least one test that would have caught the bug or validated the feature.
- Run `pytest tests/ -v --cov=src --cov=api` before opening a PR.

---

## Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature / enhancement | `enh/issue-N-short-description` | `enh/issue-12-kmer-features` |
| Bug fix | `fix/issue-N-short-description` | `fix/issue-7-rbs-scoring-crash` |
| Documentation | `doc/issue-N-short-description` | `doc/issue-3-pipeline-diagram` |
| Performance | `perf/issue-N-short-description` | `perf/issue-9-lightgbm-speed` |
| Refactor | `refactor/issue-N-short-description` | `refactor/issue-5-modularize-cli` |
| Tests | `tst/issue-N-short-description` | `tst/issue-15-orf-unit-tests` |
| ML | `ml/issue-N-short-description` | `ml/issue-123-lgb-feature-reduction` |

---

## Repository Layout

```
src/                        # Core Python pipeline
  pipeline.py               # Unified entry point: predict_genome(), write_gff()
  traditional_methods.py    # ORF detection, RBS, IMM, scoring
  ml_models.py              # LightGBM + CNN+Dense classifiers
  data_management.py        # NCBI download, file I/O
  comparative_analysis.py   # Validation metrics
  config.py                 # Thresholds, weights, genome catalog
  cache.py                  # ORF cache layer
  validation.py             # Validation wrapper functions
api/                        # FastAPI REST backend
  main.py                   # 10+ prediction and validation endpoints
  models.py                 # Pydantic request/response schemas
scripts/                    # Standalone utility scripts (not importable)
  benchmark.py              # Run benchmark on TEST_GENOMES, save to experiments/log.json
  predict_batch.py          # Batch prediction over multiple FASTA files
  train_lgb.py              # Train OrfGroupClassifier (LightGBM)
  train_hybrid.py           # Train HybridGeneFilter
  manage_lgb.py             # LGB model management and evaluation
  calibrate_start_weights.py# Calibrate START_SELECTION_WEIGHTS
  compare_lgb_models.py     # Compare LGB model versions
gene-prediction-frontend/   # React 19 + Vite + Tailwind web UI
models/                     # Trained ML model artifacts (.pkl)
notebooks/                  # Research notebooks (not production)
tests/                      # pytest test suite
  integration/              # Integration tests (marked @pytest.mark.slow)
.github/
  workflows/ci.yml          # Lint + pytest on Python 3.9–3.11
  ISSUE_TEMPLATE/           # bug_report, feature_request, performance,
                            # refactor, tests, documentation
```
