# Contributing to Bacterial Gene Prediction

Thank you for your interest in contributing. This guide follows the workflow used by projects like pandas and scikit-learn.

---

## Table of Contents

- [Development workflow](#development-workflow)
- [Branch naming](#branch-naming)
- [Commit messages](#commit-messages)
- [Code style](#code-style)
- [Docstrings](#docstrings)
- [Tests](#tests)
- [Submitting a pull request](#submitting-a-pull-request)

---

## Development workflow

Every change starts with a GitHub issue.

```
open issue → branch → code → tests → PR → review → merge
```

1. **Open or find an issue** — never start work without a linked issue.
2. **Comment on the issue** to say you're working on it.
3. **Create a branch** from `main` (see naming below).
4. **Make changes** in small, focused commits.
5. **Write or update tests** for anything you changed.
6. **Open a PR** and fill in the template — link it to the issue.
7. **Address review feedback** in new commits (do not force-push).
8. A maintainer will merge once CI is green and review is approved.

---

## Branch naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature / enhancement | `feature/issue-N-short-description` | `feature/issue-12-benchmark-prodigal` |
| Bug fix | `fix/issue-N-short-description` | `fix/issue-7-rbs-scoring-crash` |
| Documentation | `doc/issue-N-short-description` | `doc/issue-3-readme-pipeline-diagram` |
| Performance | `perf/issue-N-short-description` | `perf/issue-9-lightgbm-inference-speed` |
| Refactor | `refactor/issue-N-short-description` | `refactor/issue-5-modularize-pipeline` |

---

## Commit messages

Follow the format:

```
<type>: short imperative description (#issue)

Optional longer explanation of why, not what.
```

Types: `ENH`, `BUG`, `DOC`, `PERF`, `REF`, `TST`, `BLD`

Examples:

```
ENH: add benchmark comparison against Prodigal (#12)
BUG: fix RBS scoring crash on sequences < 20 bp (#7)
DOC: add pipeline architecture diagram to README (#3)
TST: add unit tests for ORF detection module (#15)
```

---

## Code style

- Formatter: **black** (line length 100)
- Import sorter: **isort**
- Linter: **flake8**

Install and run before committing:

```bash
pip install black isort flake8
black src/ api/ hybrid_predictor.py
isort src/ api/ hybrid_predictor.py
flake8 src/ api/ hybrid_predictor.py --max-line-length=100
```

CI will reject PRs that fail these checks.

---

## Docstrings

Use **numpydoc** style for all public functions and classes:

```python
def score_orf(sequence: str, codon_table: dict) -> float:
    """
    Compute the codon usage bias score for a candidate ORF.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence of the ORF (must be divisible by 3).
    codon_table : dict
        Mapping of codon -> frequency from the self-training step.

    Returns
    -------
    float
        Normalized codon usage bias score in [0, 1].

    Raises
    ------
    ValueError
        If sequence length is not divisible by 3.

    Examples
    --------
    >>> score_orf("ATGAAATAA", codon_table)
    0.72
    """
```

---

## Tests

Tests live in `tests/`. Run with:

```bash
pytest tests/ -v
```

- Mirror the `src/` structure: `src/traditional_methods.py` → `tests/test_traditional_methods.py`
- Use small synthetic sequences for unit tests (no real genome downloads).
- Mark slow integration tests with `@pytest.mark.slow`.

---

## Submitting a pull request

1. Push your branch: `git push -u origin feature/issue-N-description`
2. Open a PR on GitHub against `main`.
3. Fill in the PR template — include the issue number, what changed, and how to test it.
4. Make sure CI passes (lint + tests).
5. Request a review.

Do not merge your own PRs.
