---
name: Tests
about: Add missing tests, fix broken tests, or improve test coverage
title: "TST: <short description>"
labels: ["tests", "needs-triage"]
assignees: ''
---

## What is being tested?

<!-- Describe the module, function, or pipeline stage that needs test coverage -->

## Current test coverage gap

<!-- What scenario, edge case, or code path is not currently covered?
     If tests are broken, paste the failure output below. -->

```
<pytest output or traceback here>
```

## Proposed tests

<!-- Describe the test cases you plan to add -->

- [ ] Happy path: <!-- e.g., valid genome input produces correct gene count -->
- [ ] Edge case: <!-- e.g., empty sequence, single gene, max-length contig -->
- [ ] Regression: <!-- e.g., ensure issue #N does not reoccur -->

## Test data / fixtures

<!-- List any new fixtures, sample genomes, or mock data that need to be created -->

## Additional context

<!-- Related issues, PRs, or coverage reports -->
