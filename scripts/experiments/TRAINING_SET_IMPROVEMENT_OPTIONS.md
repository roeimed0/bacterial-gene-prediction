# Training Set Improvement — Explored Options & Future Potential

**Context (2026-05-18):** Problem genomes (Bordetella 47% TP, Streptomyces 48%,
M. leprae 66%) have training sets dominated by FPs (pseudogenes, IS elements,
gene fragments). Every de novo filtering approach was explored and benchmarked.
Current best result: adaptive codon bias filter (|GC3-GC12| >= GC-scaled threshold)
gives +0.22pp overall F1 with M. leprae still slightly regressing.

**Decision:** Drop training set experiments for now. Document all options with
their potential for future revisitation.

---

## KILLED — No Path Forward

### 1. RBS Score Filter
- **Signal:** rbs_score threshold
- **Why failed:** Bimodal distribution (−5.0 or >8.0). Glimmer/Flexible already
  select for purine-rich upstream regions, so FPs have similar RBS scores to TPs.
  Max FP removal: 4-11%. Required: 40%.
- **Future potential:** LOW. The bimodal nature is structural — inherent to how
  the RBS scorer works.

### 2. Overlap Count / Group Size Filters
- **Signal:** n_overlap (competing ORFs), grp_size (same stop codon alternatives)
- **Why failed:** TPs have MORE overlap/larger groups than FPs. Real gene loci
  attract more alternative starts. Signal is backwards.
- **Future potential:** LOW. Could use as POSITIVE selector (keep high-overlap
  ORFs) but that doesn't remove FPs selectively.

### 3. "Take Longest ORF" per Stop Codon
- **Signal:** Replace training ORF with longest ORF at same stop
- **Why failed:** Glimmer already selects the longest ORF at each stop. 0% change.
- **Future potential:** NONE. Already done by design.

### 4. Downstream Coding Leakage
- **Signal:** Score 120bp downstream of training ORF's stop in same reading frame
- **Why failed:** Corrupted initial model (47-66% FP) cannot reliably detect
  coding signal. All downstream scores negative. 0/6 genomes significant.
- **Future potential:** MEDIUM — if starting from a cleaner initial model (e.g.,
  after one pass of codon bias filter). May work as iteration step 2 rather than
  step 1.

### 5. Iterative Self-Training (Prodigal-style)
- **Signal:** Re-score training ORFs with their own model; keep high-scorers
- **Why failed:** Circular — corrupted model re-confirms FPs. Gene-fragment FPs
  score HIGH (they're inside coding regions), making iteration reinforce them.
- **Future potential:** MEDIUM-HIGH — viable if starting seed is clean enough
  (TP% > 75%). Could work AFTER codon bias filter pre-cleans the training set.

---

## PARTIAL SUCCESS — Keep as Complement

### 6. Adaptive Codon Position Bias Filter (|GC3-GC12|)
- **Signal:** |GC3 - (GC1+GC2)/2| >= adaptive threshold based on genome GC%
- **Result:** +0.22pp overall F1. Activates for 4 of 20 genomes. M. leprae
  still regresses −0.32pp. Already implemented in `src/traditional_methods.py`.
- **Current status:** Active in production (`filter_training_adaptive()`).
- **Future tuning:** Try percentile-based threshold instead of fixed value.
  Could also add Nc as secondary gate.

---

## NOT TRIED — Future Potential

### 7. Stop-Codon-Neighborhood Scoring (no model needed)
- **Idea:** For each training ORF, count how many other ORFs (from all_orfs)
  have stops within ±30bp of this ORF's stop. FP gene-fragments would have
  MORE nearby stops (being inside a gene that has many reading frames) while
  real complete genes have isolated stops.
- **Signal:** `nearby_stop_density` — de novo, no model
- **Why not tried:** Emerged too late in exploration. Distinct from the
  "group size" signal (which was same-stop, not near-stop).
- **Potential:** MEDIUM. Targets the gene-fragment FPs specifically (30-36%
  of problem genome FPs). Quick to implement and test.

### 8. ORF Density Context
- **Idea:** Count ORFs per kb in the genomic region around each training ORF.
  Real gene regions in problem genomes tend to cluster, while pseudogene deserts
  might have isolated ORFs.
- **Signal:** de novo, no model
- **Potential:** LOW-MEDIUM. Dependent on genome architecture.

### 9. Two-Phase Training (clean seed + downstream iteration)
- **Idea:**
  1. Apply codon bias filter (|GC3-GC12|) to get a cleaner initial seed
  2. Build rough models on the filtered seed
  3. Use downstream leakage signal to remove gene-fragment FPs (now detectable
     because the model is cleaner)
  4. Rebuild final models
- **Why not tried:** Complexity; also M. leprae's |GC3-GC12| signal is weak,
  so the initial seed might still be contaminated.
- **Potential:** MEDIUM-HIGH. Most principled approach. Requires ~2x runtime.

### 10. Reading-Frame Consistency Score
- **Idea:** For a training ORF at position (start, stop), score the reading
  frame UPSTREAM of the start. If the upstream region in the same reading frame
  also looks coding (high IMM score), the true start is further upstream and
  this training ORF is a wrong-start version. Not a wrong-start FP (those
  already match by stop-tolerance) but an alternative-start that might confuse
  the codon model.
- **Signal:** upstream_same_frame_imm > threshold
- **Potential:** LOW-MEDIUM. Requires an initial model. Different from downstream
  leakage but same circular dependency issue.

### 11. Phylogenetic Codon Table Bootstrap
- **Idea:** Instead of self-training, bootstrap codon frequencies from a
  pre-computed database of genus/family-level codon usage (e.g., NCBI CDS stats).
  This breaks the circular dependency entirely.
- **Violates de novo constraint?** YES — requires external database.
- **Potential:** HIGH if de novo constraint is relaxed. A single lookup table
  per bacterial family would dramatically improve initial model quality for
  problem genomes.

### 12. Minimum Description Length (MDL) Training Set Selection
- **Idea:** Select training ORFs that MINIMIZE the description length of the
  resulting model (Rissanen's MDL principle). ORFs that are hard to explain by
  the coding model (gene fragments, pseudogenes) would increase MDL and be
  excluded.
- **Potential:** HIGH but complex (O(n²) or requires approximation). This is
  what GeneMark-S2 effectively does. Would require significant implementation.

---

## Summary Table

| Option | Signal type | Potential | Effort | De novo? |
|--------|------------|-----------|--------|----------|
| Nearby-stop density | geometric | MEDIUM | Low | ✓ |
| Two-phase (clean seed + downstream) | model | MEDIUM-HIGH | Medium | ✓ |
| Iterative training (from clean seed) | model | MEDIUM-HIGH | Low | ✓ |
| Downstream leakage (from clean seed) | model | MEDIUM | Low | ✓ |
| Reading-frame upstream score | model | LOW-MEDIUM | Low | ✓ |
| ORF density context | geometric | LOW-MEDIUM | Low | ✓ |
| Phylogenetic codon bootstrap | external | HIGH | Medium | ✗ |
| MDL training selection | information | HIGH | High | ✓ |

**Bottom line:** The most feasible next step is "nearby-stop density" (geometric,
no model needed) and "two-phase" (codon bias seed + downstream iteration). Neither
requires external databases. Both are implementable in <200 lines.
