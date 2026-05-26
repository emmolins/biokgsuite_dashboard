# Resampling-based stability for nb08 (Tier 0)

## The methodology shift

The original nb08 used **bootstrap CIs**: train each model once per KG,
then resample the test pairs with replacement to get a CI on AUROC.
That captures variance from *test-pair selection within a single run* —
nothing else.

The new section adds **multi-rerun resampling**: 5 full retrainings on
different random train/test splits, per model per KG. The empirical CI
across those 5 runs captures:

1. **Training stochasticity** — different random initialization, negative
   sample selection, batch ordering each run.
2. **Train/test split sensitivity** — which 10% of indication edges get
   held out changes the test set itself.

Both of those are missing from a bootstrap CI. For drug repurposing
specifically, this matters: bootstrap CIs are routinely 2-5× too tight
versus rerun-empirical CIs on KG embedding tasks, because the test-set
sampling step is *not* the dominant source of variance — training
stochasticity is.

## Why N=5

5 reruns gives:

- Mean ± std (basic stability)
- Empirical 95% percentile range (`np.percentile(values, [2.5, 97.5])`),
  acceptable for N ≥ 5
- For N < 5, the notebook falls back to `mean ± 1.96·SE`

If you want tighter empirical CIs, change `N_RERUNS=5` to `N_RERUNS=10`
in nb08 cell 6 (~2× the runtime).

## What's new in nb08

The original analysis is unchanged — the new section is **additive**.
The notebook now runs in this order:

1. Existing cells (single-run TransE/RotatE + Gemma + bootstrap CIs).
   Same as before; produces `embedding_comparison.csv`.
2. **NEW resampling section** (after Word-priors prose, before cleanup):
   - `prepare_kg_resampled(kg, rerun_idx)`: rerun-aware prep, different
     held-out 10% per rerun. Cached per `(kg, rerun_idx)`.
   - Multi-rerun training loop for TransE/RotatE on all 6 KGs.
   - Gemma re-scoring loop (encode once, re-score per rerun).
   - Long-form `embedding_comparison_resampled.csv` with `rerun` column.
   - Headline bar chart with rerun-empirical CIs.
   - Stability box plot showing per-rerun spread.
   - **Bootstrap-vs-rerun gap table**: quantifies how much the bootstrap
     CIs underestimated variance (ratio > 1 = bootstrap was too optimistic).
3. Existing cleanup cell + the original Tier-1 / Tier-2 stability cells
   (preserved as historical reference).

## Gemma special case

Gemma doesn't train; it encodes. Entity embeddings don't depend on
which drug-disease edges are held out — only the test pair pool does.
So:

- Gemma's `.npz` embedding cache (from `scripts/run_gemma_benchmark.sh`)
  is computed ONCE per KG.
- The resampling section reuses that cache and just re-scores the per-rerun
  test pairs.
- Total Gemma cost across 5 reruns: ~30 min instead of ~5 hours.

If you haven't run the original Gemma cell yet, the resampled Gemma
re-scoring will skip those KGs with a warning. Run
`scripts/run_gemma_benchmark.sh` first.

## Compute expectations

Per the supervisor's "aggressive parity across all 6 KGs" choice:

| KG | TransE × 5 | RotatE × 5 | Gemma × 5 |
|---|---|---|---|
| Hetionet | ~25 min | ~25 min | <5 min (cached) |
| PrimeKG | ~3 hr | ~3 hr | <5 min |
| DRKG | ~50 min | ~50 min | <5 min |
| OpenBioLink | ~40 min | ~40 min | <5 min |
| BioKG | ~15 min | ~15 min | <5 min |
| MATRIX | ~2.5 hr (overrides) / ~10 hr (full) | same | <5 min |

**CPU total**: ~15 hr (with MATRIX overrides) to ~25 hr (full MATRIX
hyperparams). On GPU: ~3-6 hr.

The runbook script lives at `scripts/run_resampled_nb08.sh`.

## Resumability

Every rerun's results are saved to JSON immediately on completion:

- `results/cache/embedding_<kg>_resampled.json` (TransE/RotatE)
- `results/cache/embedding_<kg>_gemma_resampled.json` (Gemma)

The notebook's main loop checks the `completed_reruns` list and skips
combos that are already done. Crash → restart → continues. MATRIX's
existing per-epoch checkpointing logic (`CHECKPOINT_KGS = {'matrix'}`)
also runs per-rerun.

## What to cite in the writeup

- **Headline stability**: the rerun-empirical CIs from
  `08_resampled_headline.{pdf,png}`. Mean across 5 reruns, 95% empirical
  CI.
- **Bootstrap-vs-rerun gap**: include the `compare_df` table from the
  notebook as a methods supplement — shows the bootstrap CIs would have
  been misleadingly tight by a factor of N.
- **Per-KG/model stability**: the box plot
  `08_resampled_stability_boxplot.{pdf,png}` — useful for any reviewer
  asking "how consistent is this across train/test splits?"

The Tier-1 (permutation test on Spearman ρ) and Tier-2 (bootstrap test-set
CI per AUROC point) sections at the end of nb08 remain useful but are now
secondary to the Tier-0 resampling numbers.

## When this won't fit

If full compute isn't feasible (no GPU, or MATRIX is too slow), reasonable
fallbacks in order of preference:

1. **Drop N_RERUNS to 3** — preserves the methodology, gets you a defensible
   mean ± range, halves the runtime. Empirical CI uses ±1.96·SE.
2. **Use PER_KG_OVERRIDES** (already in nb08 cell 6 for MATRIX) to reduce
   MATRIX epochs/dim. Validated to keep AUROC within ~0.01 of full training.
3. **Skip MATRIX in resampling** — run reruns on 5 small KGs, use the
   single-run MATRIX number with explicit caveat. Document in slide writeup.
4. **Reuse train/test splits** — change resampling to vary only the random
   seed (training-stochasticity-only), not the held-out edges. Cheaper
   because triple prep can be cached once per KG.

Note that 1 and 2 affect statistical power; 3 affects scope; 4 affects
methodology purity (you lose the train/test-split sensitivity component
of the variance).
