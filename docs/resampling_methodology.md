# Resampling-based stability for nb08

## Why resampling, not bootstrap CIs

An earlier version of nb08 reported **bootstrap CIs**: train each model once
per KG, then resample the test pairs with replacement to get a CI on AUROC.
That only captures variance from test-pair selection within a single training
run, which is not the dominant source of uncertainty for KG embeddings.

nb08 now reports **multi-rerun resampling** instead. For each rerun it draws a
fresh random 90/10 split and retrains TransE and RotatE from scratch, so the
empirical spread across reruns captures the two sources a bootstrap CI misses:

1. **Training stochasticity**: random initialization, negative-sample
   selection, and batch ordering differ each run.
2. **Train/test split sensitivity**: which 10% of indication edges are held
   out changes the test set itself.

On these tasks bootstrap CIs are routinely 2 to 5 times too tight, because the
test-pair sampling step is not where most of the variance lives. The bootstrap
machinery has been removed from nb08 entirely; resampling is the only stability
reported.

## Choosing N_RERUNS

`N_RERUNS` (nb08 hyperparameters cell, default 3, override via the `N_RERUNS`
environment variable) controls the number of reruns:

- Mean and standard deviation across reruns (basic stability).
- For N of 5 or more, a 95% empirical percentile range
  (`np.percentile(values, [2.5, 97.5])`).
- For N below 5, the figures fall back to mean plus or minus 1.96 times the
  standard error.

Three reruns gives a defensible mean and range at modest cost; raise N_RERUNS
for tighter empirical CIs (roughly linear in runtime).

## Outputs

The resampling section writes a long-form table with one row per
`(kg, model, strategy, rerun)`:

- `results/tables/08_embedding_comparison_resampled.csv`

and the figures are computed from it:

- `results/figures/08_resampled_auroc.{pdf,png}`: mean AUROC per KG for TransE,
  RotatE, and the Gemma name prior, with 95% CIs across reruns.
- `results/figures/08_lift_over_gemma.{pdf,png}`: per-rerun paired lift of each
  trained model over the Gemma name prior.

## Gemma special case

Gemma does not train, it encodes. Entity embeddings do not depend on which
drug-disease edges are held out, only the test-pair pool does. So Gemma is
encoded once per KG (`scripts/run_gemma_benchmark.sh`, cached as a `.npz`), and
the resampling section reuses that cache and re-scores the per-rerun test pairs.
This makes Gemma reruns far cheaper than the trained models. By default nb08
reuses the committed single-run Gemma metrics from
`results/tables/08_embedding_comparison.csv` and does not re-encode at all; set
`GEMMA_FORCE_REENCODE = True` (with `HF_TOKEN` set and the model licence
accepted) to encode for real.

## Resumability

The resampling section is idempotent. If
`results/tables/08_embedding_comparison_resampled.csv` already exists, the
training and re-scoring loops are skipped and the figures load straight from the
CSV. Per-rerun progress is also cached under `results/cache/`
(`embedding_<kg>_resampled.json`, `embedding_<kg>_gemma_resampled.json`), so an
interrupted run resumes from the last completed rerun. MATRIX additionally uses
per-epoch checkpointing (`CHECKPOINT_KGS = {'matrix'}`).

## Compute and fallbacks

Full parity across all 6 KGs is roughly 15 to 25 CPU-hours, or 3 to 6 hours on
a GPU, dominated by PrimeKG and MATRIX. See `scripts/run_resampled_nb08.sh` and
`scripts/hpc/` for the batch runbooks. If full compute is not feasible:

1. Lower `N_RERUNS` (preserves the methodology, fewer reruns).
2. Use the per-KG hyperparameter overrides already set for MATRIX.
3. Skip MATRIX in resampling and report its single-run number with a caveat.
