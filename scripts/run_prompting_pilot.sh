#!/usr/bin/env bash
# Run the prompting-strategy pilot for nb09.
#
# What this does:
#   Re-executes notebook 09 against the new strategy axis. Each of the 8
#   new strategies + the baseline runs over (kg × condition × pair × reseed)
#   and writes one row per cell into results/tables/09_llm_runs/responses.parquet.
#
# What it does NOT do:
#   - Start Ollama for you (must be already running on localhost:11434)
#   - Pull the model (run `ollama pull llama3.1:8b` once beforehand)
#   - Resample the pairs.parquet file (uses whatever pairs are already on disk)
#
# Pilot scale (default):
#   N_POS_APPROVED + N_POS_PHASE3 + N_POS_PHASE12 = 150 positives
#   N_NEG_RANDOM + N_NEG_PLAUSIBLE = 150 negatives  → 300 pairs
#   The new strategies skip shuffled_kg, so cross is: 9 strategies × 2 conds × 3 reseeds × 6 KGs × 300 pairs
#   with per-strategy n_calls weights: 1+1+2+1+1+5+1+1+2 = 15 calls per cell
#
#   Total calls ≈ 300 × 2 × 3 × 6 × 15 ≈ 162,000 calls
#
#   IF you want a TRUE pilot (50 pairs first), edit nb09 cell 2 to set
#   N_POS_APPROVED=15, N_POS_PHASE3=15, N_POS_PHASE12=20, N_NEG_RANDOM=25,
#   N_NEG_PLAUSIBLE=25  →  100 pairs / 30K calls / ~30 min wall time
#
# Expected wall-clock on llama3.1:8b @ ~3 sec/call:
#   pilot (50 pairs):    ~30 min
#   medium (150 pairs):  ~2-3 hr
#   full (300 pairs):    ~4-6 hr
# Add overhead for self_consistency (5 sequential samples per cell).
#
# Resumable: the strategy loop checks responses.parquet for existing
# (llm, kg, strategy, condition, pair_idx, reseed) tuples and skips them.
# Crash → restart → continues where it left off.

set -euo pipefail

cd "$(dirname "$0")/.."

# ── Preflight checks ────────────────────────────────────────────────
echo "Preflight…"

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "ERROR: Ollama not reachable at localhost:11434" >&2
    echo "  Start with:  ollama serve" >&2
    exit 1
fi
echo "  Ollama reachable ✓"

if ! curl -fsS http://localhost:11434/api/tags | grep -q 'llama3.1:8b'; then
    echo "ERROR: llama3.1:8b not pulled" >&2
    echo "  Pull with:   ollama pull llama3.1:8b" >&2
    exit 1
fi
echo "  llama3.1:8b model available ✓"

if [[ ! -f "results/tables/09_llm_runs/pairs.parquet" ]]; then
    echo "WARNING: pairs.parquet doesn't exist yet."
    echo "  First run cells 1-4 of nb09 once to sample the pair pool, then re-run this."
    echo "  Continuing anyway (the notebook will sample on its own if needed)..."
fi

# ── Run the notebook end-to-end ─────────────────────────────────────
# Long timeout because the strategy loop is the bulk of the work
# and we don't want it killed mid-call.
echo
echo "Executing nb09 (this is the long part)..."
echo "Progress is written to results/tables/09_llm_runs/responses.parquet"
echo "incrementally — safe to Ctrl-C and re-run later."
echo

jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=86400 \
        eval_notebooks/09_llm_integration.ipynb

echo
echo "Done."
echo
echo "Outputs:"
echo "  - results/tables/09_llm_runs/responses.parquet  (raw rows)"
echo "  - results/figures/09_prompting_strategies.png    (headline)"
echo "  - results/figures/09_prompting_calibration.png   (calibration)"
echo "  - results/figures/09_punchline.png               (existing KG-quality plot)"
