#!/usr/bin/env bash
#SBATCH --job-name=biokgsuite-main
#SBATCH --partition=def
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=logs/main-%j.out
#SBATCH --error=logs/main-%j.err
#
# Runs the main benchmark notebooks 01-07 in order, then 00 (which aggregates
# their checkpoints), on one CPU compute node. No GPU is requested: embedding
# training and Gemma encoding live in nb08 (scripts/hpc/run_all.sbatch), not
# here. MATRIX is the memory driver, hence 120 GB.
#
# Submit:   sbatch scripts/hpc/run_main.sbatch
# Monitor:  squeue -u $USER
#           tail -f logs/main-<jobid>.out
# Resumable: each notebook caches to results/checkpoints/ and results/cache/,
#            so a re-submit skips work that already finished.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/biokgsuite}"
cd "$REPO_DIR"
mkdir -p logs

echo "================================================================="
echo "HOST:  $(hostname)  DATE: $(date)  JOBID: ${SLURM_JOB_ID:-(no slurm)}"
echo "REPO:  $REPO_DIR"
echo "================================================================="

# ── Environment ─────────────────────────────────────────────────────
module purge
module load python/3.12.11
module load uv/latest
source .venv/bin/activate
echo "  python: $(which python) ($(python --version))"

# 00-07 need networkx (topology + community detection); the nb08 setup does
# not install it. Add it on demand so a fresh venv still works.
python -c "import networkx" 2>/dev/null || uv pip install "networkx>=3.2,<4"

export PYTHONUNBUFFERED=1     # stream prints to the Slurm log immediately
export TQDM_MININTERVAL=30    # one progress line per ~30s (readable in logs)
export MPLBACKEND=Agg         # headless matplotlib

# ── Run 01-07 in order, then 00 (aggregation) ───────────────────────
# set -e stops the job if any notebook errors, so 00 never aggregates a
# partial set of checkpoints.
for nb in 01_coverage 02_annotation_accuracy 03_trustworthiness \
          04_topology 05_stability 06_task_performance \
          07_generalization 00_benchmark_summary; do
    echo
    echo "================================================================="
    echo ">>> Executing eval_notebooks/${nb}.ipynb   ($(date +%H:%M:%S))"
    echo "================================================================="
    jupyter nbconvert \
        --to notebook --execute --inplace \
        --log-level=INFO \
        --ExecutePreprocessor.timeout=86400 \
        "eval_notebooks/${nb}.ipynb"
done

echo
echo "================================================================="
echo "Done. Outputs in results/figures, results/tables, results/checkpoints,"
echo "and the executed notebooks (00-07) with inline cell outputs."
echo
echo "Pull results back to your Mac:"
echo "  rsync -avz hpc-iad-core-001-login:biokgsuite/results/ ~/biokgsuite/results/"
echo "  rsync -avz hpc-iad-core-001-login:biokgsuite/eval_notebooks/ ~/biokgsuite/eval_notebooks/"
echo "================================================================="
