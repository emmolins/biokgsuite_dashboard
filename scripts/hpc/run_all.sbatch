#!/usr/bin/env bash
#SBATCH --job-name=biokgsuite-nb08
#SBATCH --partition=def
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:H100:1
#SBATCH --output=logs/nb08-%j.out
#SBATCH --error=logs/nb08-%j.err
#
# Runs notebook 08 end-to-end on one HPC compute node with 1 H100.
#
#   - 32 CPU cores: plenty for TransE/RotatE training (NumPy, multi-threaded BLAS)
#                   plus loading 30 GB of MATRIX edges
#   - 96 GB RAM:    enough for MATRIX (~14 GB edges + working set)
#   - 1 H100:       accelerates EmbeddingGemma encoding by ~30-50× vs CPU
#   - 24 hr wall:   generous; expected wall time is 2-4 hr
#
# Submit:
#   sbatch scripts/hpc/run_all.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/nb08-<jobid>.out
#
# Resumable: if the job is killed mid-run, just sbatch again. nb08 caches
# per-rerun per-KG results to results/cache/embedding_<kg>_resampled.json
# and skips completed combos.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/biokgsuite}"
cd "$REPO_DIR"
mkdir -p logs

echo "================================================================="
echo "HOST:     $(hostname)"
echo "DATE:     $(date)"
echo "JOBID:    ${SLURM_JOB_ID:-(no slurm)}"
echo "REPO:     $REPO_DIR"
echo "================================================================="
nvidia-smi -L 2>&1 | head -3 || echo "  (no GPU on this node?)"
echo

# ── Environment ─────────────────────────────────────────────────────
module purge
module load python/3.12.11
module load uv/latest
# Some clusters require an explicit CUDA module. Try loading; ignore if absent.
module load cuda 2>/dev/null || true

source .venv/bin/activate
echo "  python:  $(which python)"
echo "  torch:   $(python -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available())' 2>&1)"
echo

# Faster HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Make sure cached HF token is on PATH (huggingface-cli login wrote it here)
export HF_HOME="$HOME/.cache/huggingface"

# ── Run nb08 end-to-end ─────────────────────────────────────────────
echo "Executing eval_notebooks/08_embedding_validation.ipynb ..."
echo "This runs ALL sections:"
echo "  - existing single-run TransE/RotatE + bootstrap CIs"
echo "  - Gemma word-priors encoding (uses GPU) + scoring"
echo "  - resampling: 5 reruns × 6 KGs × 2 models = 60 training runs"
echo "  - aggregation, figures, bootstrap-vs-rerun comparison"
echo "Progress streams to this log file."
echo "================================================================="

# 86,400s = 24h timeout (matches Slurm wall time)
jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=86400 \
        eval_notebooks/08_embedding_validation.ipynb

echo
echo "================================================================="
echo "Done. Outputs:"
echo "  - results/embedding_comparison_resampled.csv       (long-form)"
echo "  - results/figures/08_resampled_headline.{pdf,png}"
echo "  - results/figures/08_resampled_stability_boxplot.{pdf,png}"
echo "  - results/figures/08_gemma_vs_kge_auroc.{pdf,png}"
echo "  - results/checkpoints/08_embedding_validation.pkl"
echo
echo "To pull results back to your Mac:"
echo "  rsync -avz hpc-iad-core-001-login:biokgsuite/results/ ~/biokgsuite/results/"
echo "================================================================="
