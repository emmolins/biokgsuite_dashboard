#!/usr/bin/env bash
#SBATCH --job-name=biokg-ladder
#SBATCH --partition=def
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:H100:1
#SBATCH --output=logs/ladder-%j.out
#SBATCH --error=logs/ladder-%j.err
#
# Same-family Llama 3.x capability-scaling sweep — ONE (model, seed) per job, single H100.
# Use for the small/mid rungs: llama3.2:1b, llama3.2:3b, llama3.1:8b, llama3.1:70b.
# (llama3.3:70b is already done; 405B needs the multi-GPU script run_llm_405b.sbatch.)
#
# Submit one job per (model, seed) so they run in PARALLEL and finish fast:
#   for M in llama3.2:1b llama3.2:3b llama3.1:8b; do
#     for S in 1 2 3; do
#       sbatch --export=ALL,MODEL=$M,SEED=$S scripts/hpc/run_llm_ladder.sbatch
#     done
#   done
#
# Monitor:   tail -f logs/ladder-<jobid>.out
# Resumable: just resubmit a job if it dies (idempotent output name per model+seed).

set -euo pipefail
MODEL="${MODEL:?set MODEL, e.g. --export=ALL,MODEL=llama3.1:8b,SEED=1}"
SEED="${SEED:-1}"
SHUFFLES="${SHUFFLES:-2}"
GOLD="${GOLD:-gold_standard_v4_bigset.tsv}"
SAFE="$(echo "$MODEL" | tr ':/.' '___')"           # llama3.1:8b -> llama3_1_8b
OUT="09_big_${SAFE}_s${SEED}.csv"

REPO_DIR="${REPO_DIR:-$HOME/biokgsuite}"
cd "$REPO_DIR"; mkdir -p logs
echo "HOST $(hostname)  DATE $(date)  JOB ${SLURM_JOB_ID:-none}  MODEL $MODEL  SEED $SEED  -> $OUT"

module purge 2>/dev/null || true
module load python/3.12.11 2>/dev/null || module load python 2>/dev/null || true
module load uv/latest 2>/dev/null || true
source .venv/bin/activate
python -c "import sys; print('python', sys.version.split()[0])" || { echo "FATAL: venv python unusable on $(hostname)"; exit 1; }
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=15

# ── Ollama: install to ~/.local if missing (no root needed) ─────────────────
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p "$HOME/.local/bin" "$OLLAMA_MODELS"
if ! command -v ollama >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/ollama" ]]; then
    echo "[ollama] downloading standalone binary..."
    curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
    ( tar --zstd -xf /tmp/ollama.tar.zst -C "$HOME/.local" 2>/dev/null \
      || tar -I zstd -xf /tmp/ollama.tar.zst -C "$HOME/.local" )
fi
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi || true

# ── start server, wait ──────────────────────────────────────────────────────
ollama serve > "logs/ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 2; done
echo "[ollama] up"

# ── pull this rung's model (idempotent) ─────────────────────────────────────
ollama pull "$MODEL"

# ── run the sweep ───────────────────────────────────────────────────────────
python eval_notebooks/run_llm_sweep.py --model "$MODEL" --gold "$GOLD" \
    --seeds "$SEED" --shuffles "$SHUFFLES" --out "$OUT"

echo "DONE -> results/tables/09_llm_runs/$OUT"
