#!/usr/bin/env bash
# One-time HPC setup. Run this ONCE on the login node before submitting any
# Slurm jobs. Idempotent — safe to re-run if anything fails partway through.
#
# Usage:   bash scripts/hpc/setup.sh
#
# What this does:
#   1. Loads required modules (python 3.12, uv)
#   2. Creates a uv-managed venv at ~/biokgsuite/.venv
#   3. Installs all deps (biokgsuite + torch + transformers + sentence-transformers)
#   4. Reminds you to do the manual steps (HF login, KG data, accept Gemma license)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/biokgsuite}"
cd "$REPO_DIR"

echo "================================================================="
echo "BioKGSuite HPC setup"
echo "  repo:    $REPO_DIR"
echo "  python:  3.12.11"
echo "================================================================="

# ── 1. Modules ──────────────────────────────────────────────────────
echo
echo "[1/4] Loading modules..."
module purge
module load python/3.12.11
module load uv/latest
echo "      python: $(which python) ($(python --version))"
echo "      uv:     $(uv --version)"

# ── 2. Virtualenv ───────────────────────────────────────────────────
echo
echo "[2/4] Creating venv at .venv ..."
if [[ -d .venv ]]; then
    echo "      .venv already exists — skipping creation"
else
    uv venv --python 3.12
fi
source .venv/bin/activate

# ── 3. Deps ──────────────────────────────────────────────────────────
echo
echo "[3/4] Installing dependencies..."
# Core biokgsuite (editable install picks up pyproject.toml + environment.yml deps)
uv pip install -e .
# Notebook execution
uv pip install jupyter nbconvert ipykernel
# Gemma stack
uv pip install torch transformers sentence-transformers
# HF auth helper
uv pip install huggingface_hub[cli]
# Make sure scientific stack is in (in case pyproject didn't pull them)
uv pip install numpy pandas scipy scikit-learn matplotlib pyarrow tqdm pyyaml

echo "      installed: $(pip list 2>/dev/null | grep -iE '^(torch|transformers|sentence|numpy|pandas|scipy|matplotlib|jupyter)' | wc -l) key packages"

# ── 4. Reminders ────────────────────────────────────────────────────
echo
echo "[4/4] DONE. Three manual steps before you can submit jobs:"
echo
echo "  (a) HuggingFace auth (one-time, paste your hf_... token):"
echo "        huggingface-cli login"
echo
echo "  (b) Accept the EmbeddingGemma license at:"
echo "        https://huggingface.co/google/embeddinggemma-300m"
echo "      (wait for Google's review email if not already approved)"
echo
echo "  (c) Get the KG data files under data/  (see scripts/hpc/README.md"
echo "      for the rsync recipe)."
echo
echo "Once those are done, submit the run with:"
echo "  sbatch scripts/hpc/run_all.sbatch"
