#!/bin/bash
#SBATCH --job-name=dream_hist
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dream_history_%j.out
#SBATCH --error=logs/dream_history_%j.err

mkdir -p logs

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the CDG2.0 checkout}"
: "${VENV_ACTIVATE:?Set VENV_ACTIVATE to the environment activation script}"
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME_STUBS:-$PROJECT_ROOT/runtime_stubs}:$PROJECT_ROOT"

echo "===== JOB INFO ====="
hostname
date
nvidia-smi

echo "===== EXISTING PROGRESS ====="
find outputs_dream_native_history_full -type f -name "*.pt" 2>/dev/null | wc -l || true
wc -l outputs_dream_native_history_full/manifest.jsonl 2>/dev/null || true

echo "===== RUN RESUME ====="
python -u dream_native_history_record_resume.py

echo "===== FINISHED ====="
date
