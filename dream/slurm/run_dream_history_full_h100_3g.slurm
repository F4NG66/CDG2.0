#!/bin/bash
#SBATCH --job-name=dream_h3g
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dream_h3g_%j.out
#SBATCH --error=logs/dream_h3g_%j.err

mkdir -p logs

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the CDG2.0 checkout}"
: "${VENV_ACTIVATE:?Set VENV_ACTIVATE to the environment activation script}"
source "$VENV_ACTIVATE"
cd "$PROJECT_ROOT"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME_STUBS:-$PROJECT_ROOT/runtime_stubs}:$PROJECT_ROOT"

echo "===== JOB INFO ====="
echo "job id: $SLURM_JOB_ID"
hostname
date
nvidia-smi

echo
echo "===== EXISTING PROGRESS BEFORE RUN ====="
find outputs_dream_native_history_full -type f -name "*.pt" 2>/dev/null | wc -l || true
wc -l outputs_dream_native_history_full/manifest.jsonl 2>/dev/null || true

echo
echo "===== RUN RESUME ====="
python -u dream_native_history_record_resume.py

echo
echo "===== PROGRESS AFTER RUN ====="
find outputs_dream_native_history_full -type f -name "*.pt" 2>/dev/null | wc -l || true
wc -l outputs_dream_native_history_full/manifest.jsonl 2>/dev/null || true

echo
echo "===== FINISHED ====="
date
