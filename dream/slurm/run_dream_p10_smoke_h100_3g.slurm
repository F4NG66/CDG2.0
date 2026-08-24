#!/bin/bash
#SBATCH --job-name=dream_p10_smoke
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dream_p10_smoke_%j.out
#SBATCH --error=logs/dream_p10_smoke_%j.err

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

python -u scripts/p10_steer2.py \
  --task dose_response \
  --backend dream_attack \
  --out-dir outputs_dream_native_history_full \
  --tag dream_outmask_L14_smoke \
  --dirs-out analysis_output/p10_dream_outmask_directions \
  --steer-out outputs_p10_dream_outmask_L14_smoke \
  --dir-scope out_mask \
  --dir-frac 0.05 \
  --dir-layers 14 \
  --steer-layer 14 \
  --steer-direction shared \
  --steer-scope-region output \
  --steer-pos mask \
  --alpha-values 0.0 0.25 0.5 1.0 2.0 4.0 8.0 \
  --limit 10 \
  --device cuda \
  --no-plot

echo "===== FINISHED ====="
date
