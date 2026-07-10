#!/bin/bash
#SBATCH --job-name=dream_harm_f1
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dream_harm_f1_%j.out
#SBATCH --error=logs/dream_harm_f1_%j.err

set -euo pipefail

cd /home/nmmd294/scratch/fang

source /home/nmmd294/hive_lab/diffusion_model/venv_llada/bin/activate

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/nmmd294/scratch/fang/runtime_stubs:/home/nmmd294/scratch/fang

echo "===== JOB INFO ====="
hostname
date
nvidia-smi || true

echo "===== RUN DREAM HARM F1 L14 MID ====="

python scripts/p10_steer2.py \
  --task dose_response \
  --backend dream_attack \
  --out-dir outputs_dream_native_history_full \
  --dir-scope harm \
  --dir-frac 1.0 \
  --dir-layers 14 \
  --steer-layer 14 \
  --steer-scope-region harm \
  --steer-pos unmask \
  --steer-direction shared \
  --tag dream_harm_f1_L14_mid \
  --limit 30 \
  --alpha-values 0.0 -0.5 -1.0 -2.0 2.0 \
  --no-plot

echo "===== FINISHED ====="
date
