#!/bin/bash
#SBATCH --job-name=dream_p10_neg
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dream_p10_neg_%j.out
#SBATCH --error=logs/dream_p10_neg_%j.err

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

echo "===== RUN DREAM P10 NEGATIVE ALPHA ====="

python scripts/p10_steer2.py \
  --task dose_response \
  --backend dream_attack \
  --out-dir outputs_dream_native_history_full \
  --dir-scope out_mask \
  --dir-frac 0.05 \
  --dir-layers 14 \
  --steer-layer 14 \
  --steer-scope-region output \
  --steer-pos mask \
  --steer-direction shared \
  --tag dream_outmask_L14_neg \
  --limit 30 \
  --alpha-values 0.0 -0.5 -1.0 -2.0 \
  --no-plot

echo "===== FINISHED ====="
date
