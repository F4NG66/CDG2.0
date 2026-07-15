#!/bin/bash
#SBATCH --job-name=rraeV2L11sw
#SBATCH --account=def-zshakeri
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}
source ${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate

RANKS=(8 16 24 32 48 64)
RANK="${RANKS[$SLURM_ARRAY_TASK_ID]}"

echo "===== RRAE V2 TRAIN ====="
echo "layer: L11"
echo "rank: ${RANK}"

python scripts/train_rrae_v2.py \
  --dataset ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/hidden_states/abcd_v2/H_input_region_L11_f0p05.pt \
  --out-dir ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/rrae_runs_v2/input_region_L11_f0p05_rank${RANK} \
  --rank-k "${RANK}" \
  --epochs 2000 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed 0
