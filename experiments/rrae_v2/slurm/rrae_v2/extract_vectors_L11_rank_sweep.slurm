#!/bin/bash
#SBATCH --job-name=vecV2L11sw
#SBATCH --account=def-zshakeri
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}
source ${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate

RANKS=(8 16 24 32 48 64)
RANK="${RANKS[$SLURM_ARRAY_TASK_ID]}"

python scripts/extract_rrae_v2_vectors.py \
  --dataset ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/hidden_states/abcd_v2/H_input_region_L11_f0p05.pt \
  --ae ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/rrae_runs_v2/input_region_L11_f0p05_rank${RANK}/best_model.pt \
  --out-dir ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_vectors_v2 \
  --scope input_region \
  --layer 11 \
  --ranks ${RANK} \
  --tag abcd_v2_clean_N2000
