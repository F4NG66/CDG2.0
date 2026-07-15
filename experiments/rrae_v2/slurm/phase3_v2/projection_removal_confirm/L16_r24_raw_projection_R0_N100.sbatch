#!/usr/bin/env bash
#SBATCH --job-name=PR100_L16
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
source "${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate"

export PYTHONPATH="${RRAE_DATA_ROOT:?Set RRAE_DATA_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "Projection-removal confirmation"
echo "layer: L16"
echo "rank: 24"
echo "groups: B C"
echo "limit per group: 100"
echo "lambdas: 1.0 1.5 2.0"
echo "vector: ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_vectors_v2/vector_source_ablation/rrae_vector_ablation__source-raw_BA_plus_CD__layer-L16__rank-r24__train-v2.pt"
echo "output: ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/projection_removal_confirm/L16_r24_raw_projection_R0_N100"
echo "start: $(date)"
echo "============================================================"

python "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/scripts/phase3_v2/run_projection_removal_generation_matrix_v2.py" \
  --prompt-root "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/data/phase3_prompt_roots/abcd_test_heldout_v1" \
  --vector-path "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_vectors_v2/vector_source_ablation/rrae_vector_ablation__source-raw_BA_plus_CD__layer-L16__rank-r24__train-v2.pt" \
  --out-dir "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/projection_removal_confirm/L16_r24_raw_projection_R0_N100" \
  --layer 16 \
  --rank 24 \
  --position-spec template:mask \
  --groups B C \
  --limit-per-group 100 \
  --alphas 1.0 1.5 2.0 \
  --seed 0 \
  --num-shards 1 \
  --shard-index 0 \
  --device cuda

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
