#!/usr/bin/env bash
#SBATCH --job-name=VS_16_raw
#SBATCH --account=def-zshakeri
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORK_ROOT="${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
ENV_PATH="${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}"

cd "$WORK_ROOT"
source "$ENV_PATH/bin/activate"

export PYTHONPATH="${RRAE_DATA_ROOT:?Set RRAE_DATA_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "Vector-source steering ablation"
echo "run: L16_r24_raw_R0"
echo "dataset: held-out v1"
echo "layer: L16"
echo "rank label: 24"
echo "vector source: raw_BA_plus_CD"
echo "position: template:mask"
echo "groups: A B C D"
echo "limit per group: 20"
echo "alphas: -1.0 -2.0 -4.0 -8.0"
echo "seed: 0"
echo "host: $(hostname)"
echo "start: $(date)"
echo "============================================================"

python "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/scripts/phase3_v2/run_rrae_steering_generation_matrix_v2.py" \
  --prompt-root "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/data/phase3_prompt_roots/abcd_test_heldout_v1" \
  --model-path "${LLADA_MODEL_PATH:?Set LLADA_MODEL_PATH}" \
  --vector-path "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_vectors_v2/vector_source_ablation/rrae_vector_ablation__source-raw_BA_plus_CD__layer-L16__rank-r24__train-v2.pt" \
  --out-dir "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/vector_source_ablation/L16_r24_raw_R0/R0_template_mask" \
  --layer 16 \
  --rank 24 \
  --position-spec "template:mask" \
  --groups A B C D \
  --limit-per-group 20 \
  --alphas -1.0 -2.0 -4.0 -8.0 \
  --seed 0 \
  --num-shards 1 \
  --shard-index 0 \
  --device cuda

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
