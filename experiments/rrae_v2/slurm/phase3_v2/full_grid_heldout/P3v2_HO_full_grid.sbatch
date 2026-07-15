#!/usr/bin/env bash
#SBATCH --job-name=P3HO_GRID
#SBATCH --account=def-zshakeri
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --array=0-191%8
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORK_ROOT="${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
ENV_PATH="${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}"
MANIFEST="$WORK_ROOT/jobs/phase3_v2/full_grid_heldout/config_manifest.tsv"

NUM_SHARDS=8
TASK_ID="${SLURM_ARRAY_TASK_ID}"

# 8 shards لكل configuration.
CONFIG_INDEX=$((TASK_ID / NUM_SHARDS))
SHARD_INDEX=$((TASK_ID % NUM_SHARDS))

CONFIG_LINE=$(sed -n "$((CONFIG_INDEX + 1))p" "$MANIFEST")

if [[ -z "$CONFIG_LINE" ]]; then
    echo "ERROR: no manifest line for CONFIG_INDEX=$CONFIG_INDEX"
    exit 1
fi

IFS=$'\t' read -r \
    LAYER \
    RANK \
    CONFIG_NAME \
    POSITION_SPEC \
    VECTOR_PATH \
    <<< "$CONFIG_LINE"

OUT_DIR="$WORK_ROOT/steering_runs/phase3_v2/full_grid_heldout/L${LAYER}_rank${RANK}/${CONFIG_NAME}"

cd "$WORK_ROOT"
source "$ENV_PATH/bin/activate"

export PYTHONPATH="${RRAE_DATA_ROOT:?Set RRAE_DATA_ROOT}:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Phase 3 v2 held-out FULL GRID"
echo "array task: $TASK_ID"
echo "config index: $CONFIG_INDEX / 23"
echo "shard index: $SHARD_INDEX / $((NUM_SHARDS - 1))"
echo "layer: L$LAYER"
echo "rank: $RANK"
echo "configuration: $CONFIG_NAME"
echo "position spec: $POSITION_SPEC"
echo "alphas: -1 -2 -4 -8 -12 -16"
echo "vector: $VECTOR_PATH"
echo "output: $OUT_DIR"
echo "host: $(hostname)"
echo "start: $(date)"
echo "============================================================"

python scripts/phase3_v2/run_rrae_steering_generation_matrix_v2.py \
  --prompt-root \
  data/phase3_prompt_roots/abcd_test_heldout_v1 \
  --model-path \
  ${LLADA_MODEL_PATH:?Set LLADA_MODEL_PATH} \
  --vector-path \
  "$VECTOR_PATH" \
  --out-dir \
  "$OUT_DIR" \
  --layer \
  "$LAYER" \
  --rank \
  "$RANK" \
  --position-spec \
  "$POSITION_SPEC" \
  --groups A B C D \
  --limit-per-group 0 \
  --alphas -1 -2 -4 -8 -12 -16 \
  --seed 0 \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --device cuda

echo "============================================================"
echo "COMPLETED"
echo "L${LAYER} rank${RANK} $CONFIG_NAME shard${SHARD_INDEX}"
echo "end: $(date)"
echo "============================================================"
