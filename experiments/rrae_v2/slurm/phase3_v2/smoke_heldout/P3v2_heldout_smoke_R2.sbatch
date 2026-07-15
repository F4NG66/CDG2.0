#!/usr/bin/env bash
#SBATCH --job-name=P3v2_HO_smoke_R2
#SBATCH --account=def-zshakeri
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORK_ROOT="${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
ENV_PATH="${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}"

cd "$WORK_ROOT"
source "$ENV_PATH/bin/activate"

export PYTHONPATH="${RRAE_DATA_ROOT:?Set RRAE_DATA_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "Phase 3 v2 held-out smoke generation"
echo "config: R2_template_mask_plus_output_mask"
echo "position_spec: template:mask+output:mask"
echo "vector: L16 rank48"
echo "groups: A B C D"
echo "limit_per_group: 2"
echo "steered alpha: -1.0"
echo "baseline: generated automatically"
echo "host: $(hostname)"
echo "time: $(date)"
echo "============================================================"

python "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/scripts/phase3_v2/run_rrae_steering_generation_matrix_v2.py" \
  --prompt-root "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/data/phase3_prompt_roots/abcd_test_heldout_v1" \
  --model-path "${LLADA_MODEL_PATH:?Set LLADA_MODEL_PATH}" \
  --vector-path "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_vectors_v2/rrae_v_injection__scope-input_region__layer-L16__rank-r48__source-residual_BA_plus_CD__abcd_v2_clean_N2000.pt" \
  --out-dir "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/smoke_heldout/L16_rank48/R2_template_mask_plus_output_mask" \
  --layer 16 \
  --rank 48 \
  --position-spec "template:mask+output:mask" \
  --groups A B C D \
  --limit-per-group 2 \
  --alphas -1.0 \
  --seed 0 \
  --num-shards 1 \
  --shard-index 0 \
  --device cuda

echo "============================================================"
echo "DONE"
echo "time: $(date)"
echo "============================================================"
