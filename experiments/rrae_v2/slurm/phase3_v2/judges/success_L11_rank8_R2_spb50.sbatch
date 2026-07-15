#!/usr/bin/env bash
#SBATCH --job-name=Jsucc_L11r8R2
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

WORK_ROOT="${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
ENV_PATH="${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}"

cd "$WORK_ROOT"
source "$ENV_PATH/bin/activate"

export PYTHONPATH="${RRAE_DATA_ROOT:?Set RRAE_DATA_ROOT}:${PYTHONPATH:-}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set"
    exit 1
fi

INPUT="steering_runs/phase3_v2/full_grid_heldout_merged/L11_rank8/R2_template_mask_plus_output_mask/merged_all_configs__rank8__L11.jsonl"

OUT_DIR="steering_runs/phase3_v2/judges/success_spb50/L11_rank8_R2_template_mask_plus_output_mask"

python scripts/phase3_v2/judges/evaluate_success_judge_v2.py \
  --input "$INPUT" \
  --out-dir "$OUT_DIR" \
  --configs custom_template_mask_plus_output_mask \
  --groups B C \
  --judge-template injection \
  --judge-model deepseek-v4-flash \
  --sample-per-bucket 50
