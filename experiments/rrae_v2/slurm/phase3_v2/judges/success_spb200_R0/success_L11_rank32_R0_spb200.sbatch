#!/usr/bin/env bash
#SBATCH --job-name=Js_L11r32R0
#SBATCH --account=def-zshakeri
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
source ${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate

export PYTHONPATH="${RRAE_DATA_ROOT:?Set RRAE_DATA_ROOT}:${PYTHONPATH:-}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set"
    exit 1
fi

INPUT="steering_runs/phase3_v2/full_grid_heldout_merged/L11_rank32/R0_template_mask/merged_all_configs__rank32__L11.jsonl"

OUT_DIR="steering_runs/phase3_v2/judges/success_spb200_R0/L11_rank32_R0_template_mask"

python scripts/phase3_v2/judges/evaluate_success_judge_v2.py \
  --input "$INPUT" \
  --out-dir "$OUT_DIR" \
  --configs custom_template_mask \
  --groups B C \
  --judge-template injection \
  --judge-model deepseek-v4-flash \
  --sample-per-bucket 200
