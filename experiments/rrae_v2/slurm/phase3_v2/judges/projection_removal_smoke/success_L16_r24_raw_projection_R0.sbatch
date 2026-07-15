#!/usr/bin/env bash
#SBATCH --job-name=JPR_L16
#SBATCH --account=def-zshakeri
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
source "${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate"
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before submitting this job}"

export PYTHONPATH="${RRAE_DATA_ROOT:?Set RRAE_DATA_ROOT}:${PYTHONPATH:-}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set"
    exit 1
fi

echo "============================================================"
echo "Projection-removal success judge"
echo "layer: L16"
echo "rank: 24"
echo "groups: B C"
echo "input: ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/projection_removal_smoke/L16_r24_raw_projection_R0/results__custom_template_mask__rank24__L16.jsonl"
echo "output: ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/judges/projection_removal_smoke/L16_r24_raw_projection_R0"
echo "start: $(date)"
echo "============================================================"

python "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/scripts/phase3_v2/judges/evaluate_success_judge_v2.py" \
  --input "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/projection_removal_smoke/L16_r24_raw_projection_R0/results__custom_template_mask__rank24__L16.jsonl" \
  --out-dir "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/judges/projection_removal_smoke/L16_r24_raw_projection_R0" \
  --configs custom_template_mask \
  --groups B C \
  --judge-template injection \
  --judge-model deepseek-v4-flash \
  --sample-per-bucket 0

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
