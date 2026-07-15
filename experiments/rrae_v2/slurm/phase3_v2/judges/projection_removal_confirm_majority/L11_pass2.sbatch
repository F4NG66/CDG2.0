#!/usr/bin/env bash
#SBATCH --job-name=JPR_L11_p2
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
source "${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate"

export PYTHONPATH="${RRAE_DATA_ROOT:?Set RRAE_DATA_ROOT}:${PYTHONPATH:-}"

if [[ -z "${DEEPSEEK_API_KEY_1:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY_1 is not available"
    exit 1
fi

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY_1}"

echo "============================================================"
echo "target: L11"
echo "judge pass: 2"
echo "API key slot: 1"
echo "input: ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/data/phase3_judge_inputs/projection_removal_confirm/L11_steered_N600.jsonl"
echo "output: ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/judges/projection_removal_confirm_majority/L11_pass2"
echo "start: $(date)"
echo "============================================================"

python "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/scripts/phase3_v2/judges/evaluate_success_judge_v2.py" \
  --input "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/data/phase3_judge_inputs/projection_removal_confirm/L11_steered_N600.jsonl" \
  --out-dir "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/judges/projection_removal_confirm_majority/L11_pass2" \
  --configs custom_template_mask \
  --groups B C \
  --judge-template injection \
  --judge-model deepseek-v4-flash \
  --sample-per-bucket 0

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
