#!/usr/bin/env bash
#SBATCH --job-name=CL11l2p2
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}"
source ${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate

if [ -z "${DEEPSEEK_API_KEY_1:-}" ] || [ -z "${DEEPSEEK_API_KEY_2:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY_1 or DEEPSEEK_API_KEY_2 is not set"
  exit 1
fi

export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY_1"

echo "========================================"
echo "TASK=C"
echo "PASS=2"
echo "INPUT=${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/data/phase3_judge_inputs/p11_best_L11_lambda2/C_baseline_plus_L11_lambda2_N200.jsonl"
echo "OUT=${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/judges/p11_best_L11_lambda2/C_utility_pass2"
echo "MODEL=deepseek-v4-flash"
echo "START=$(date)"
echo "========================================"

mkdir -p "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/judges/p11_best_L11_lambda2/C_utility_pass2"

python "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/scripts/phase3_v2/judges/benign_utility/evaluate_benign_utility_collapse_judge.py" \
  --input "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/data/phase3_judge_inputs/p11_best_L11_lambda2/C_baseline_plus_L11_lambda2_N200.jsonl" \
  --out-dir "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/judges/p11_best_L11_lambda2/C_utility_pass2" \
  --configs custom_template_mask \
  --sample-per-bucket 100 \
  --judge-model deepseek-v4-flash

echo "========================================"
echo "DONE"
echo "FINISH=$(date)"
echo "========================================"
