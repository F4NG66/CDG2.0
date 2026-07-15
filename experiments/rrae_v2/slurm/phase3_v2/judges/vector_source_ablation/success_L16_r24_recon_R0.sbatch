#!/usr/bin/env bash
#SBATCH --job-name=JVS16_recon
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

INPUT="${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/vector_source_ablation/L16_r24_recon_R0/R0_template_mask/results__custom_template_mask__rank24__L16.jsonl"
OUT_DIR="${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/steering_runs/phase3_v2/judges/vector_source_ablation/L16_r24_recon_R0"

echo "============================================================"
echo "Vector-source success judge"
echo "layer: L16"
echo "rank: 24"
echo "source: recon"
echo "configuration: custom_template_mask"
echo "groups: B C"
echo "input: $INPUT"
echo "output: $OUT_DIR"
echo "model: deepseek-v4-flash"
echo "host: $(hostname)"
echo "start: $(date)"
echo "============================================================"

python "${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}/scripts/phase3_v2/judges/evaluate_success_judge_v2.py" \
  --input "$INPUT" \
  --out-dir "$OUT_DIR" \
  --configs custom_template_mask \
  --groups B C \
  --judge-template injection \
  --judge-model deepseek-v4-flash \
  --sample-per-bucket 0

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
