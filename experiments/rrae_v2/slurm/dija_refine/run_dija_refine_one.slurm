#!/bin/bash
#SBATCH --job-name=DIJA_refine
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd ${DIJA_ROOT:?Set DIJA_ROOT}/run_harmbench/refine_prompt
source ${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate

if [ -z "${INPUT_CSV:-}" ]; then
  echo "ERROR: INPUT_CSV missing"
  exit 1
fi

if [ -z "${OUTPUT_JSON:-}" ]; then
  echo "ERROR: OUTPUT_JSON missing"
  exit 1
fi

HF_MODEL_PATH="${DIJA_ROOT:?Set DIJA_ROOT}/hf_models/Qwen2.5-7B-Instruct"
TEMPLATE_PATH="${DIJA_ROOT:?Set DIJA_ROOT}/run_harmbench/refine_prompt/redteam_prompt_template.txt"

echo "===== DIJA refine job ====="
echo "HOSTNAME=$HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "INPUT_CSV=$INPUT_CSV"
echo "OUTPUT_JSON=$OUTPUT_JSON"
echo "HF_MODEL_PATH=$HF_MODEL_PATH"
echo "TEMPLATE_PATH=$TEMPLATE_PATH"
echo "TIME=$(date)"
echo

python main.py \
  --hf_model_path "$HF_MODEL_PATH" \
  --prompt_template_path "$TEMPLATE_PATH" \
  --attack_prompt "$INPUT_CSV" \
  --output_json "$OUTPUT_JSON" \
  --max_new_tokens 200

echo
echo "===== output check ====="
python - <<PY
import json
from pathlib import Path

p = Path("$OUTPUT_JSON")
data = json.loads(p.read_text(encoding="utf-8"))

print("output:", p)
print("type:", type(data).__name__)
print("rows:", len(data))

mask_rows = sum(1 for r in data if "<mask:" in str(r.get("Refined_behavior", "")))
print("rows_with_maskN:", mask_rows)

assert isinstance(data, list)
assert len(data) == 500
assert mask_rows > 0
PY

echo
echo "===== done ====="
echo "TIME=$(date)"
