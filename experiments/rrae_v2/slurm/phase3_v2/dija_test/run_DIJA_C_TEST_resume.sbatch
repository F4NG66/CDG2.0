#!/bin/bash
#SBATCH --job-name=DIJA_C_TEST
#SBATCH --account=def-zshakeri
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module load python/3.11
module load arrow/24.0.0

source ${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate

cd ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}

python scripts/phase3_v2/run_DIJA_refine_resume.py \
  --hf-model-path ${DIJA_ROOT:?Set DIJA_ROOT}/hf_models/Qwen2.5-7B-Instruct \
  --prompt-template-path ${DIJA_ROOT:?Set DIJA_ROOT}/run_harmbench/refine_prompt/redteam_prompt_template.txt \
  --attack-prompt data/TEST_HELDOUT_V1/dija_inputs/D_TEST_for_DIJA_refine.csv \
  --output-jsonl data/TEST_HELDOUT_V1/dija_raw/C_TEST_DIJA_Qwen_resume.jsonl \
  --output-json data/TEST_HELDOUT_V1/dija_raw/C_TEST_DIJA_Qwen_raw.json \
  --max-new-tokens 200
