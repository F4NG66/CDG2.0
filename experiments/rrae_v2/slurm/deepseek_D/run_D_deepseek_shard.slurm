#!/bin/bash
#SBATCH --job-name=D_deepseek
#SBATCH --account=def-zshakeri
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}
source ${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate

if [ -z "${SHARD_NAME:-}" ]; then
  echo "ERROR: SHARD_NAME is missing"
  exit 1
fi

if [ -z "${START:-}" ]; then
  echo "ERROR: START is missing"
  exit 1
fi

if [ -z "${N:-}" ]; then
  echo "ERROR: N is missing"
  exit 1
fi

if [ -z "${KEY_FILE:-}" ]; then
  echo "ERROR: KEY_FILE is missing"
  exit 1
fi

if [ ! -f "$KEY_FILE" ]; then
  echo "ERROR: KEY_FILE not found: $KEY_FILE"
  exit 1
fi

export DEEPSEEK_API_KEY="$(cat "$KEY_FILE" | tr -d '\n\r')"
export DEEPSEEK_MODEL="deepseek-chat"

echo "===== job info ====="
echo "HOSTNAME=$HOSTNAME"
echo "SHARD_NAME=$SHARD_NAME"
echo "START=$START"
echo "N=$N"
echo "MODEL=$DEEPSEEK_MODEL"
echo "TIME=$(date)"
echo

python scripts/make_group_D_from_A_deepseek.py \
  --start "$START" \
  --n "$N" \
  --temperature 0.0 \
  --top-p 0.95 \
  --max-tokens 200 \
  --sleep 0.3 \
  --out-jsonl "data/group_D/D_benign_clean_deepseek_chat_FULL_${SHARD_NAME}.jsonl" \
  --out-csv "data/group_D/D_benign_clean_deepseek_chat_FULL_${SHARD_NAME}.csv" \
  --audit-json "data/audit/D_benign_clean_deepseek_chat_FULL_${SHARD_NAME}_summary.json"

echo
echo "===== done ====="
echo "TIME=$(date)"
