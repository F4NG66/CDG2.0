#!/usr/bin/env bash
set -euo pipefail

TSV="${1:?Missing TSV path}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}"
SPB="${SPB:-200}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"

cd ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}
source ${RRAE_ENV_ROOT:?Set RRAE_ENV_ROOT}/bin/activate

if [ -z "${DEEPSEEK_API_KEY_1:-}" ] || [ -z "${DEEPSEEK_API_KEY_2:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY_1 or DEEPSEEK_API_KEY_2 is not set"
  exit 1
fi

if (( TASK_ID % 2 == 1 )); then
  export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY_1"
  KEY_LABEL="KEY1"
else
  export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY_2"
  KEY_LABEL="KEY2"
fi

LINE="$(awk -v n="$TASK_ID" 'NR==n+1 {print}' "$TSV")"

if [ -z "$LINE" ]; then
  echo "ERROR: no TSV row for TASK_ID=$TASK_ID"
  exit 1
fi

IFS=$'\t' read -r CANDIDATE_ID INPUT_PATH JUDGE_CONFIG ALPHA OUT_TAG <<< "$LINE"

echo "TASK_ID=$TASK_ID"
echo "KEY_LABEL=$KEY_LABEL"
echo "CANDIDATE_ID=$CANDIDATE_ID"
echo "INPUT_PATH=$INPUT_PATH"
echo "JUDGE_CONFIG=$JUDGE_CONFIG"
echo "ALPHA=$ALPHA"
echo "OUT_TAG=$OUT_TAG"
echo "SPB=$SPB"
echo "JUDGE_MODEL=$JUDGE_MODEL"

if [ ! -f "$INPUT_PATH" ]; then
  echo "ERROR: input does not exist: $INPUT_PATH"
  exit 1
fi

TMP_ROOT="tmp_p11_candidate_inputs_v2"
OUT_ROOT="steering_runs/phase3_v2/judges/P11_candidates_spb${SPB}"

mkdir -p "$TMP_ROOT" "$OUT_ROOT"

FILTERED_INPUT="${TMP_ROOT}/${OUT_TAG}__baseline_plus_alpha.jsonl"
OUT_DIR="${OUT_ROOT}/${OUT_TAG}"

python - <<PY
import json
from pathlib import Path
from collections import Counter

inp = Path("$INPUT_PATH")
out = Path("$FILTERED_INPUT")
alpha_target = float("$ALPHA")

counts = Counter()
kept = 0

with inp.open() as r, out.open("w") as w:
    for line in r:
        if not line.strip():
            continue

        obj = json.loads(line)

        if obj.get("position_config") != "$JUDGE_CONFIG":
            continue

        group = obj.get("group")
        mode = obj.get("mode")

        try:
            alpha = float(obj.get("alpha"))
        except Exception:
            continue

        keep = False

        if group in ("B", "C") and mode == "baseline" and alpha == 0.0:
            keep = True

        if group in ("B", "C") and mode == "steered" and alpha == alpha_target:
            keep = True

        if keep:
            w.write(line)
            counts[(group, mode, str(obj.get("alpha")))] += 1
            kept += 1

print("filtered rows:", kept)
for k, v in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1] != "baseline", float(x[0][2]))):
    print(k, v)

if kept == 0:
    raise SystemExit("ERROR: filtered input is empty")
PY

mkdir -p "$OUT_DIR"

python scripts/phase3_v2/judges/p11/evaluate_rrae_with_p11_judge.py \
  --input "$FILTERED_INPUT" \
  --out-dir "$OUT_DIR" \
  --configs "$JUDGE_CONFIG" \
  --groups B C \
  --sample-per-bucket "$SPB" \
  --judge-model "$JUDGE_MODEL"

echo "DONE: $CANDIDATE_ID"
