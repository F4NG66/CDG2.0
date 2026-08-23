#!/usr/bin/env bash
set -uo pipefail
cd /home/ore99/serverFiles
PY=/home/ore99/env_llada/bin/python3
OUT=dijawithprefill/runs/full
MODEL=deepseek-v4-flash
MISSING="mid_3 both_1 both_2 both_3 start_1"
echo "[resume] START $(date)  arms: $MISSING"
for arm in $MISSING; do
  if [ -s "$OUT/$arm/graded_judge.jsonl" ]; then
    echo "[resume] SKIP $arm (already graded)"; continue
  fi
  echo "[resume] grading $arm $(date)"
  $PY dijawithprefill/graded_judge.py \
      --manifest "$OUT/$arm/manifest.jsonl" \
      --out      "$OUT/$arm/graded_judge.jsonl" \
      --groups B,D --model "$MODEL" \
    && echo "[resume] done $arm" || echo "[resume] FAIL $arm"
done
echo "[resume] ALL DONE $(date)"
