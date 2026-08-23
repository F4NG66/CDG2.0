#!/bin/bash
# Study 1 PART B: straddler yield pilot -- GENERATE ONLY, no hidden capture.
# Detached runner (setsid nohup + flock), one model load, offline, key-free.
set -uo pipefail

OUT=/scratch/ore99/study1_straddle
LOG=$OUT/straddle.log
LOCK=$OUT/.straddle.lock
PY=/home/ore99/env_llada/bin/python

mkdir -p "$OUT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset DEEPSEEK_API_KEY          # generation job stays key-free; judging is separate
export TOKENIZERS_PARALLELISM=false

exec 9>"$LOCK"
flock -n 9 || { echo "another straddle run holds $LOCK -- abort"; exit 3; }

{
  echo "=== study1 straddler yield pilot (generate-only) ==="
  date -Is
  echo "host=$(hostname) MIG=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi -L
  echo "out=$OUT"
  echo
  "$PY" /home/ore99/experement/study1/straddle_pilot.py \
      --n-ids 15 --k 8 --temp 0.2 \
      --det-check 2 --det-check-ids 5 \
      --out "$OUT"
  rc=$?
  echo
  echo "=== exit rc=$rc ==="
  date -Is
  du -sh "$OUT"
  exit $rc
} >>"$LOG" 2>&1
