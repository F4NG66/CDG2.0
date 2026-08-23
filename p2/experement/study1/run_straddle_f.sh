#!/bin/bash
# Study 1 PART F2: straddler yield pass -- all 25 expansion A-ids x K=32 @ temp 0.2
# (paper config). GENERATE ONLY, no hidden capture. Detached, offline, key-free.
# Resumes over the Part B/C samples already on disk, so only the missing seeds run.
set -uo pipefail

OUT=/scratch/ore99/study1_straddle
LOG=$OUT/straddle_f.log
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
  echo "=== study1 PART F2: straddler yield pass (generate-only, no capture) ==="
  date -Is
  echo "host=$(hostname) MIG=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi -L
  echo "25 A-ids (A000-A024) x K=32 @ temp 0.2, seeds 1000-1031"
  echo "out=$OUT"
  echo
  "$PY" /home/ore99/experement/study1/straddle_pilot.py \
      --n-ids 25 \
      --temps 0.2 --ks 32 \
      --det-check-ids 0 \
      --out "$OUT"
  rc=$?
  echo
  echo "=== exit rc=$rc ==="
  date -Is
  du -sh "$OUT"
  exit $rc
} >>"$LOG" 2>&1
