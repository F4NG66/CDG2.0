#!/bin/bash
# Study 1 PART C1+C2: bound-tightening (K=32 @ 0.2) + temperature sweep (K=16 @ 0.4, 0.7).
# GENERATE ONLY, no hidden capture. ONE model load for all three temperatures.
# Detached (setsid nohup + flock), offline, key-free. Resumes over the Part B samples:
# the 8 existing temp=0.2 seeds per A-id are skipped, so only the new draws are generated.
set -uo pipefail

OUT=/scratch/ore99/study1_straddle
LOG=$OUT/straddle_c.log
LOCK=$OUT/.straddle.lock
PY=/home/ore99/env_llada/bin/python
IDS=A000,A001,A002,A003,A004          # the 5 one-sided A-ids (4 delivered-side, 1 not-side)

mkdir -p "$OUT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset DEEPSEEK_API_KEY          # generation job stays key-free; judging is separate
export TOKENIZERS_PARALLELISM=false

exec 9>"$LOCK"
flock -n 9 || { echo "another straddle run holds $LOCK -- abort"; exit 3; }

{
  echo "=== study1 PART C: bound-tightening + temperature sweep (generate-only) ==="
  date -Is
  echo "host=$(hostname) MIG=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi -L
  echo "ids=$IDS  schedule=(0.2,K=32) (0.4,K=16) (0.7,K=16)"
  echo "out=$OUT"
  echo
  "$PY" /home/ore99/experement/study1/straddle_pilot.py \
      --ids "$IDS" \
      --temps 0.2,0.4,0.7 --ks 32,16,16 \
      --det-check-ids 0 \
      --out "$OUT"
  rc=$?
  echo
  echo "=== exit rc=$rc ==="
  date -Is
  du -sh "$OUT"
  exit $rc
} >>"$LOG" 2>&1
