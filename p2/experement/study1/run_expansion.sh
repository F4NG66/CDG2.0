#!/bin/bash
# Study 1 EXPANSION capture: 25 A-ids x {dija, benign_op}, ragged out_mask-only storage.
# Detached runner (setsid nohup + flock) -- one model load, offline, no network.
set -uo pipefail

OUT=/scratch/ore99/study1_expansion
LOG=$OUT/capture.log
LOCK=$OUT/.capture.lock
PY=/home/ore99/env_llada/bin/python

mkdir -p "$OUT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset DEEPSEEK_API_KEY          # judging is login-node only; keep this job key-free
export TOKENIZERS_PARALLELISM=false

exec 9>"$LOCK"
flock -n 9 || { echo "another capture holds $LOCK -- abort"; exit 3; }

{
  echo "=== study1 expansion capture ==="
  date -Is
  echo "host=$(hostname) MIG=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi -L
  echo "out=$OUT"
  echo
  "$PY" /home/ore99/experement/study1/capture_union.py \
      --arm dija+benign_op \
      --limit 25 \
      --storage ragged \
      --out "$OUT"
  rc=$?
  echo
  echo "=== exit rc=$rc ==="
  date -Is
  du -sh "$OUT"
  exit $rc
} >>"$LOG" 2>&1
