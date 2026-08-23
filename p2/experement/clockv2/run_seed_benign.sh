#!/bin/bash
# clockv2/run_seed_benign.sh — last seed gap: ARM 2 (benign, dataset-sourced) at seeds 1,2.
#
# Why: arm 2 carries the two-arm-agreement argument that mitigates the authored-vs-derived
# confound in §7 (arm 2 dataset-sourced vs arm 3 authored, agreeing on the content term).
# That mitigation was the last load-bearing claim still resting on a single temp=0 run.
#
# Launch detached:
#   cd /home/ore99/experement/clockv2
#   setsid nohup bash run_seed_benign.sh > results/seed_benign.log 2>&1 < /dev/null &
#
# Same guards: flock (one instance ever), sanitize-before-resume, headroom check, retry.
# Does NOT touch v_refusal or tau — capture_seed.py projects only v_injection_svd and
# v_random_null and never loads the refusal probe.

set -u
cd /home/ore99/experement/clockv2

LOCK=/tmp/clockv2_seed_sweep.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[seed_benign] ANOTHER INSTANCE HOLDS THE LOCK ($LOCK) — refusing to start."
  exit 1
fi

PY=/home/ore99/env_llada/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG_TAG="[seed_benign]"
NEED_MIB=26000

echo "$LOG_TAG start $(date +%F_%H:%M:%S) host=$(hostname) pid=$$ ppid=$PPID"
echo "$LOG_TAG CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "$LOG_TAG lock acquired: $LOCK"

free_mib() {
  "$PY" - <<'EOF'
import torch
free, total = torch.cuda.mem_get_info(0)
print(int(free / 1024**2))
EOF
}

sanitize() {
  local f="$1"
  [ -f "$f" ] || return 0
  if ! tail -1 "$f" | "$PY" -c "import sys,json; json.loads(sys.stdin.read())" 2>/dev/null; then
    echo "$LOG_TAG sanitize: dropping incomplete trailing line of $f"
    head -n -1 "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
}

for S in 1 2; do
  C=benign
  F="results/probe_readings_s${S}_${C}.jsonl"
  for ATTEMPT in 1 2 3; do
    sanitize "$F"
    HAVE=$(free_mib 2>/dev/null || echo 0)
    if [ "${HAVE:-0}" -lt "$NEED_MIB" ]; then
      echo "$LOG_TAG seed=$S cond=$C attempt=$ATTEMPT: only ${HAVE}MiB free (<${NEED_MIB}); waiting 120s"
      sleep 120
      continue
    fi
    echo "$LOG_TAG === seed=$S cond=$C attempt=$ATTEMPT start $(date +%H:%M:%S) free=${HAVE}MiB ==="
    "$PY" capture_seed.py --cond "$C" --seed "$S" --limit 0 2>&1 \
      | grep -vE "it/s\]|FutureWarning|warnings.warn|Special tokens"
    ROWS=$(wc -l < "$F" 2>/dev/null || echo 0)
    echo "$LOG_TAG === seed=$S cond=$C attempt=$ATTEMPT end $(date +%H:%M:%S) rows=$ROWS/38400 ==="
    [ "$ROWS" -eq 38400 ] && break
    echo "$LOG_TAG incomplete ($ROWS/38400) — retrying after 30s"
    sleep 30
  done
  ROWS=$(wc -l < "$F" 2>/dev/null || echo 0)
  if [ "$ROWS" -ne 38400 ]; then
    echo "$LOG_TAG FATAL: seed=$S cond=$C stuck at $ROWS/38400. Stopping."
    exit 2
  fi
done

echo "$LOG_TAG row counts:"
wc -l results/probe_readings_s*_*.jsonl
echo "$LOG_TAG ALL_DONE $(date +%F_%H:%M:%S)"
