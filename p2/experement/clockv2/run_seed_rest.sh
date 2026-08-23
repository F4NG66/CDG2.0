#!/bin/bash
# clockv2/run_seed_rest.sh — resume the seed sweep, disconnect-proof and collision-proof.
#
# Remaining work at temp=0.2 (s1_dija is COMPLETE at 38,400 rows and is not re-run):
#   s1_benign_op  — resumes at A071 (A000..A070 already captured, verified clean)
#   s2_dija       — from scratch
#   s2_benign_op  — from scratch
#
# Launch detached (survives the interactive session dropping):
#   cd /home/ore99/experement/clockv2
#   setsid nohup bash run_seed_rest.sh > results/seed_sweep.log 2>&1 < /dev/null &
#
# LOCK: flock guarantees only ONE instance ever runs. The previous OOM was caused by two
# capture processes loading the model onto the same MIG slice concurrently (23.4GB + 16GB
# on a 40GB slice). This makes that failure mode unreachable rather than merely unlikely.
#
# SHARED SLICE: CUDA_VISIBLE_DEVICES (MIG-433e3836) is shared with OTHER USERS' jobs
# (~9.5GB resident at time of writing, not ours — do not kill them). ~30GB is free, which
# fits exactly one capture. Hence the lock, the retry, and the pre-flight headroom check.

set -u
cd /home/ore99/experement/clockv2

LOCK=/tmp/clockv2_seed_sweep.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[seed_rest] ANOTHER INSTANCE HOLDS THE LOCK ($LOCK) — refusing to start."
  echo "[seed_rest] This is the guard against the concurrent-model-load OOM. Exiting."
  exit 1
fi

PY=/home/ore99/env_llada/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on a shared slice
LOG_TAG="[seed_rest]"
NEED_MIB=26000        # one capture peaked at ~23.4GB; require headroom before starting

echo "$LOG_TAG start $(date +%F_%H:%M:%S) host=$(hostname) pid=$$ ppid=$PPID"
echo "$LOG_TAG CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "$LOG_TAG lock acquired: $LOCK"

free_mib() {   # MIG-aware free memory on OUR slice
  "$PY" - <<'EOF'
import torch
free, total = torch.cuda.mem_get_info(0)
print(int(free / 1024**2))
EOF
}

sanitize() {   # drop a truncated trailing line so --resume can't append after it
  local f="$1"
  [ -f "$f" ] || return 0
  if ! tail -1 "$f" | "$PY" -c "import sys,json; json.loads(sys.stdin.read())" 2>/dev/null; then
    echo "$LOG_TAG sanitize: dropping incomplete trailing line of $f"
    head -n -1 "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
}

for SPEC in "1 benign_op" "2 dija" "2 benign_op"; do
  set -- $SPEC
  S=$1; C=$2
  F="results/probe_readings_s${S}_${C}.jsonl"

  for ATTEMPT in 1 2 3 4 5; do
    sanitize "$F"
    HAVE=$(free_mib 2>/dev/null || echo 0)
    if [ "${HAVE:-0}" -lt "$NEED_MIB" ]; then
      echo "$LOG_TAG seed=$S cond=$C attempt=$ATTEMPT: only ${HAVE}MiB free (<${NEED_MIB}); neighbours busy, waiting 180s"
      sleep 180
      continue
    fi
    echo "$LOG_TAG === seed=$S cond=$C attempt=$ATTEMPT start $(date +%H:%M:%S) free=${HAVE}MiB ==="
    # no --overwrite: resumes, skipping ids already present
    "$PY" capture_seed.py --cond "$C" --seed "$S" --limit 0 2>&1 \
      | grep -vE "it/s\]|FutureWarning|warnings.warn|Special tokens"
    ROWS=$(wc -l < "$F" 2>/dev/null || echo 0)
    echo "$LOG_TAG === seed=$S cond=$C attempt=$ATTEMPT end $(date +%H:%M:%S) rows=$ROWS/38400 ==="
    [ "$ROWS" -eq 38400 ] && break
    echo "$LOG_TAG seed=$S cond=$C incomplete ($ROWS/38400) — retrying after 60s"
    sleep 60
  done

  ROWS=$(wc -l < "$F" 2>/dev/null || echo 0)
  if [ "$ROWS" -ne 38400 ]; then
    echo "$LOG_TAG FATAL: seed=$S cond=$C stuck at $ROWS/38400 after retries. Stopping."
    exit 2
  fi
done

echo "$LOG_TAG row counts (each must be 38400):"
wc -l results/probe_readings_s*_*.jsonl
echo "$LOG_TAG ALL_DONE $(date +%F_%H:%M:%S)"
