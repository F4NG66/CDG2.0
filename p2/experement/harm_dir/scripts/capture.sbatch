#!/bin/bash
#SBATCH --job-name=harmdir_cap
#SBATCH --account=def-zshakeri_gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=/home/ore99/experement/harm_dir/logs/slurm-%j.out
#SBATCH --error=/home/ore99/experement/harm_dir/logs/slurm-%j.out
# ---------------------------------------------------------------------------
# PHASE 1 - teacher-forced hidden-state capture for the harm-vs-safe direction.
# Pass LIMIT=2 for the smoke test, LIMIT=0 (default) for the full run.
# Slurm sets CUDA_VISIBLE_DEVICES for the allocated MIG - do NOT hardcode it.
# ---------------------------------------------------------------------------
set -u
module load StdEnv/2023 2>/dev/null || true
export HF_HOME=/scratch/ore99/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PY=/scratch/ore99/cdg_venv/bin/python
DIR=/home/ore99/experement/harm_dir
LIMIT=${LIMIT:-0}
OUT=${OUT:-$DIR/data/states.pt}

echo "=== harm_dir capture | LIMIT=$LIMIT | OUT=$OUT | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$DIR/scripts" || exit 1
$PY capture_states.py --limit "$LIMIT" --out "$OUT"
rc=$?

echo "=== exit=$rc | $(date) ==="
if [ $rc -ne 0 ]; then
  echo "CAPTURE FAILED - not writing a success marker"
  exit $rc
fi
# fail loud rather than let an empty capture masquerade as a completed run
$PY - "$OUT" <<'EOF'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
n = len(d["records"])
assert n > 0, "capture produced 0 records"
print(f"VERIFY ok: {n} records, layers={d['layers']}, layouts={d['layouts']}")
EOF
exit $?
