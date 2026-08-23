#!/bin/bash
#SBATCH --job-name=harmdir_steer
#SBATCH --account=def-zshakeri_gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=/home/ore99/experement/harm_dir/logs/slurm-%j.out
#SBATCH --error=/home/ore99/experement/harm_dir/logs/slurm-%j.out
# ---------------------------------------------------------------------------
# PHASE 3 PILOT - causal steering test for v_harm_ds (h -> h - alpha*v_hat at L16).
# LIMIT=1 for the n=2 smoke (1 harmful + 1 benign). LIMIT=0 = full pilot.
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
OUT=${OUT:-$DIR/data/steer_pilot.jsonl}

echo "=== harm_dir steer_pilot | LIMIT=$LIMIT | OUT=$OUT | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$DIR/scripts" || exit 1
$PY steer_pilot.py --limit "$LIMIT" --out "$OUT"
rc=$?

echo "=== exit=$rc | $(date) ==="
if [ $rc -ne 0 ]; then
  echo "STEER PILOT FAILED - not writing a success marker"
  exit $rc
fi
$PY - "$OUT" <<'EOF'
import sys, json, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
assert rows, "pilot produced 0 rows"
by = collections.Counter((r["group"], r["arm"]) for r in rows)
print(f"VERIFY ok: {len(rows)} generations; cells={dict(by)}")
EOF
exit $?
