#!/bin/bash
#SBATCH --job-name=harmdir_steer2
#SBATCH --account=def-zshakeri_gpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=/home/ore99/experement/harm_dir/logs/slurm-%j.out
#SBATCH --error=/home/ore99/experement/harm_dir/logs/slurm-%j.out
# ---------------------------------------------------------------------------
# PHASE 3.1 - scaffold-region steering (Arm S) + steer-early/remask-late (Arm SR).
# LIMIT=2 for the smoke (2 harmful + 1 benign). LIMIT=0 = full run (needs sign-off).
# ---------------------------------------------------------------------------
set -u
module load StdEnv/2023 2>/dev/null || true
export HF_HOME=/scratch/ore99/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PY=/scratch/ore99/cdg_venv/bin/python
DIR=/home/ore99/experement/harm_dir
LIMIT=${LIMIT:-0}
BENIGN=${BENIGN:-20}
ALPHAS=${ALPHAS:-16,32,48,64}
WITH_C=${WITH_C:-1}          # 1 = add C_neutral_injected benign-injected control
OUT=${OUT:-$DIR/data/steer_pilot2.jsonl}
CFLAG=""; [ "$WITH_C" = "1" ] && CFLAG="--with-c"

echo "=== harm_dir steer_pilot2 | LIMIT=$LIMIT | BENIGN=$BENIGN | ALPHAS=$ALPHAS | WITH_C=$WITH_C | OUT=$OUT | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$DIR/scripts" || exit 1
$PY steer_pilot2.py --limit "$LIMIT" --benign-target "$BENIGN" --alphas "$ALPHAS" $CFLAG --out "$OUT"
rc=$?

echo "=== exit=$rc | $(date) ==="
if [ $rc -ne 0 ]; then
  echo "STEER PILOT2 FAILED - not writing a success marker"
  exit $rc
fi
$PY - "$OUT" <<'EOF'
import sys, json, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
assert rows, "pilot2 produced 0 rows"
by = collections.Counter((r["group"], r["mode"], r["arm"]) for r in rows)
deg = sum(1 for r in rows if r.get("degenerate"))
print(f"VERIFY ok: {len(rows)} generations; degenerate={deg}; cells={dict(by)}")
EOF
exit $?
