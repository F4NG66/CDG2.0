#!/bin/bash
#SBATCH --job-name=harmdir_gensafe
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
# PHASE 2'' - LLaDA generates the matched-length SAFE arm for Gate E.
# LLaDA diffusion generation (not teacher-forcing) -> MUST be a GPU job.
# Pass LIMIT=2 for the smoke test, LIMIT=0 (default) for the full run.
# ---------------------------------------------------------------------------
set -u
module load StdEnv/2023 2>/dev/null || true
export HF_HOME=/scratch/ore99/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PY=/scratch/ore99/cdg_venv/bin/python
DIR=/home/ore99/experement/harm_dir
LIMIT=${LIMIT:-0}
OUT=${OUT:-$DIR/data/llada_safe.jsonl}
LENMULT=${LENMULT:-1.0}

echo "=== harm_dir gen_safe | LIMIT=$LIMIT | LENMULT=$LENMULT | OUT=$OUT | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$DIR/scripts" || exit 1
$PY generate_llada_safe.py --limit "$LIMIT" --out "$OUT" --len-mult "$LENMULT"
rc=$?

echo "=== exit=$rc | $(date) ==="
if [ $rc -ne 0 ]; then
  echo "GEN FAILED - not writing a success marker"
  exit $rc
fi
$PY - "$OUT" <<'EOF'
import sys, json
rows = [json.loads(l) for l in open(sys.argv[1])]
assert rows, "gen produced 0 rows"
print(f"VERIFY ok: {len(rows)} safe responses, "
      f"safe words median {sorted(r['safe_words'] for r in rows)[len(rows)//2]}")
EOF
exit $?
