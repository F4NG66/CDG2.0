#!/bin/bash
#SBATCH --job-name=cdg_tplmask
#SBATCH --account=def-zshakeri_gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=/home/ore99/experement/region_steer/batch_tplmask/slurm-%j.out
#SBATCH --error=/home/ore99/experement/region_steer/batch_tplmask/slurm-%j.out
# ---------------------------------------------------------------------------
# TPL_MASK (TM) ARM — batch version of the EXACT n=40 locked design + one added arm.
#   B arms: E/T/O/R/Rw/TM ; C arms: E/T/O/TM. alpha in {21,42,84,126}. -v de-inject,
#   block-16, identical per-case RNG. Self-chained: generate -> judge -> analyze.
# REUSES the fixed confirm40.sbatch recipe: mkdir -p up front, bounded gen retry, and a
# hard fail-loud-on-0/short-rows guard so a dead run can NEVER masquerade as COMPLETED.
# Slurm sets CUDA_VISIBLE_DEVICES to the allocated MIG automatically — do NOT hardcode it.
# cdg_denoise.py and graded_judge.py are imported unedited.
# ---------------------------------------------------------------------------
set -u
module load StdEnv/2023 2>/dev/null || true      # cvmfs base python resolves under StdEnv/2023
export HF_HOME=/scratch/ore99/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # generation is offline + key-free
export PYTHONUNBUFFERED=1
PY=/scratch/ore99/cdg_venv/bin/python
cd /home/ore99/experement/region_steer || exit 3
OUT=/home/ore99/experement/region_steer/batch_tplmask
# ALL script-written paths live under $OUT: gen.jsonl, gen.log, judge.log, judged.jsonl,
# judged_dedup.json, analysis.txt, slurm-%j.out. Create it up front so no redirect can ever
# fail on a missing dir (run40_tplmask.py also mkdirs its --out dir).
mkdir -p "$OUT" || { echo "[sbatch] FATAL: cannot create $OUT"; exit 10; }
EXPECT=1360   # 40 cases * 34 traj each (B: E + 5 arms*4 =21 ; C: E + 3 arms*4 =13)

echo "[sbatch] START $(date) node=$SLURMD_NODENAME job=$SLURM_JOB_ID"
echo "[sbatch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2

# 1) GENERATION (GPU, offline, resumable by traj_id; writes incrementally). Bounded retry
#    guards the transient "CUDA device busy/unavailable" hiccup at model.to(cuda); run40_tplmask
#    resumes by traj_id, so retrying is safe/idempotent.
GRC=1
for attempt in 1 2 3; do
  echo "[sbatch] gen attempt $attempt $(date)"
  $PY run40_tplmask.py --out "$OUT/gen.jsonl" >> "$OUT/gen.log" 2>&1
  GRC=$?
  ROWS=$(wc -l < "$OUT/gen.jsonl" 2>/dev/null || echo 0); ROWS=${ROWS:-0}
  echo "[sbatch] gen attempt $attempt rc=$GRC rows=$ROWS"
  [ "$GRC" -eq 0 ] && break
  echo "[sbatch] gen exited non-zero (transient GPU/CUDA error?) — waiting then retrying ..."
  sleep 20
done
ROWS=$(wc -l < "$OUT/gen.jsonl" 2>/dev/null || echo 0); ROWS=${ROWS:-0}

# 2) HARD GUARD — a 0-row or short generation must NEVER masquerade as a completed job. -----
if [ "$ROWS" -eq 0 ]; then
  echo "[sbatch] FATAL: generation wrote 0 rows after $attempt attempts (see $OUT/gen.log)."
  exit 11
fi
if [ "$ROWS" -lt "$EXPECT" ]; then
  echo "[sbatch] FATAL: generation incomplete ($ROWS/$EXPECT rows) — not judging. Resubmit to resume."
  exit 12
fi
echo "[sbatch] generation COMPLETE ($ROWS/$EXPECT rows)."

# 3) JUDGE (needs api.deepseek.com; unique-text dedup; idempotent full rewrite) ----------
$PY judge40.py --gen "$OUT/gen.jsonl" --out "$OUT/judged.jsonl" >> "$OUT/judge.log" 2>&1
JRC=$?
if [ $JRC -eq 0 ]; then
  # 4) ANALYZE (TM primary + DIRECT TM-vs-T + strata + selectivity + guard + decision) ---
  $PY analyze40_tplmask.py --judged "$OUT/judged.jsonl" | tee "$OUT/analysis.txt"
  echo "[sbatch] ALL DONE $(date)"
else
  echo "[sbatch] JUDGE FAILED (rc=$JRC) — likely no outbound internet on this batch node."
  echo "[sbatch] gen.jsonl ($ROWS rows) is saved. Finish from a node with api.deepseek.com access:"
  echo "         bash $OUT/check_tplmask.sh"
fi
