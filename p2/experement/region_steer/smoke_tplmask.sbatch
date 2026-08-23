#!/bin/bash
#SBATCH --job-name=cdg_tm_smoke
#SBATCH --account=def-zshakeri_gpu
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=/home/ore99/experement/region_steer/batch_tplmask/smoke-%j.out
#SBATCH --error=/home/ore99/experement/region_steer/batch_tplmask/smoke-%j.out
set -u
module load StdEnv/2023 2>/dev/null || true
export HF_HOME=/scratch/ore99/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
PY=/scratch/ore99/cdg_venv/bin/python
cd /home/ore99/experement/region_steer || exit 3
echo "[smoke.sbatch] START $(date) node=$SLURMD_NODENAME job=$SLURM_JOB_ID CVD=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
$PY smoke_tplmask.py
RC=$?
echo "[smoke.sbatch] rc=$RC DONE $(date)"
exit $RC
