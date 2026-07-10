#!/bin/bash
#SBATCH --job-name=dream_hist
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dream_history_%j.out
#SBATCH --error=logs/dream_history_%j.err

mkdir -p logs

source /home/nmmd294/hive_lab/diffusion_model/venv_llada/bin/activate

cd /home/nmmd294/scratch/fang

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/nmmd294/scratch/fang/runtime_stubs:/home/nmmd294/scratch/fang

echo "===== JOB INFO ====="
hostname
date
nvidia-smi

echo "===== EXISTING PROGRESS ====="
find outputs_dream_native_history_full -type f -name "*.pt" 2>/dev/null | wc -l || true
wc -l outputs_dream_native_history_full/manifest.jsonl 2>/dev/null || true

echo "===== RUN RESUME ====="
python -u dream_native_history_record_resume.py

echo "===== FINISHED ====="
date
