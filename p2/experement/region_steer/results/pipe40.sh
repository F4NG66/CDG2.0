#!/bin/bash
cd /home/ore99/experement/region_steer
export HF_HOME=/scratch/ore99/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=MIG-74a035eb-c068-5f15-ab64-f03c63785d1c
PY=/scratch/ore99/cdg_venv/bin/python
echo "[pipe40] START $(date)"
$PY run40.py >> results/gen40.log 2>&1 || { echo "[pipe40] GEN FAILED"; exit 1; }
echo "[pipe40] gen rows: $(wc -l < results/gen40.jsonl)"
$PY judge40.py >> results/judge40.log 2>&1 || { echo "[pipe40] JUDGE FAILED"; exit 2; }
$PY analyze40.py > results/analysis40.txt 2>>results/judge40.log
cat results/analysis40.txt
echo "[pipe40] ALL DONE $(date)"
