#!/bin/bash
cd /home/ore99/experement/region_steer
export HF_HOME=/scratch/ore99/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/scratch/ore99/cdg_venv/bin/python
# 1) wait for generation to finish
while true; do
  if grep -q "\[run\] wrote" results/gen.log 2>/dev/null; then echo "[pipe] gen DONE"; break; fi
  if ! pgrep -f 'run_arms.py' >/dev/null; then echo "[pipe] gen process exited WITHOUT marker — aborting"; exit 1; fi
  sleep 20
done
echo "[pipe] gen rows: $(wc -l < results/gen.jsonl)"
# 2) judge (deepseek reachable here)
echo "[pipe] judging ..."
$PY judge_arms.py >> results/judge.log 2>&1 || { echo "[pipe] judge FAILED"; exit 2; }
echo "[pipe] judged rows: $(wc -l < results/judged.jsonl)"
# 3) analyze
echo "[pipe] analysis ==================="
$PY analyze.py > results/analysis.txt 2>>results/judge.log
cat results/analysis.txt
echo "[pipe] ALL DONE"
