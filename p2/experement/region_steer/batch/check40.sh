#!/bin/bash
# ONE command to check/finish the powered confirmation batch job.
#  - if analysis is done -> print it
#  - if generation is done but judging isn't (e.g. batch node had no internet) -> finish
#    judge+analyze here (run this on a node where api.deepseek.com is reachable, e.g. login)
#  - else -> show progress + job state
D=/home/ore99/experement/region_steer/batch
PY=/scratch/ore99/cdg_venv/bin/python
EXPECT=1040
cd /home/ore99/experement/region_steer || exit 3
export HF_HOME=/scratch/ore99/hf_cache

if [ -s "$D/analysis.txt" ]; then
  cat "$D/analysis.txt"; exit 0
fi
NG=$(wc -l < "$D/gen.jsonl" 2>/dev/null || echo 0)
if [ "${NG:-0}" -ge "$EXPECT" ]; then
  echo "[check] generation complete ($NG rows); finishing judge+analyze here ..."
  NJ=$(wc -l < "$D/judged.jsonl" 2>/dev/null || echo 0)
  if [ "${NJ:-0}" -lt "$EXPECT" ]; then $PY judge40.py --gen "$D/gen.jsonl" --out "$D/judged.jsonl"; fi
  $PY analyze40.py --judged "$D/judged.jsonl" | tee "$D/analysis.txt"
else
  echo "[check] not finished: $NG/$EXPECT generations written."
  [ -f "$D/JOBID" ] && { echo "[check] job state:"; squeue -j "$(cat "$D/JOBID")" 2>/dev/null || echo "  (job not in queue — check $D/slurm-*.out)"; }
fi
