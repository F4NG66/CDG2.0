#!/bin/bash
# ONE command to check/finish the TM (tpl_mask) batch job.
#  - analysis done      -> print it
#  - generation done but not judged (e.g. batch node had no internet) -> finish judge+analyze here
#                          (run on a node where api.deepseek.com is reachable)
#  - else               -> show progress + job state
D=/home/ore99/experement/region_steer/batch_tplmask
PY=/scratch/ore99/cdg_venv/bin/python
EXPECT=1360
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
  $PY analyze40_tplmask.py --judged "$D/judged.jsonl" | tee "$D/analysis.txt"
else
  echo "[check] not finished: $NG/$EXPECT generations written."
  [ -f "$D/JOBID" ] && { echo "[check] job state:"; squeue -j "$(cat "$D/JOBID")" 2>/dev/null || echo "  (job not in queue — check $D/slurm-*.out)"; }
fi
