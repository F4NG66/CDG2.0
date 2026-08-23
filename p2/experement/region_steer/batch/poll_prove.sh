#!/bin/bash
D=/home/ore99/experement/region_steer/batch
JID=$(cat "$D/JOBID")
for i in $(seq 1 240); do
  ST=$(squeue -j "$JID" -h -o %T 2>/dev/null)
  ROWS=$(wc -l < "$D/gen.jsonl" 2>/dev/null || echo 0); ROWS=${ROWS:-0}
  printf '[poll %3d] %s state=%-10s rows=%s\n' "$i" "$(date +%H:%M:%S)" "${ST:-GONE}" "$ROWS"
  if [ "$ROWS" -ge 6 ]; then echo "RESULT=GENERATING rows=$ROWS"; exit 0; fi
  if [ -z "$ST" ]; then echo "RESULT=JOB_LEFT_QUEUE rows=$ROWS (inspect $D/slurm-$JID.out)"; exit 2; fi
  sleep 15
done
echo "RESULT=TIMEOUT last_rows=$ROWS"; exit 3
