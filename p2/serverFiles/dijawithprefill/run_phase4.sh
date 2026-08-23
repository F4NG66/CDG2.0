#!/usr/bin/env bash
# Phase 4: full DIJA+prefill run on the 8 non-redundant arms, n=100/group.
#   generation (existing runner, model loaded once) + inline binary ASR judge on B
#   (= "template-filled" rate), then the graded judge post-pass (harm-delivered
#   inclusive+strict, specificity/collapse on B; coherence/collapse on D).
# Everything lands under dijawithprefill/runs/full/. Nothing else is touched.
set -uo pipefail
cd /home/ore99/serverFiles
PY=/home/ore99/env_llada/bin/python3
ARMS="mid_0,mid_1,mid_2,mid_3,both_1,both_2,both_3,start_1"
OUT=dijawithprefill/runs/full
MODEL=deepseek-v4-flash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DEEPSEEK_API_KEY="$(cat dijawithprefill/.deepseek_key)"

echo "[phase4] START $(date)  arms=$ARMS"
echo "[phase4] === GENERATION (+ inline binary ASR judge on B) ==="
$PY -u dijawithprefill/run_arms.py \
    --arms "$ARMS" --groups B,D --out-root "$OUT" \
    --binary-judge --judge-model "$MODEL" --judge-groups B
gen_rc=$?
echo "[phase4] generation rc=$gen_rc  $(date)"

echo "[phase4] === GRADED JUDGE post-pass ==="
IFS=',' read -ra ARM_ARR <<< "$ARMS"
for arm in "${ARM_ARR[@]}"; do
    m="$OUT/$arm/manifest.jsonl"
    if [ -f "$m" ]; then
        echo "[phase4] grading $arm"
        $PY dijawithprefill/graded_judge.py \
            --manifest "$m" --out "$OUT/$arm/graded_judge.jsonl" \
            --groups B,D --model "$MODEL"
    else
        echo "[phase4] WARN: no manifest for $arm (generation incomplete?)"
    fi
done
echo "[phase4] === SUMMARY (Phase 5) ==="
$PY dijawithprefill/summarize.py --runs "$OUT" --out dijawithprefill/SUMMARY.md \
    && echo "[phase4] SUMMARY.md written" || echo "[phase4] WARN summarize failed"

echo "[phase4] ALL DONE $(date)"
