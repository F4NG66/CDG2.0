#!/usr/bin/env bash
# run_p8_p6.sh — Compute logit-lens vocab labels then re-annotate modules
# Usage: bash run_p8_p6.sh
set -e
cd "$(dirname "$0")"

PY="/home/f4ng/cdg/bin/python3"
LLADA_CACHE="/home/f4ng/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
OUT="outputs"

echo "━━━  p8: Logit-lens vocab labels for ALL active features  ━━━"
echo "  (loads LLaDA model once, then batch-computes for all 8732 features)"
$PY scripts/p8_vocab.py \
    --module-csv "$OUT/analysis/modules/feature_module.csv" \
    --model-path "$LLADA_CACHE" \
    --layer 16 --topk 12 \
    --device cuda \
    --batch-size 2048 \
    --out analysis_output/vocab_labels.json

echo ""
echo "━━━  p6: Re-annotate modules with full logit-lens coverage  ━━━"

if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    API_ARGS="--api-key $DEEPSEEK_API_KEY --api-model deepseek-chat"
else
    API_ARGS="--no-interp"
    echo "  [info] DEEPSEEK_API_KEY not set — will aggregate tokens but skip LLM"
fi

$PY scripts/p6_annotate.py "$OUT" \
    --module-csv "$OUT/analysis/modules/feature_module.csv" \
    --vocab-labels-json analysis_output/vocab_labels.json \
    --records-dir "$OUT" \
    --scope out_unmask --layer 16 \
    --trace-scope tpl_mask \
    --tokenizer-path "$LLADA_CACHE" \
    --top-per-module 8 \
    --min-support 1 \
    --trace-top 3 \
    --save-dir "$OUT/analysis/annotation" \
    $API_ARGS

echo ""
echo "Done. Check: $OUT/analysis/annotation/module_labels.csv"
