#!/usr/bin/env bash
# run_pipeline.sh — End-to-end pipeline: Phase-1 recording → full analysis
#
# Usage:
#   bash run_pipeline.sh                        # full run with defaults
#   bash run_pipeline.sh --skip-record          # skip recording, use existing outputs
#   bash run_pipeline.sh --seeds "0 1 2"        # multiple seeds
#   bash run_pipeline.sh --limit 1              # 1 case/group (quick smoke test)
#
# Requirements:
#   export DEEPSEEK_API_KEY=sk-...              # for judge + LLM annotation
#   GPU available (or add --dummy for CPU test)
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

# ── Configurable defaults ─────────────────────────────────────────────────────
PY="${PY:-python}"
BACKEND="${BACKEND:-llada_attack}"
PROMPT_ROOT="${PROMPT_ROOT:-prompts/cdg_injection}"
SAE_ROOT="${SAE_ROOT:-./saes}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0}"                  # space-separated, e.g. "0 1 2"
LIMIT="${LIMIT:-0}"                  # 0 = all cases
OUT="${OUT:-outputs}"                # Phase-1 records (standard)
OUT_TOK="${OUT_TOK:-outputs_tok}"    # Phase-1 records WITH token-level activations
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
LLADA_CACHE="${LLADA_CACHE:-GSAI-ML/LLaDA-8B-Instruct}"

SKIP_RECORD=0
DUMMY_FLAG=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-record)   SKIP_RECORD=1 ;;
        --dummy)         DUMMY_FLAG="--dummy"; DEVICE="cpu" ;;
        --seeds)         SEEDS="$2"; shift ;;
        --limit)         LIMIT="$2"; shift ;;
        --out)           OUT="$2"; shift ;;
        --out-tok)       OUT_TOK="$2"; shift ;;
        --backend)       BACKEND="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

SEED_ARGS=$(echo "$SEEDS" | tr ' ' '\n' | xargs -I{} echo -n "{} ")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          CDG2  End-to-End Pipeline                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  backend     : $BACKEND"
echo "║  seeds       : $SEEDS"
echo "║  limit/group : ${LIMIT:-all}"
echo "║  device      : $DEVICE"
echo "║  outputs     : $OUT  (standard)"
echo "║              : $OUT_TOK  (token-level)"
echo "║  skip record : $SKIP_RECORD"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1A — Standard recording (for pseudobulk analysis)
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SKIP_RECORD -eq 0 ]]; then
    # Single recording pass with --record-token-level:
    # Captures BOTH the pseudobulk pooled activations (for p2/p4/p5)
    # AND per-token sparse activations at frac=1.0 (for p6 annotation).
    # The two are stored in separate fields of the same .pt file:
    #   rec['sae'][scope][frac][layer]        ← pseudobulk (mean over tokens)
    #   rec['sae_tokens'][scope][layer]       ← per-token TopK sparse (frac=1.0 only)
    # Overhead: ~50-100 KB extra per generation → negligible.
    # OUT and OUT_TOK point to the same directory since we only need one run.
    OUT_TOK="$OUT"
    echo -e "\n━━━  Phase 1: Recording (pseudobulk + token-level) → $OUT  ━━━"
    $PY run_record.py \
        --backend "$BACKEND" \
        --prompt-root "$PROMPT_ROOT" \
        --sae-root "$SAE_ROOT" \
        --out "$OUT" \
        --device "$DEVICE" \
        --seeds $SEED_ARGS \
        ${LIMIT:+--limit $LIMIT} \
        ${DEEPSEEK_API_KEY:+--judge} \
        ${DEEPSEEK_API_KEY:+--judge-template injection} \
        ${DEEPSEEK_API_KEY:+--judge-model $JUDGE_MODEL} \
        --record-token-level \
        $DUMMY_FLAG
else
    OUT_TOK="$OUT"
    echo -e "\n[skip] Phase 1 recording (--skip-record)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — QC check
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n━━━  p1: QC check  ━━━"
$PY scripts/p1_check.py --out-dir "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Pseudobulk assembly
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n━━━  p2: Batch pseudobulk assembly  ━━━"
$PY scripts/p2_assemble.py "$OUT" \
    --scopes tpl_mask out_mask \
    --layers 16 \
    --fracs 0.10

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Linear probe sweep (can run in parallel with p4)
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n━━━  p3: Linear probe sweep  ━━━"
$PY scripts/s2_probe_sweep.py 2>/dev/null || echo "  [warn] probe sweep skipped (small N expected)"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Δ vectors + differential analysis + visualizations
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n━━━  p4 quick: feature ranking (tpl_mask)  ━━━"
$PY scripts/p4_delta.py "$OUT" \
    --scope tpl_mask --layer 16 --frac 0.10 \
    --quick --topk 20

echo -e "\n━━━  p4 full: statistical DE + cosine atlas (out_mask, all groups)  ━━━"
$PY scripts/p4_delta.py "$OUT" \
    --scope out_mask --layer 16 --frac 0.10 \
    --full

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Co-activation clustering + heatmap
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n━━━  p5: Co-activation modules (tpl_mask, injection-specific)  ━━━"
$PY scripts/p5_modules.py "$OUT" \
    --scope tpl_mask --layer 16 --frac 0.10 \
    --k-min 3 --k-max 15 \
    --marker-dir "$OUT/analysis/delta" \
    --save-dir "$OUT/analysis/modules"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Cluster annotation + per-module activation traces
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n━━━  p6: Cluster annotation + activation traces  ━━━"
if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    P6_INTERP="--api-key $DEEPSEEK_API_KEY --api-model deepseek-chat"
else
    P6_INTERP="--no-interp"
    echo "  [info] DEEPSEEK_API_KEY not set — running without LLM annotation"
fi
$PY scripts/p6_annotate.py "$OUT_TOK" \
    --module-csv "$OUT/analysis/modules/feature_module.csv" \
    --vocab-labels-json analysis_output/vocab_labels.json \
    --records-dir "$OUT" \
    --scope out_unmask --layer 16 \
    --trace-scope tpl_mask \
    --tokenizer-path "$LLADA_CACHE" \
    --top-per-module 6 --min-support 2 \
    --trace-top 3 \
    --save-dir "$OUT_TOK/analysis/annotation" \
    $P6_INTERP

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pipeline complete!                                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Records (all):        $OUT/"
echo "║  Assembled matrices:   $OUT/assembled/"
echo "║  Δ vectors + atlas:    $OUT/analysis/delta/"
echo "║    delta_atlas.png     (6×6 cosine orthogonality heatmap)"
echo "║    top_features.png    (top-20 features by |Δ| per pair)"
echo "║  Co-activation mods:   $OUT/analysis/modules/"
echo "║    corr_heatmap.png    (clustered correlation matrix)"
echo "║  Annotation + traces:  $OUT_TOK/analysis/annotation/"
echo "║    module_annotation.png"
echo "║    traces/module_*.png (activation traces per module)"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Optional next steps:"
echo "║  Steering defense:     $PY scripts/p7_steer.py"
echo "║  Logit-lens (needs model):"
echo "║    $PY scripts/p8_vocab.py --features 12130 5255 3338"
echo "╚══════════════════════════════════════════════════════════════╝"
