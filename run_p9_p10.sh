#!/usr/bin/env bash
# run_p9_p10.sh — Corrected SAE Analysis (p9) + Raw Residual-Stream Steering (p10)
#
# Summary of what this fixes vs. p3/p7:
#   p9: adds B-vs-A and C-vs-D probe sweeps (true injection-mechanism contrasts),
#       TF-IDF baseline to confirm B-vs-C was content-danger not injection,
#       and optional k-SAE diagnostic if injection signal is weak.
#   p10: builds steering direction from B-vs-A / C-vs-D (not B-vs-C),
#        applies it DIRECTLY in the residual stream (no SAE encode→zero→decode),
#        sweeps dose-response and selects best (layer, α) configuration.
#
# Limit parameter:
#   --limit 0  : use all cases/records (first full run, takes longer)
#   --limit 30 : use 30 cases per group (fast comparison runs, default)
#
# Usage:
#   bash run_p9_p10.sh            # default: limit=30
#   bash run_p9_p10.sh --full     # limit=0 (all cases, production run)
#   bash run_p9_p10.sh --p9-only  # only run p9 diagnostics
#   bash run_p9_p10.sh --p10-only # only run p10 steering (requires p9 results)
#   bash run_p9_p10.sh --limit 50 # custom limit

set -e
cd "$(dirname "$0")"

PY="/home/f4ng/cdg/bin/python3"

# ── argument parsing ──────────────────────────────────────────────────────────
LIMIT=30
RUN_P9=true
RUN_P10=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)    LIMIT=0; shift ;;
        --p9-only) RUN_P10=false; shift ;;
        --p10-only) RUN_P9=false; shift ;;
        --limit)   LIMIT="$2"; shift 2 ;;
        --limit=*) LIMIT="${1#*=}"; shift ;;
        *) echo "[warn] unknown arg: $1"; shift ;;
    esac
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CDG Phase 9-10: Corrected SAE Analysis + Raw Steering"
echo "  limit=${LIMIT}  (0=all, 30=fast comparison)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Phase 9: Diagnostics + Corrected Probe Sweep ──────────────────────────────
if $RUN_P9; then
    echo ""
    echo "━━━  p9  Diagnostic + Injection-Mechanism Probe Sweep  ━━━"
    echo ""
    echo "  Task A: TF-IDF baseline (B-vs-C, B-vs-A, C-vs-D)"
    echo "  Task B: CV grouping check"
    echo "  Task C: 72-point probe sweep for B-vs-A and C-vs-D"
    echo "          (limit=${LIMIT} records per group; TF-IDF/CV use all)"
    echo "  Task D: k-SAE diagnostic (auto-triggered if AUC < 0.65)"
    echo ""

    $PY scripts/p9_diagnose.py \
        --limit "$LIMIT" \
        --task all \
        --auc-thresh 0.65 \
        --k-values 160 320 \
        --layers 11 16 26 \
        --fracs 0.05 0.10 0.20 0.35 0.50 1.00 \
        --scopes harm out_mask out_unmask \
        --out-json analysis_output/p9_results.json

    echo ""
    echo "  [p9 done]  analysis_output/p9_results.json"
    echo "             analysis_output/p9_probe_inj_sweep.json"
fi

# ── Phase 10 Config 1: harm-only ─────────────────────────────────────────────
# Direction: harm scope. Steering position: harm/unmask.
# All 4 groups (A/B/C/D) are steered and evaluated.
# A and D serve as side-effect controls (should not change).
# Answers: "does suppressing the harm-scope direction affect harmful compliance?"
if $RUN_P10; then
    echo ""
    echo "━━━  p10 [harm_only] Config 1: direction=harm, steer@harm/unmask, all 4 groups  ━━━"
    echo ""
    echo "  Key metrics per α:"
    echo "    B_asr       ↓ = defense works (main target)"
    echo "    A_refusal   ↑ = model still refuses harmless-clean requests (no side effect)"
    echo "    C_FR, D_FR  ↓ = neutral cases not disrupted"
    echo ""

    $PY scripts/p10_steer2.py \
        --limit "$LIMIT" \
        --task all \
        --tag harm_only \
        --dir-scope harm \
        --dir-frac 0.05 \
        --dir-layers 11 16 26 \
        --steer-layer 16 \
        --steer-direction shared \
        --steer-scope-region harm \
        --steer-pos unmask \
        --alpha-values 0.0 2.0 4.0 8.0 16.0 24.0 32.0 \
        --max-false-reject 0.20 \
        --p9-results analysis_output/p9_probe_inj_sweep.json \
        --top-k-configs 3 \
        --alpha-best-n 3

    echo ""
    echo "  [done] analysis_output/p10_harm_only_*"
fi

# ── Phase 10 Config 2: harm-dir + template/mask position ─────────────────────
# Direction: harm scope (same clean direction as Config 1).
# Steering position: template/mask tokens (only exists in B/C groups).
# A and D have no template region → steering has no effect on them (natural control).
# Answers: "does suppressing at the injection scaffold reduce compliance?"
if $RUN_P10; then
    echo ""
    echo "━━━  p10 [harm_dir_mask] Config 2: direction=harm, steer@template/mask  ━━━"
    echo ""
    echo "  A/D groups: no template → steering has zero effect (natural control)."
    echo "  B/C groups: steering applied at injection scaffold positions."
    echo ""

    $PY scripts/p10_steer2.py \
        --limit "$LIMIT" \
        --task all \
        --tag harm_dir_mask \
        --dir-scope harm \
        --dir-frac 0.05 \
        --dir-layers 11 16 26 \
        --steer-layer 16 \
        --steer-direction shared \
        --steer-scope-region template \
        --steer-pos mask \
        --alpha-values 0.0 2.0 4.0 8.0 16.0 24.0 32.0 \
        --max-false-reject 0.20 \
        --p9-results analysis_output/p9_probe_inj_sweep.json \
        --top-k-configs 3 \
        --alpha-best-n 3

    echo ""
    echo "  [done] analysis_output/p10_harm_dir_mask_*"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All done.  Key outputs:"
echo ""
if $RUN_P9; then
    echo "  p9 diagnostics:"
    echo "    analysis_output/p9_results.json          (TF-IDF / CV / probe / k-diag)"
    echo "    analysis_output/p9_probe_inj_sweep.json  (flat rows → consumed by p10)"
    echo "    analysis_output/p9_probe_inj_heatmap_*.png"
fi
if $RUN_P10; then
    echo "  p10 Config 1 (harm_only — steer at harm/unmask positions, all 4 groups):"
    echo "    analysis_output/p10_harm_only_directions/"
    echo "    analysis_output/p10_harm_only_dose_response.json"
    echo "    analysis_output/p10_harm_only_layer_sweep.json"
    echo "    analysis_output/p10_harm_only_dose_response_plot.png"
    echo ""
    echo "  p10 Config 2 (harm_dir_mask — direction from harm, steer at template/mask):"
    echo "    analysis_output/p10_harm_dir_mask_directions/"
    echo "    analysis_output/p10_harm_dir_mask_dose_response.json"
    echo "    analysis_output/p10_harm_dir_mask_layer_sweep.json"
    echo "    analysis_output/p10_harm_dir_mask_dose_response_plot.png"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
