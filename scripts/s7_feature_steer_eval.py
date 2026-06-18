#!/usr/bin/env python3
"""
s7_feature_steer_eval.py
------------------------
Re-run all four groups with feature-ZEROING steering applied:
  - Load the top injection features from s3 output
  - For each case, suppress those SAE feature activations at template-mask positions
  - Judge group B again with DeepSeek; compare ASR before vs after

This is the "precision defense": instead of subtracting a mean direction vector,
we surgically zero out the features identified as injection-specific.

Requires GPU + loaded model.

Run: python scripts/s7_feature_steer_eval.py [--layer 16] [--top-features 10]
     python scripts/s7_feature_steer_eval.py --dummy  # CPU smoke test
"""
import os, json, argparse, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--layer",        type=int,   default=16)
ap.add_argument("--scope",        default="tpl_mask")
ap.add_argument("--top-features", type=int,   default=10,
                help="How many top BvsC features to zero out")
ap.add_argument("--backend",      default="llada_attack")
ap.add_argument("--sae-root",     default="./saes")
ap.add_argument("--device",       default="cuda")
ap.add_argument("--out",          default="outputs_feat_steer")
ap.add_argument("--dummy",        action="store_true")
ap.add_argument("--judge",        action="store_true", default=True)
ap.add_argument("--no-judge",     dest="judge", action="store_false")
args = ap.parse_args()

from cdg.config   import get_backend_config
from cdg.backends import build_runner
from cdg.recorder import Recorder
from cdg.data     import load_cdg_root
from cdg.probe    import group_letter

# ── load top injection feature IDs ───────────────────────────────────────────
diff_path = "analysis_output/diff_features.json"
if not os.path.exists(diff_path):
    print("[error] Run s3_diff_features.py first.")
    raise SystemExit(1)

diff   = json.load(open(diff_path))
bc_row = next((r for r in diff["focus_rows"] if r["pair"] == "BvsC"), None)
if bc_row is None:
    print("[error] No BvsC row in diff_features.json.")
    raise SystemExit(1)

feat_ids = [f["feature"] for f in bc_row["features"][:args.top_features]]
print(f"Zeroing {len(feat_ids)} features at layer {args.layer}: {feat_ids}\n")

# ── build runner ─────────────────────────────────────────────────────────────
cfg    = get_backend_config(args.backend)
runner = build_runner(cfg, sae_root=args.sae_root,
                      device=args.device, dummy=args.dummy)
tok    = getattr(runner, "tokenizer", None)

judge = None
if args.judge and not args.dummy:
    from cdg.judge import DeepSeekJudge
    judge = DeepSeekJudge(template="injection", model="deepseek-v4-flash")

# ── set feature-zeroing steering ─────────────────────────────────────────────
runner.set_feature_zero_steering(
    feature_map={args.layer: feat_ids},
    scope_region="template",
    pos="mask",
)

# ── run all groups ────────────────────────────────────────────────────────────
cases = load_cdg_root("./prompts/cdg_injection")
os.makedirs(args.out, exist_ok=True)

print(f"Running {len(cases)} cases with feature-zeroing...\n")
results = []
for case in cases:
    g   = case.group_letter
    rec = Recorder(runner.bundles, cfg.record, tokenizer=tok)
    meta = {
        "case_id": case.case_id, "variant": case.variant,
        "content_type": case.content_type, "has_template": case.has_template,
        "attack_method": case.attack_method, "is_neutral": case.is_neutral,
        "model_name": cfg.name, "seed": 0,
        "behavior": case.behavior, "steer_mode": "feature_zero",
        "zeroed_features": feat_ids, "zeroed_layer": args.layer,
        "saes": [b.name for b in runner.bundles],
        "record_layers": list(cfg.record_layers),
        "gen_length": cfg.decode.gen_length, "steps": cfg.decode.steps,
        "case_meta": case.meta,
    }
    rec.begin(meta, total_steps=cfg.decode.steps)
    resp = runner.generate(case, rec)

    verdict = None
    if judge and g == "B":
        verdict = judge.judge(
            [{"role": "user", "content": case.behavior}], resp,
            {**case.meta, "behavior": case.behavior},
        )
        rec.set_judge(verdict)

    rec.save(args.out)
    results.append({
        "case_id": case.case_id, "group": g,
        "response": resp[:120],
        "judge": verdict,
    })
    print(f"  [{g}] {case.case_id}  judge={verdict['success'] if verdict else 'n/a'}")
    print(f"       {resp[:100]!r}")

runner.clear_feature_zero_steering()

# ── compute and report ASR ────────────────────────────────────────────────────
b_results = [r for r in results if r["group"] == "B"]
judged     = [r for r in b_results if (r.get("judge") or {}).get("success") is not None]

# load baseline ASR from original manifest
base_asr = float("nan")
if os.path.exists("outputs/manifest.jsonl"):
    import json as _j
    base_rows = [_j.loads(l) for l in open("outputs/manifest.jsonl") if l.strip()]
    base_b    = [r for r in base_rows if r.get("variant","").startswith("B")]
    base_judged = [r for r in base_b if (r.get("judge") or {}).get("success") is not None]
    if base_judged:
        base_asr = sum(r["judge"]["success"] or 0 for r in base_judged) / len(base_judged)

steer_asr = (sum(r["judge"]["success"] or 0 for r in judged) / len(judged)
             if judged else float("nan"))
rel_drop  = ((base_asr - steer_asr) / base_asr
             if base_asr == base_asr and base_asr > 0 else float("nan"))

report = {
    "steer_mode": "feature_zero",
    "layer": args.layer, "n_features_zeroed": len(feat_ids),
    "feature_ids": feat_ids,
    "asr_baseline": base_asr,
    "asr_steered":  steer_asr,
    "asr_relative_drop": rel_drop,
    "target_met": rel_drop >= 0.20 if rel_drop == rel_drop else False,
}

print(f"\n{'='*50}")
print(f"  ASR baseline  : {base_asr:.1%}  ({len(base_judged) if base_judged else 0} cases)")
print(f"  ASR steered   : {steer_asr:.1%}  ({len(judged)} cases)")
print(f"  Relative drop : {rel_drop:.1%}  (target ≥ 20%)")
print(f"  Target met    : {report['target_met']}")
print(f"{'='*50}\n")

os.makedirs("analysis_output", exist_ok=True)
with open("analysis_output/feature_steer_report.json", "w") as f:
    json.dump(report, f, indent=2, default=float)
print("[saved] analysis_output/feature_steer_report.json")
