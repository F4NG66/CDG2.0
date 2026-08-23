#!/usr/bin/env python
"""Run one or more DIJA+prefill arms through the EXISTING LLaDA runner.

This WRAPS the existing pipeline by IMPORTING it (cdg.backends.build_runner,
cdg.recorder.Recorder, cdg.data.load_cdg_root, cdg.config.get_backend_config) and
replicating run_record.py's per-case loop so the 8B model + SAEs load ONCE and are
reused across many arms (Phase 4 runs 12 arms; reloading each time is wasteful).
No existing file is modified.

Outputs land under  <out-root>/<arm>/<model_name>/<variant>/...  with a per-arm
manifest.jsonl, mirroring the existing recorder layout.

Inline judging is OFF here (the graded judge runs as a separate post-step via
graded_judge.py, which reads the manifest).  Optionally pass --binary-judge to use
the existing cdg.judge.DeepSeekJudge on group B inline (needs DEEPSEEK_API_KEY).

Examples
--------
# smoke: mid dose-2 and the dose-0 baseline, 3 cases/group, no judge:
python dijawithprefill/run_arms.py --arms mid_2,mid_0 --limit 3 \
    --out-root dijawithprefill/runs

# full grid (Phase 4):
python dijawithprefill/run_arms.py --arms ALL --out-root dijawithprefill/runs
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from cdg.config import get_backend_config  # noqa: E402
from cdg.backends import build_runner       # noqa: E402
from cdg.recorder import Recorder           # noqa: E402
from cdg.data import load_cdg_root          # noqa: E402
import prefill_seeds as ps                  # noqa: E402

PROMPTS_ROOT = os.path.join(HERE, "prompts")


def all_arms() -> list[str]:
    return [f"{p}_{d}" for p in ps.POSITIONS for d in ps.DOSES]


def _limit_cases(cases, limit, groups):
    seen, kept = {}, []
    for c in cases:
        g = c.group_letter
        if groups and g not in groups:
            continue
        seen[g] = seen.get(g, 0) + 1
        if not limit or seen[g] <= limit:
            kept.append(c)
    return kept


def run_arm(runner, cfg, arm, out_root, limit, groups, seed, judge, judge_groups):
    prompt_root = os.path.join(PROMPTS_ROOT, arm)
    out_dir = os.path.join(out_root, arm)
    cases = _limit_cases(load_cdg_root(prompt_root), limit, groups)
    tok = getattr(runner, "tokenizer", None)
    counts = {}
    t0 = time.time()
    torch.manual_seed(seed)
    for case in cases:
        rec = Recorder(runner.bundles, cfg.record, tokenizer=tok)
        meta = {
            "case_id": case.case_id, "variant": case.variant,
            "content_type": case.content_type, "has_template": case.has_template,
            "attack_method": case.attack_method, "is_neutral": case.is_neutral,
            "model_name": cfg.name, "seed": seed,
            "behavior": case.behavior,
            "saes": [b.name for b in runner.bundles],
            "record_layers": list(cfg.record_layers),
            "gen_length": cfg.decode.gen_length, "steps": cfg.decode.steps,
            "case_meta": case.meta,          # carries group/position/dose
            "arm": arm,
        }
        rec.begin(meta, total_steps=cfg.decode.steps)
        resp = runner.generate(case, rec)
        if judge is not None and case.group_letter in judge_groups:
            jmeta = {**case.meta, "behavior": case.behavior}
            rec.set_judge(judge.judge([{"role": "user", "content": case.behavior}],
                                      resp, jmeta))
        rec.save(out_dir)
        counts[case.group_letter] = counts.get(case.group_letter, 0) + 1
    dt = time.time() - t0
    summ = "  ".join(f"{g}={counts.get(g, 0)}" for g in ("B", "D"))
    print(f"[arm {arm}] {summ}  ({dt:.1f}s)  -> {out_dir}/manifest.jsonl", flush=True)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ALL",
                    help="comma list of <position>_<dose>, or ALL")
    ap.add_argument("--backend", default="llada_attack")
    ap.add_argument("--sae-root", default=os.path.join(ROOT, "saes"))
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap cases/group (0=all)")
    ap.add_argument("--groups", default="B,D")
    ap.add_argument("--binary-judge", action="store_true",
                    help="inline cdg.judge.DeepSeekJudge on B (needs DEEPSEEK_API_KEY)")
    ap.add_argument("--judge-template", default="injection")
    ap.add_argument("--judge-model", default="deepseek-chat")
    ap.add_argument("--judge-groups", default="B")
    args = ap.parse_args()

    arms = all_arms() if args.arms.strip().upper() == "ALL" \
        else [a.strip() for a in args.arms.split(",") if a.strip()]
    groups = {g.strip().upper() for g in args.groups.split(",") if g.strip()}
    judge_groups = {g.strip().upper() for g in args.judge_groups.split(",") if g.strip()}

    cfg = get_backend_config(args.backend)
    print(f"[load] backend={args.backend} model={cfg.model_id} "
          f"saes={[s.name for s in cfg.saes]} layers={cfg.record_layers}", flush=True)
    runner = build_runner(cfg, sae_root=args.sae_root, device=args.device, dummy=False)

    judge = None
    if args.binary_judge:
        from cdg.judge import DeepSeekJudge
        judge = DeepSeekJudge(template=args.judge_template, model=args.judge_model)

    for arm in arms:
        run_arm(runner, cfg, arm, args.out_root, args.limit, groups, args.seed,
                judge, judge_groups)
    print("[done] arms:", ", ".join(arms), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
