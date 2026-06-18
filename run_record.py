#!/usr/bin/env python
"""Phase-1 driver: run the 4-group injection corpus and record activations.

This is the part that must run FIRST and correctly: it extracts SAE features +
hidden states from the INJECTED TEMPLATE region (scope `tpl_mask`/`tpl_ctx`),
not just the output region.

Layout expected under --prompt-root (one subdir per group):

    prompts/cdg_injection/
        A_harmful_clean/     harmful, no template   -> should refuse
        B_harmful_injected/  harmful + template     -> attack target (positive)
        C_neutral_injected/  neutral + template     -> benign-fill control
        D_neutral_clean/     neutral, no template   -> baseline

Examples
--------
# CPU smoke test, no model / no SAEs / no judge:
python run_record.py --prompt-root prompts/cdg_injection --dummy

# Real run on LLaDA with attack-aligned decoding + injection judge:
export DEEPSEEK_API_KEY=sk-...
python run_record.py --backend llada_attack \
    --prompt-root prompts/cdg_injection --sae-root ./saes \
    --judge --judge-template injection --out outputs
"""
from __future__ import annotations
import argparse
import torch

from cdg.config import get_backend_config
from cdg.backends import build_runner
from cdg.recorder import Recorder
from cdg.data import load_cdg_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",
                    choices=["llada", "llada_attack", "dream", "dream_attack"],
                    default="llada_attack")
    ap.add_argument("--prompt-root", required=True,
                    help="dir with A_/B_/C_/D_ group subfolders")
    ap.add_argument("--sae-root", default="", help="root holding the SAE repos")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--dummy", action="store_true",
                    help="tiny CPU fake model: exercises the full pipeline")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap cases per group (0 = all)")
    # judge
    ap.add_argument("--judge", action="store_true", help="run the DeepSeek judge")
    ap.add_argument("--judge-template",
                    choices=["injection", "jailbreak", "sycophancy"],
                    default="injection")
    ap.add_argument("--judge-model", default="deepseek-v4-flash")
    ap.add_argument("--judge-groups", default="B",
                    help="comma list of group letters to judge (default B only)")
    args = ap.parse_args()

    cfg = get_backend_config(args.backend)
    runner = build_runner(cfg, sae_root=args.sae_root, device=args.device,
                          dummy=args.dummy)
    tok = getattr(runner, "tokenizer", None)

    judge = None
    if args.judge:
        from cdg.judge import DeepSeekJudge
        judge = DeepSeekJudge(template=args.judge_template, model=args.judge_model)
    judge_groups = {g.strip().upper() for g in args.judge_groups.split(",") if g.strip()}

    cases = load_cdg_root(args.prompt_root)
    if args.limit:
        seen: dict[str, int] = {}
        kept = []
        for c in cases:
            g = c.group_letter
            seen[g] = seen.get(g, 0) + 1
            if seen[g] <= args.limit:
                kept.append(c)
        cases = kept

    counts: dict[str, int] = {}
    for seed in args.seeds:
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
                "case_meta": case.meta,
            }
            rec.begin(meta, total_steps=cfg.decode.steps)
            resp = runner.generate(case, rec)

            if judge is not None and case.group_letter in judge_groups:
                jmeta = {**case.meta, "behavior": case.behavior}
                rec.set_judge(judge.judge([{"role": "user", "content": case.behavior}],
                                          resp, jmeta))
            rec.save(args.out)
            counts[case.group_letter] = counts.get(case.group_letter, 0) + 1

    summary = "  ".join(f"{g}={counts.get(g, 0)}" for g in ("A", "B", "C", "D"))
    print(f"[done] {summary}  ->  {args.out}/manifest.jsonl")


if __name__ == "__main__":
    main()
