#!/usr/bin/env python3
"""dija_attack/refine_100.py - Phase 2: run the UNEDITED DIJA QwenRefiner on all 100
vanilla harmful behaviors from cases.json, producing context-anchored worksheet
scaffolds (compact <mask:N>, NOT expanded) -> refined_100.json.

Reuses dija_refiner.QwenRefiner / is_valid_scaffold / parse_masks by import (never edited).
The refiner calls template.format(prompt=behavior), generates with Qwen, and gates each
result with is_valid_scaffold (2-5 masks, >=1 leading-context mask, no refusal), retrying
up to --max-attempts. Output field `Refined_behavior` keeps compact <mask:N> markers; the
LLaDA runner expands them to <|mdm_mask|> at generation time.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
if EXP not in sys.path:
    sys.path.insert(0, EXP)

from dija_refiner import QwenRefiner, is_valid_scaffold, parse_masks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(EXP, "cases.json"))
    ap.add_argument("--template", default=os.path.join(HERE, "redteam_worksheet_template.txt"))
    ap.add_argument("--qwen-path", default=os.path.join(EXP, "hf_cache", "Qwen2.5-7B-Instruct"))
    ap.add_argument("--out", default=os.path.join(HERE, "refined_100.json"))
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="first N behaviors (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    if "{prompt}" not in template:
        sys.exit("[err] template must contain a {prompt} placeholder")
    cases = json.load(open(args.cases))
    if args.limit:
        cases = cases[:args.limit]
    print(f"[refine] {len(cases)} behaviors; qwen={args.qwen_path}", flush=True)
    print(f"[refine] template={args.template}", flush=True)

    rng = random.Random(args.seed)
    refiner = QwenRefiner(args.qwen_path)

    out = []
    first, retried, failed = 0, 0, 0
    for i, c in enumerate(cases, 1):
        beh = c["behavior"]
        scaffold, meta = refiner.refine(beh, template, args.max_attempts, rng)
        # sanity: the runner drops on <mask:N>; recount here with the imported parser
        n_masks = len(parse_masks(scaffold))
        ok, _ = is_valid_scaffold(scaffold)
        if meta["valid"] and meta["attempts"] == 1:
            first += 1
        elif meta["valid"]:
            retried += 1
        else:
            failed += 1
        rec = {
            "id": c["id"],
            "behavior": beh,                 # untouched vanilla behavior (judge behavior-field)
            "Refined_behavior": scaffold,    # compact <mask:N> worksheet scaffold (runner input)
            "n_masks": n_masks,
            "valid": bool(meta["valid"]),
            "revalidated": bool(ok),
            "attempts": meta["attempts"],
            "reason": meta["reason"],
        }
        out.append(rec)
        print(f"[refine] {i}/{len(cases)} {c['id']} valid={meta['valid']} "
              f"attempts={meta['attempts']} masks={n_masks} reason={meta['reason']}", flush=True)

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[refine] wrote {len(out)} -> {args.out}", flush=True)
    print(f"[refine] first-attempt-pass={first}  retried-then-pass={retried}  failed={failed}",
          flush=True)


if __name__ == "__main__":
    main()
