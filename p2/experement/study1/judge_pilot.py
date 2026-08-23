#!/usr/bin/env python3
"""study1/judge_pilot.py — P2 (LOGIN NODE, needs internet): judge the pilot's dija
generations with BOTH harm definitions, side by side. NEVER run from the MIG capture
job — this is a separate step and the ONLY step that calls DeepSeek.

For each generation it reports:
  (a) HarmBench-DeepSeek  -> yes/no   (standing primary; strict "instance of behavior")
  (b) valence graded judge -> valence_category, then
        harm_delivered_inclusive  (paper primary: disclaimer != safe)
        harm_delivered_strict

Both judges hit the SAME deepseek backend (default deepseek-chat) so the ONLY thing
that differs is the prompt/definition, not the model. Reuses, unedited:
  * score_harmbench_deepseek.{build_prompt,call_deepseek,parse_label}
  * dijawithprefill/graded_judge.{_llm_judge,harm_delivered_inclusive,harm_delivered_strict,_resolve_key}

  python study1/judge_pilot.py --input /scratch/ore99/study1_pilot/generations.jsonl \
      --output /scratch/ore99/study1_pilot/judged.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIJA = "/home/ore99/serverFiles/dijawithprefill"
for p in (EXP, DIJA):
    if p not in sys.path:
        sys.path.insert(0, p)

from score_harmbench_deepseek import build_prompt, call_deepseek, parse_label  # noqa: E402
from graded_judge import (  # noqa: E402
    _llm_judge, harm_delivered_inclusive, harm_delivered_strict, _resolve_key,
)


def harmbench_judge(behavior, response, *, api_key, base_url, model):
    row = {"behavior": behavior, "response": response}
    prompt = build_prompt(row, "behavior", "response", contextual=False, context_field="context")
    raw, err = call_deepseek(prompt, api_key=api_key, base_url=base_url, model=model,
                             temperature=0.0, max_tokens=8, retries=4, timeout=60.0)
    label, score = parse_label(raw)
    return label, score, (err or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="generations.jsonl from the P1 capture")
    ap.add_argument("--output", default=None)
    ap.add_argument("--arm", default="dija", help="which arm to judge (default dija)")
    ap.add_argument("--model", default="deepseek-chat",
                    help="deepseek model for BOTH judges (constant backend)")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    args = ap.parse_args()

    key = _resolve_key(None)
    if not key:
        sys.exit("no deepseek key: set DEEPSEEK_API_KEY or write dijawithprefill/.deepseek_key")

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    rows = [r for r in rows if r.get("arm") == args.arm]
    out_path = args.output or os.path.splitext(args.input)[0] + "_judged.jsonl"

    judged = []
    print(f"[judge] {len(rows)} {args.arm} rows | model={args.model}\n")
    hdr = f"{'id':6} | {'HB':4} | {'valence_category':30} | {'incl':5} | {'strict':6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        beh, resp = r["behavior"], r["response"]
        hb_label, hb_score, hb_err = harmbench_judge(
            beh, resp, api_key=key, base_url=args.base_url, model=args.model)
        v = _llm_judge(beh, resp, key, args.model)
        incl = harm_delivered_inclusive(v)
        strict = harm_delivered_strict(v)
        cat = v.get("valence_category")
        rec = {"id": r["id"], "arm": r["arm"], "behavior": beh,
               "response": resp[:4000],
               "harmbench_label": hb_label, "harmbench_score": hb_score,
               "valence_category": cat, "valence": v,
               "harm_delivered_inclusive": incl, "harm_delivered_strict": strict,
               "harmbench_err": hb_err}
        judged.append(rec)
        print(f"{str(r['id']):6} | {str(hb_label):4} | {str(cat):30} | "
              f"{str(incl):5} | {str(strict):6}")

    with open(out_path, "w", encoding="utf-8") as f:
        for j in judged:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    n = len(judged)
    hb_yes = [j for j in judged if j["harmbench_label"] == "yes"]
    inc = [j for j in judged if j["harm_delivered_inclusive"] is True]
    strc = [j for j in judged if j["harm_delivered_strict"] is True]
    inc_ids = {j["id"] for j in inc}
    hb_subset = all(j["id"] in inc_ids for j in hb_yes)
    print("\n" + "=" * 60)
    print(f"  n judged                = {n}")
    print(f"  HarmBench yes           = {len(hb_yes)}/{n}")
    print(f"  valence inclusive True  = {len(inc)}/{n}")
    print(f"  valence strict True     = {len(strc)}/{n}")
    print(f"  HB-yes ⊆ inclusive ?    = {hb_subset}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
