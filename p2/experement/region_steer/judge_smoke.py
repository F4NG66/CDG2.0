#!/usr/bin/env python3
"""Judge the smoke generations with the graded valence judge (harm_delivered_inclusive
primary). Imports graded_judge VERBATIM (no edits)."""
import os, sys, json
sys.path.insert(0, "/home/ore99/serverFiles/dijawithprefill")
from graded_judge import _llm_judge, harm_delivered_inclusive, harm_delivered_strict, _resolve_key

EXP = "/home/ore99/experement"
rows_meta = {r["id"]: r for r in json.load(open(os.path.join(EXP, "dija_attack/refined_100.json")))}
key = _resolve_key(None)
assert key, "no deepseek key"

gens = [json.loads(l) for l in open(os.path.join(EXP, "region_steer/results/smoke.jsonl"))]
out = []
for g in gens:
    beh = rows_meta[g["id"]]["behavior"]
    v = _llm_judge(beh, g["response"], key, "deepseek-chat")
    hi = harm_delivered_inclusive(v)
    hs = harm_delivered_strict(v)
    out.append({**{k: g[k] for k in ("id", "arm", "alpha")}, "harm_inc": hi, "harm_str": hs,
                "cat": v["valence_category"], "spec": v["specificity_score"],
                "collapse_judge": v["collapse_score"], "metrics": g["metrics"]})
    print(f"{g['id']} {g['arm']:2s} a={g['alpha']:>7.2f} | harm_inc={hi} strict={hs} "
          f"cat={v['valence_category']:>28s} spec={v['specificity_score']} "
          f"collapse={v['collapse_score']} uniq={g['metrics']['uniq_ratio']}")
json.dump(out, open(os.path.join(EXP, "region_steer/results/smoke_judged.json"), "w"), indent=1)
print("\n[judge_smoke] wrote smoke_judged.json")
