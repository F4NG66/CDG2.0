#!/usr/bin/env python3
"""PHASE 3 step 2 - judge the steered generations with the EXACT DIJA graded judge.

Per PI lock #1: the harm/utility scores must come from the SAME graded judge as the
attack folder, verbatim - same prompt, rubric, thresholds - so they are directly
comparable to the baseline attack numbers. We therefore IMPORT the judge functions
from dijawithprefill/graded_judge.py rather than reimplement anything:

    _llm_judge            the DeepSeek call + _VALENCE_SYS/_VALENCE_USR prompt
    _heuristic_metrics    trigram_rep / length / zwsp / heuristic_collapse
    harm_delivered_inclusive (PRIMARY ASR) / harm_delivered_strict

Primary metric = graded specificity_score + valence_score (the paper's line);
binary ASR = harm_delivered rate (inclusive). D (benign) gets collapse + coherence
heuristics, no harm valence - same as the original judge_manifest's D handling.

Login-node job (needs the DeepSeek API). Resumable; parallel workers.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, "/home/ore99/serverFiles")
sys.path.insert(0, "/home/ore99/serverFiles/dijawithprefill")

from cdg.data import load_cdg_root
from graded_judge import (_heuristic_metrics, _llm_judge, _resolve_key,
                          harm_delivered_inclusive, harm_delivered_strict)

HARM_DIR = "/home/ore99/experement/harm_dir"
DATA = os.path.join(HARM_DIR, "data")
PROMPTS = "/home/ore99/serverFiles/prompts/cdg_injection"


def behaviors():
    return {c.case_id: c.behavior for c in load_cdg_root(PROMPTS)}


def judge_row(row, beh, key, model):
    behavior = beh.get(row["case_id"], "")
    resp = row.get("response_text") or ""
    v = _llm_judge(behavior, resp, key, model)
    out = dict(row)
    out["graded"] = v
    out["heuristics"] = _heuristic_metrics(resp)
    if row["group"] == "harmful":
        out["harm_delivered"] = harm_delivered_inclusive(v)
        out["harm_delivered_strict"] = harm_delivered_strict(v)
    else:
        out["harm_delivered"] = None
        out["harm_delivered_strict"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(DATA, "steer_pilot.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "steer_pilot_judged.jsonl"))
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    key = _resolve_key(None)
    if not key:
        raise SystemExit("no DeepSeek key resolved")
    beh = behaviors()
    rows = [json.loads(l) for l in open(args.inp)]

    # resume: key each row by (case_id, group, mode, arm, alpha, seed).
    # mode is required — S and SR rows otherwise share (case,group,arm,alpha,seed)
    # and would collide. Absent (Phase 3 rows) -> None, harmless.
    def rk(r):
        return (r["case_id"], r["group"], r.get("mode"), r["arm"], r["alpha"], r["seed"])
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done[rk(r)] = r
    todo = [r for r in rows if rk(r) not in done]
    print(f"rows={len(rows)}  judged_cached={len(done)}  todo={len(todo)}  model={args.model}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as out, \
            cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_row, r, beh, key, args.model): r for r in todo}
        n = 0
        for fut in cf.as_completed(futs):
            j = fut.result()
            out.write(json.dumps(j, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            if n % 25 == 0:
                print(f"  judged {n}/{len(todo)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
