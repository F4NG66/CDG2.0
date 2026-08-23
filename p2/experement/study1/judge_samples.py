#!/usr/bin/env python3
"""study1/judge_samples.py — judge the Part B straddler samples, preserving `sid`.

Same two judges and the same deepseek backend as `study1/judge_pilot.py`
(imported unedited from score_harmbench_deepseek + graded_judge); the ONLY
difference is that this keeps the per-SAMPLE identity fields (sid / seed /
temperature / lengths), which judge_pilot.py drops because it judges one row
per A-id. Part B has K samples per A-id, so `id` alone is not a key.

Separate NON-GPU process — the generation job never holds a key.
Resumable: already-judged sids are skipped.

  python study1/judge_samples.py --input /scratch/ore99/study1_straddle/samples.jsonl \
      --output /scratch/ore99/study1_straddle/judged.jsonl --workers 4
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

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
    ap.add_argument("--input", default="/scratch/ore99/study1_straddle/samples.jsonl")
    ap.add_argument("--output", default="/scratch/ore99/study1_straddle/judged.jsonl")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    key = _resolve_key(None)
    if not key:
        sys.exit("no deepseek key: set DEEPSEEK_API_KEY or write dijawithprefill/.deepseek_key")

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    done = set()
    if os.path.exists(args.output):
        for l in open(args.output):
            l = l.strip()
            if l:
                try:
                    done.add(json.loads(l)["sid"])
                except Exception:
                    pass
    todo = [r for r in rows if r["sid"] not in done]
    print(f"[judge] {len(rows)} samples, {len(done)} already judged, {len(todo)} to go "
          f"| model={args.model} workers={args.workers}\n", flush=True)

    lock = threading.Lock()
    fout = open(args.output, "a", encoding="utf-8")

    def work(r):
        beh, resp = r["behavior"], r["response"]
        hb_label, hb_score, hb_err = harmbench_judge(
            beh, resp, api_key=key, base_url=args.base_url, model=args.model)
        v = _llm_judge(beh, resp, key, args.model)
        rec = {
            "sid": r["sid"], "id": r["id"], "arm": r.get("arm", "dija"),
            "temperature": r.get("temperature"), "seed": r.get("seed"),
            "n_chars": r.get("n_chars"), "n_tokens": r.get("n_tokens"),
            "behavior": beh, "response": resp[:4000],
            "harmbench_label": hb_label, "harmbench_score": hb_score,
            "valence_category": v.get("valence_category"), "valence": v,
            "harm_delivered_inclusive": harm_delivered_inclusive(v),
            "harm_delivered_strict": harm_delivered_strict(v),
            "harmbench_err": hb_err,
        }
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{rec['sid']:28} | HB={str(hb_label):4} | "
                  f"{str(rec['valence_category'])[:30]:30} | "
                  f"incl={str(rec['harm_delivered_inclusive']):5}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fout.close()
    print(f"\n[done] -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
