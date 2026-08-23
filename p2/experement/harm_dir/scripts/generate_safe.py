#!/usr/bin/env python3
"""PHASE 1 step 2 - generate the missing SAFE responses (DeepSeek, login node).

For every pair row with safe_source == NEEDS_GENERATION, write a safe answer to
the SAME underlying request that:

  * refuses / redirects, or gives only general safety information
  * carries NO actionable harmful content
  * mirrors the STRUCTURE of its paired harmful response - same section headers,
    same numbered-list style, same prefill connector words ("First, you", ...),
    same trailing marker - so the direction cannot learn "LLaDA scaffold text vs
    clean DeepSeek prose"
  * matches its paired harmful response's word count to within +-20%

Because the safe text is written to mirror the harm response's arm connectors,
a generated pair has harm_arm == safe_arm by construction; the prefill watermark
is therefore matched *within* the pair and cannot drive the direction.

Resumable: already-generated rows are read back from the cache and skipped.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import sys
import time

from textnorm import quality, strip_tail

KEYFILE = "/home/ore99/serverFiles/dijawithprefill/.deepseek_key"
DATA = "/home/ore99/experement/harm_dir/data"

SYS = """You write SAFE counterpart responses for an AI-safety dataset that trains a \
harmful-content detector. You are given a harmful model output and must write the \
safe response that the model SHOULD have produced for the same request.

Hard requirements:
1. NO actionable harmful content. No doses, quantities, sources, procedures, \
settings, or step-by-step instructions that would advance the harmful request.
2. Instead: decline the harmful part, explain briefly why it is dangerous, and \
redirect to the legitimate route (a qualified professional, an official body, an \
emergency service) or give only well-known general safety information.
3. MIRROR THE STRUCTURE of the harmful example exactly: same section headers, same \
numbering/bullet style, same connective lead-in words at the same places. Only the \
CONTENT changes.
4. Do NOT end with an apology or a refusal sentence such as "I'm sorry, but I can't \
assist with that." Stop after the last content line.
5. Hit the target word count as closely as you can - aim for the target itself, not \
the upper bound. Do not pad with filler or repetition; use the space for genuine \
safety guidance.
6. Output ONLY the response text. No preamble, no commentary, no code fences."""

USR = """REQUEST (what the user asked for):
{behavior}

HARMFUL RESPONSE (structural template only - copy its shape, never its content):
---
{harm_text}
---

TARGET LENGTH: exactly {target} words (acceptable range {lo}-{hi}).

Write the safe counterpart response now."""

FIX = """That was {got} words but the target is {target} words ({lo}-{hi}).
Rewrite it at the correct length, keeping the same structure and the same safe \
content. Add or condense genuine safety guidance - do not pad with repetition.
Output ONLY the response text."""


def load_key(path=KEYFILE):
    with open(path) as f:
        return f.read().strip()


def call(messages, key, model, temperature=0.3, timeout=90):
    import requests
    last = None
    for attempt in range(4):
        try:
            r = requests.post(
                "https://api.deepseek.com/chat/completions",
                json={"model": model, "messages": messages,
                      "temperature": temperature},
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last = str(e)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"deepseek failed after 4 attempts: {last}")


def clean(text):
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # same tail normalisation the LLaDA sides get, so the scaffold-exit marker /
    # canned refusal can never be what separates generated from natural text
    return strip_tail(text)


def gen_one(row, key, model):
    target = row["harm_words"]
    lo, hi = int(target * 0.92), int(target * 1.08)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": USR.format(
                behavior=row["behavior"], harm_text=row["harm_text"][:4000],
                target=target, lo=lo, hi=hi)}]
    out = clean(call(msgs, key, model))
    n = len(out.split())
    tries = 1
    # one corrective pass if the length band was missed
    if not (target * 0.80 <= n <= target * 1.20):
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": FIX.format(
                     got=n, target=target, lo=lo, hi=hi)}]
        out2 = clean(call(msgs, key, model))
        tries = 2
        if abs(len(out2.split()) - target) < abs(n - target):
            out, n = out2, len(out2.split())
    return {"case_id": row["case_id"], "safe_text": out, "safe_words": n,
            "target_words": target, "ratio": round(n / target, 3),
            "in_band": bool(0.80 <= n / target <= 1.20),
            "attempts": tries, "model": model, **quality(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs.jsonl"))
    ap.add_argument("--cache", default=os.path.join(DATA, "safe_generated.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "pairs_complete.jsonl"))
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    key = load_key()
    pairs = [json.loads(l) for l in open(args.pairs)]
    need = [p for p in pairs if p["safe_source"] == "NEEDS_GENERATION"]

    done = {}
    if os.path.exists(args.cache):
        for l in open(args.cache):
            r = json.loads(l)
            done[r["case_id"]] = r
    todo = [p for p in need if p["case_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"pairs={len(pairs)}  need_generation={len(need)}  cached={len(done)}  "
          f"to_generate_now={len(todo)}  model={args.model}")

    if todo:
        t0 = time.time()
        with open(args.cache, "a") as cache, \
                cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(gen_one, r, key, args.model): r for r in todo}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                row = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [{i}/{len(todo)}] {row['case_id']} FAILED: {e}")
                    continue
                done[res["case_id"]] = res
                cache.write(json.dumps(res) + "\n")
                cache.flush()
                flag = "ok " if res["in_band"] else "OOB"
                print(f"  [{i}/{len(todo)}] {res['case_id']} {flag} "
                      f"{res['safe_words']}w / target {res['target_words']}w "
                      f"(ratio {res['ratio']}, {res['attempts']} call(s))")
        print(f"  generated {len(todo)} in {time.time()-t0:.1f}s")

    # ---- merge + report ----
    merged, missing = [], []
    for p in pairs:
        if p["safe_source"] == "NEEDS_GENERATION":
            r = done.get(p["case_id"])
            if not r:
                missing.append(p["case_id"])
                continue
            p = dict(p)
            p.update(safe_text=r["safe_text"], safe_words=r["safe_words"],
                     safe_arm=p["harm_arm"],      # structure mirrored -> same arm
                     safe_source="deepseek_generated", safe_model=r["model"],
                     safe_len_ratio=r["ratio"], safe_in_band=r["in_band"])
        else:
            p = dict(p)
            p["safe_len_ratio"] = round(p["safe_words"] / p["harm_words"], 3)
            p["safe_in_band"] = bool(0.80 <= p["safe_len_ratio"] <= 1.20)
        merged.append(p)

    with open(args.out, "w") as f:
        for p in merged:
            f.write(json.dumps(p) + "\n")

    import statistics
    print()
    print("=" * 70)
    print("SAFE SIDE COMPLETE")
    print("=" * 70)
    src = collections.Counter(p["safe_source"] for p in merged)
    print(f"  rows written : {len(merged)}   sources: {dict(src)}")
    if missing:
        print(f"  STILL MISSING: {missing}")
    for s in sorted(src):
        rows = [p for p in merged if p["safe_source"] == s]
        rat = [p["safe_len_ratio"] for p in rows]
        inb = sum(1 for p in rows if p["safe_in_band"])
        print(f"  {s:20s} n={len(rows):3d}  len ratio median {statistics.median(rat):.2f}  "
              f"in +-20% band {inb}/{len(rows)}")
    hw = [p["harm_words"] for p in merged]
    sw = [p["safe_words"] for p in merged]
    print(f"  harm words median {statistics.median(hw):.0f} mean {statistics.mean(hw):.1f}")
    print(f"  safe words median {statistics.median(sw):.0f} mean {statistics.mean(sw):.1f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
