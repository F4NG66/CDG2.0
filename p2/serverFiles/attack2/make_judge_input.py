#!/usr/bin/env python3
"""make_judge_input.py -- turn a run_record.py manifest into HarmBench-judge input.

The manifest rows carry `case_id` + `response_text` but NOT the behavior string, so
we join back to the source behaviors by index (case_id "Bxyz" -> source_A[xyz]).

Emits one JSONL row per generation with the fields score_harmbench_deepseek.py wants:
    behavior  = the ORIGINAL harmful request (from source_A)   -> {behavior}
    response  = manifest response_text (filled scaffold + output) -> {generation}
    traj_id   = case_id (unique key, for judge resume/dedup)
    arm       = "harmful" (so the judge's harmful-arm tally is populated)

Usage:
    python attack2/make_judge_input.py \
        --manifest attack2/outputs/manifest.jsonl \
        --source   attack2/source_A/cases.json \
        --out      attack2/outputs/judge_input.jsonl
"""
from __future__ import annotations
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.source) as f:
        src = json.load(f)
    by_index = {i: c for i, c in enumerate(src)}

    rows, n_empty, seen = [], 0, set()
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            cid = m["case_id"]                       # e.g. "B007"
            if cid in seen:                          # guard against dup manifest rows
                continue
            seen.add(cid)
            idx = int(cid[1:])
            beh = by_index[idx]["behavior"]
            src_id = by_index[idx]["id"]
            resp = m.get("response_text") or ""
            if not resp.strip():
                n_empty += 1
            rows.append({
                "traj_id": cid,
                "case_id": cid,
                "source_id": src_id,
                "arm": "harmful",
                "behavior": beh,
                "response": resp,
                "response_len": len(resp),
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(rows)} judge rows -> {args.out}")
    print(f"[ok] empty responses: {n_empty}/{len(rows)}")
    lens = sorted(r["response_len"] for r in rows)
    if lens:
        print(f"[ok] response_len  min={lens[0]}  median={lens[len(lens)//2]}  max={lens[-1]}")


if __name__ == "__main__":
    main()
