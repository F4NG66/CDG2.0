#!/usr/bin/env python3
"""dija_attack/valence_judge.py - score generations with the PAPER'S own graded
valence judge (dijawithprefill/graded_judge.py), imported unedited. This is the
judge behind the paper's ~0.85 harm-delivered headline (inclusive metric =
unchanged_harmful | euphemistic_softening | disclaimer_only). Kept alongside the
HarmBench yes/no judge so we can report both bars. Threaded + resumable by traj_id.

  python dija_attack/valence_judge.py --input results/generations.jsonl \
      --output results/generations_valence.jsonl --model deepseek-v4-flash
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

DIJA = "/home/ore99/serverFiles/dijawithprefill"
if DIJA not in sys.path:
    sys.path.insert(0, DIJA)
from graded_judge import _llm_judge, harm_delivered_inclusive, harm_delivered_strict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_done(path, key):
    done = set()
    if os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if l:
                try:
                    r = json.loads(l)
                    if r.get("valence", {}).get("valence_category") is not None:
                        done.add(r[key])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--key-file", default=os.path.join(HERE, ".deepseek_key"))
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--key-field", default="traj_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    api_key = open(args.key_file).read().strip()
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    done = set() if args.overwrite else load_done(args.output, args.key_field)
    if args.overwrite and os.path.exists(args.output):
        open(args.output, "w").close()
    todo = [r for r in rows if r.get(args.key_field) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"input={len(rows)} already={len(done)} todo={len(todo)} model={args.model}", flush=True)

    lock = threading.Lock()
    counts = {"inc": 0, "strict": 0, "scored": 0, "err": 0}

    def work(r):
        v = _llm_judge(r["behavior"], r["response"], api_key, args.model)
        r["valence"] = v
        r["harm_delivered_inclusive"] = harm_delivered_inclusive(v)
        r["harm_delivered_strict"] = harm_delivered_strict(v)
        return r

    fout = open(args.output, "a", encoding="utf-8")
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for fu in as_completed(futs):
            r = fu.result()
            cat = r["valence"].get("valence_category")
            with lock:
                if cat is None:
                    counts["err"] += 1
                else:
                    counts["scored"] += 1
                    counts["inc"] += 1 if r["harm_delivered_inclusive"] else 0
                    counts["strict"] += 1 if r["harm_delivered_strict"] else 0
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                n += 1
            if n % 20 == 0:
                print(f"  valence {n}/{len(todo)} (scored={counts['scored']} err={counts['err']})", flush=True)
    fout.close()
    s = counts["scored"]
    print(f"\n[valence] scored={s} err={counts['err']}  "
          f"harm_inclusive={counts['inc']}/{s}  harm_strict={counts['strict']}/{s}", flush=True)


if __name__ == "__main__":
    main()
