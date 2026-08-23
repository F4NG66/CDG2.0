#!/usr/bin/env python3
"""PHASE 1' step 2 - Jaccard content-divergence filter on the DeepSeek-both pairs.

Phase 2 died because harm and safe halves were near-identical templated text
(natural-pair word-set Jaccard 0.503; only 1/28 genuinely diverged). This filter
is the guard against repeating that: for every DeepSeek-both pair we compute the
word-set Jaccard of (harm, safe) and DROP any pair with Jaccard > 0.35.

Report how many of 78 survive. Decision rule (from the spec):
  - if far fewer than ~40 survive -> STOP: DeepSeek is not producing a real
    harm/safe CONTENT gap, and building a direction on it would be pointless.

Word set = lowercased [a-z0-9]+ tokens, same normalisation both sides. Jaccard is
computed on the FULL stored text (already tail-stripped by the generator), so the
shared DIJA scaffold prose counts against divergence - that is deliberate, since
the scaffold is exactly the surface form we do NOT want the direction to key on.

Output: data/pairs_ds.jsonl - one row per surviving pair, tagged author=deepseek,
carrying both texts and word counts, ready for GPU capture.
"""
import argparse
import json
import os
import re
import statistics

DATA = "/home/ore99/experement/harm_dir/data"
JACCARD_MAX = 0.35
SURVIVE_FLOOR = 40   # "far fewer than ~40" -> stop


def wordset(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def jaccard(a, b):
    A, B = wordset(a), wordset(b)
    if not (A or B):
        return 1.0
    return len(A & B) / len(A | B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(DATA, "ds_both.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "pairs_ds.jsonl"))
    ap.add_argument("--jaccard-max", type=float, default=JACCARD_MAX)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    kept, dropped = [], []
    for r in rows:
        j = jaccard(r["harm_text"], r["safe_text"])
        r["jaccard"] = round(j, 4)
        (dropped if j > args.jaccard_max else kept).append(r)

    kept.sort(key=lambda r: r["case_id"])
    js = sorted(r["jaccard"] for r in rows)
    lr = [r["len_ratio"] for r in rows]

    print("=" * 78)
    print("PHASE 1' JACCARD CONTENT-DIVERGENCE FILTER")
    print("=" * 78)
    print(f"  input pairs           : {len(rows)}")
    print(f"  Jaccard threshold     : DROP if > {args.jaccard_max}")
    print(f"  SURVIVORS             : {len(kept)} / {len(rows)}")
    print(f"  dropped               : {len(dropped)}")
    print()
    print(f"  Jaccard(harm,safe)  min {js[0]:.3f}  median {statistics.median(js):.3f}  "
          f"max {js[-1]:.3f}")
    print(f"  len_ratio(safe/harm) min {min(lr):.3f}  median {statistics.median(lr):.3f}  "
          f"max {max(lr):.3f}")
    print(f"  harm words median     : {statistics.median(r['harm_words'] for r in rows):.0f}")
    print(f"  safe words median     : {statistics.median(r['safe_words'] for r in rows):.0f}")
    if dropped:
        print("\n  dropped (Jaccard > {:.2f}):".format(args.jaccard_max))
        for r in sorted(dropped, key=lambda r: -r["jaccard"]):
            print(f"    {r['case_id']}  J={r['jaccard']:.3f}  "
                  f"harm {r['harm_words']}w / safe {r['safe_words']}w")

    # highest-Jaccard survivors are the closest calls - worth an eyeball
    print("\n  closest surviving pairs (highest Jaccard kept):")
    for r in sorted(kept, key=lambda r: -r["jaccard"])[:5]:
        print(f"    {r['case_id']}  J={r['jaccard']:.3f}")

    print("\n" + "=" * 78)
    if len(kept) < SURVIVE_FLOOR:
        print(f"  STOP: only {len(kept)} survivors (< {SURVIVE_FLOOR}).")
        print("  DeepSeek is NOT producing a real harm/safe content gap at scale.")
        print("  Do not capture; report to user.")
    else:
        print(f"  PROCEED: {len(kept)} survivors (>= {SURVIVE_FLOOR}). Real content gap.")
    print("=" * 78)

    out_rows = []
    for r in kept:
        out_rows.append({
            "case_id": r["case_id"], "behavior": r["behavior"],
            "user_content": r["user_content"], "author": "deepseek",
            "harm_text": r["harm_text"], "harm_words": r["harm_words"],
            "safe_text": r["safe_text"], "safe_words": r["safe_words"],
            "len_ratio": r["len_ratio"], "jaccard": r["jaccard"],
        })
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out_rows)} surviving pairs -> {args.out}")


if __name__ == "__main__":
    main()
