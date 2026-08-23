#!/usr/bin/env python3
"""PHASE 1' step 4 - 70/30 TRAIN/TEST group split for the DeepSeek-both pairs.

Group split by base-request id: a case_id's harm half and safe half always land on
the SAME side, so the shared prompt can never carry the answer across (the paired-
item leakage from before).

Unlike Phase 1's split there is no authorship stratum to balance - every pair is
DeepSeek-authored now - so this is a plain seeded 70/30 split over case_ids.
"""
import argparse
import json
import os
import random
import statistics

DATA = "/home/ore99/experement/harm_dir/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs_ds.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "split_ds.json"))
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs)]
    ids = sorted(p["case_id"] for p in pairs)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    k = round(len(ids) * args.train_frac)
    train, test = sorted(ids[:k]), sorted(ids[k:])

    assert not (set(train) & set(test)), "group split leaked"
    assert len(train) + len(test) == len(pairs)

    idx = {p["case_id"]: p for p in pairs}
    out = {"seed": args.seed, "train_frac": args.train_frac,
           "unit": "base-request id (case_id); harm and safe halves share a side",
           "author": "deepseek",
           "train": train, "test": test, "n_train": len(train), "n_test": len(test)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("=" * 70)
    print("70/30 GROUP SPLIT BY BASE-REQUEST ID (DeepSeek-both pairs)")
    print("=" * 70)
    print(f"  train {len(train)} ids   test {len(test)} ids   (seed {args.seed})")
    for name, sel in (("train", train), ("test", test)):
        hw = [idx[c]["harm_words"] for c in sel]
        sw = [idx[c]["safe_words"] for c in sel]
        jj = [idx[c]["jaccard"] for c in sel]
        print(f"  {name:5s}: harm words median {statistics.median(hw):5.1f}   "
              f"safe words median {statistics.median(sw):5.1f}   "
              f"jaccard median {statistics.median(jj):.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
