#!/usr/bin/env python3
"""PHASE 1 step 3 - 70/30 TRAIN/TEST split by base-request id.

Group split: a case_id's harm half and safe half always land on the SAME side.
That is the paired-item leakage we hit before - if B042-harm trains and B042-safe
tests, the shared prompt alone carries the answer across.

Stratified by safe_source so both sides carry natural (LLaDA) and generated
(DeepSeek) pairs in proportion; the source mix is what the Phase 2 source control
is read against.
"""
import argparse
import collections
import json
import os
import random
import statistics

DATA = "/home/ore99/experement/harm_dir/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs_complete.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "split.json"))
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="stratified",
                    choices=["stratified", "natural_heavy"],
                    help="stratified: 70/30, source mix held constant (primary). "
                         "natural_heavy: push natural-LLaDA pairs into TEST so the "
                         "now-blocking natural-only AUC is actually powered.")
    ap.add_argument("--nat-test", type=int, default=20,
                    help="natural_heavy: natural pairs to place in TEST")
    ap.add_argument("--gen-test", type=int, default=10,
                    help="natural_heavy: generated pairs to place in TEST")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs)]
    rng = random.Random(args.seed)

    by_src = collections.defaultdict(list)
    for p in pairs:
        by_src[p["safe_source"]].append(p["case_id"])

    train, test = [], []
    if args.mode == "stratified":
        for src, ids in sorted(by_src.items()):
            ids = sorted(ids)
            rng.shuffle(ids)
            k = round(len(ids) * args.train_frac)
            train += ids[:k]
            test += ids[k:]
    else:
        # A blocking gate read off 8 held-out pairs is a coin flip; this mode
        # trades a little train size for a natural-only AUC you can act on.
        # It is also the STRICTER test: v_harm ends up fitted mostly on
        # DeepSeek-written safe text, so if it still separates LLaDA-harm from
        # LLaDA-safe on held-out natural pairs, it is not an authorship axis.
        nat = sorted(by_src["llada_prefill_arm"])
        gen = sorted(by_src["deepseek_generated"])
        rng.shuffle(nat)
        rng.shuffle(gen)
        n_nat = min(args.nat_test, max(0, len(nat) - 4))   # leave a few to fit on
        n_gen = min(args.gen_test, len(gen))
        test += nat[:n_nat] + gen[:n_gen]
        train += nat[n_nat:] + gen[n_gen:]
    train, test = sorted(train), sorted(test)

    assert not (set(train) & set(test)), "group split leaked"
    assert len(train) + len(test) == len(pairs)

    idx = {p["case_id"]: p for p in pairs}
    out = {
        "seed": args.seed, "train_frac": args.train_frac, "mode": args.mode,
        "unit": "base-request id (case_id); harm and safe halves share a side",
        "train": train, "test": test,
        "n_train": len(train), "n_test": len(test),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("=" * 70)
    print("70/30 GROUP SPLIT BY BASE-REQUEST ID")
    print("=" * 70)
    print(f"  train {len(train)} ids   test {len(test)} ids   (seed {args.seed})")
    for name, ids in (("train", train), ("test", test)):
        src = collections.Counter(idx[c]["safe_source"] for c in ids)
        hw = [idx[c]["harm_words"] for c in ids]
        sw = [idx[c]["safe_words"] for c in ids]
        print(f"  {name:5s}: sources {dict(src)}")
        print(f"         harm words median {statistics.median(hw):5.1f}   "
              f"safe words median {statistics.median(sw):5.1f}")
    natural_test = [c for c in test if idx[c]["safe_source"] == "llada_prefill_arm"]
    print(f"\n  natural-only TEST subset (source control): {len(natural_test)} pairs")
    if len(natural_test) < 10:
        print("  NOTE: small - read the source control mainly off cos(v_harm, v_source),")
        print("        which uses the full TRAIN set rather than this subset.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
