#!/usr/bin/env python3
"""study1/analyze_straddle.py — PART B report: determinism, straddle rate, length confound.

Answers three questions and nothing else:

  1. DETERMINISM at temp=0. Do different seeds give different generations?
     (Source reading says no: cdg_denoise.add_gumbel_noise short-circuits at
     temperature==0 and low_confidence remasking draws no RNG. This is the
     empirical check, which also catches kernel-level nondeterminism.)
  2. STRADDLE RATE at temp>0. How many A-ids produced BOTH delivered and
     not-delivered outcomes, and what is the within-A-id split?
  3. LENGTH CONFOUND. Per-A-id sample length stats, and within straddlers the
     length difference between delivered and not-delivered samples. A straddle
     driven purely by length/degeneration is a confound, not a decision signal.

Then it projects the minimum (#A-ids, K) needed for a workable within-prompt n,
instead of just scaling blindly.

  python study1/analyze_straddle.py --samples /scratch/ore99/study1_straddle/samples.jsonl \
      --judged /scratch/ore99/study1_straddle/judged.jsonl
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/scratch/ore99/study1_straddle/samples.jsonl")
    ap.add_argument("--judged", default="/scratch/ore99/study1_straddle/judged.jsonl")
    ap.add_argument("--target-groups", type=int, default=10,
                    help="straddling A-ids needed for a workable LOGO design")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.samples) if l.strip()]
    judged = {}
    try:
        for l in open(args.judged):
            if l.strip():
                r = json.loads(l)
                judged[r["sid"]] = r
    except FileNotFoundError:
        pass

    # ---------------- 1. determinism at temp=0 ----------------
    print("=" * 78)
    print("1. DETERMINISM AT temp=0 (same prompt, different seed)")
    print("=" * 78)
    det = defaultdict(list)
    for s in samples:
        if s["temperature"] == 0.0:
            det[s["id"]].append(s)
    if not det:
        print("  (no temp=0 samples in this run)")
    n_ident = n_tot = 0
    for cid in sorted(det):
        g = sorted(det[cid], key=lambda r: r["seed"])
        base = g[0]["response"]
        same = [r["response"] == base for r in g[1:]]
        n_ident += sum(same)
        n_tot += len(same)
        print(f"  {cid}: seeds {[r['seed'] for r in g]} -> "
              f"{'IDENTICAL' if all(same) else 'DIFFERENT'} "
              f"(chars {[r['n_chars'] for r in g]})")
    if n_tot:
        print(f"\n  {n_ident}/{n_tot} cross-seed pairs byte-identical.")
        print("  => seed variation at temp=0 is " +
              ("A NO-OP: it cannot generate straddlers. Use temp>0."
               if n_ident == n_tot else
               "LIVE: prefer it, it keeps the deterministic sampling regime."))

    if not judged:
        print("\n[stop] no judged.jsonl yet — run study1/judge_samples.py first.")
        return

    # ---------------- 2. straddle rate ----------------
    print()
    print("=" * 78)
    print("2. STRADDLE RATE AT temp>0 (valence-inclusive harm label)")
    print("=" * 78)
    by_id = defaultdict(list)
    for s in samples:
        if s["temperature"] == 0.0:
            continue
        j = judged.get(s["sid"])
        if j is None:
            continue
        by_id[s["id"]].append((j["harm_delivered_inclusive"] is True, s, j))

    temps = sorted({s["temperature"] for s in samples if s["temperature"] > 0})
    print(f"  temperature(s) = {temps}")
    hdr = (f"  {'id':6} {'K':>3} {'delivered':>9} {'not':>4} {'p_hat':>6} {'straddle':>9} "
           f"{'chars mu':>9} {'chars sd':>9} {'tok mu':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    straddlers, rows = [], []
    for cid in sorted(by_id):
        g = by_id[cid]
        pos = sum(1 for d, _, _ in g if d)
        K = len(g)
        st = 0 < pos < K
        ch = [s["n_chars"] for _, s, _ in g]
        tk = [s["n_tokens"] for _, s, _ in g]
        rows.append({"id": cid, "K": K, "pos": pos, "straddle": st, "g": g})
        if st:
            straddlers.append(cid)
        print(f"  {cid:6} {K:3d} {pos:9d} {K-pos:4d} {pos/K:6.2f} {str(st):>9} "
              f"{mean(ch):9.0f} {sd(ch):9.0f} {mean(tk):7.0f}")

    n_ids = len(rows)
    n_st = len(straddlers)
    lo, hi = wilson(n_st, n_ids)
    all_pos = sum(r["pos"] for r in rows)
    all_n = sum(r["K"] for r in rows)
    print(f"\n  STRADDLE RATE = {n_st}/{n_ids} A-ids = {n_st/max(1,n_ids):.2f} "
          f"Wilson-95 [{lo:.2f}, {hi:.2f}]")
    print(f"  straddling A-ids: {straddlers if straddlers else '(none)'}")
    print(f"  pooled per-sample delivery rate = {all_pos}/{all_n} = {all_pos/max(1,all_n):.2f}")
    if straddlers:
        pairs = sum(min(r["pos"], r["K"] - r["pos"]) for r in rows if r["straddle"])
        insamp = sum(r["K"] for r in rows if r["straddle"])
        print(f"  usable within-prompt contrast: {n_st} groups, "
              f"{insamp} samples, {pairs} minority-side samples "
              f"(the binding constraint for a balanced probe)")

    # ---------------- 3. length confound ----------------
    print()
    print("=" * 78)
    print("3. LENGTH CONFOUND CHECK (within straddling A-ids)")
    print("=" * 78)
    if not straddlers:
        print("  (no straddlers — not applicable)")
    else:
        hdr2 = (f"  {'id':6} {'chars(delivered)':>17} {'chars(not)':>12} {'diff':>7} "
                f"{'tok(del)':>9} {'tok(not)':>9}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        diffs = []
        for r in rows:
            if not r["straddle"]:
                continue
            a = [s["n_chars"] for d, s, _ in r["g"] if d]
            b = [s["n_chars"] for d, s, _ in r["g"] if not d]
            ta = [s["n_tokens"] for d, s, _ in r["g"] if d]
            tb = [s["n_tokens"] for d, s, _ in r["g"] if not d]
            diffs.append(mean(a) - mean(b))
            print(f"  {r['id']:6} {mean(a):17.0f} {mean(b):12.0f} "
                  f"{mean(a)-mean(b):+7.0f} {mean(ta):9.0f} {mean(tb):9.0f}")
        print(f"\n  mean within-A-id length difference (delivered - not) = "
              f"{mean(diffs):+.0f} chars  (sd {sd(diffs):.0f})")
        print("  A large consistent difference means the probe could be reading "
              "length/degeneration,\n  not the harm decision — treat as a confound "
              "to control, not a signal.")

    # ---------------- 4. projection ----------------
    print()
    print("=" * 78)
    print(f"4. PROJECTION — what it takes to reach {args.target_groups} straddling A-ids")
    print("=" * 78)
    print("  P(A-id straddles at K samples) = 1 - p^K - (1-p)^K, p = that A-id's")
    print("  per-sample delivery rate. Estimated by plugging in the observed p-hats")
    print("  (Laplace-smoothed), which is optimistic for p-hat in {0,1} and honest otherwise.\n")
    ps = [(r["pos"] + 0.5) / (r["K"] + 1.0) for r in rows]
    print(f"  {'K':>4} {'E[straddle rate]':>17} {'A-ids needed':>13} {'generations':>12}")
    for K in (4, 8, 12, 16, 24, 32):
        pr = mean([1 - p ** K - (1 - p) ** K for p in ps])
        need = math.ceil(args.target_groups / pr) if pr > 1e-9 else float("inf")
        gen = need * K if need != float("inf") else float("inf")
        print(f"  {K:4d} {pr:17.2f} {need:13} {gen:12}")
    print("\n  (generations = A-ids x K, generate-only; hidden capture would follow")
    print("   for the straddling A-ids only, at ~0.4 GiB per case-arm ragged.)")


if __name__ == "__main__":
    main()
