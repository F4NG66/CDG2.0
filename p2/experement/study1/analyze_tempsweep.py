#!/usr/bin/env python3
"""study1/analyze_tempsweep.py — PART C report: tightened bound + temperature sweep.

C1  5 one-sided A-ids x K=32 @ temp 0.2 -> flips, updated pooled rule-of-three bound.
C2  same 5 A-ids, K=16 @ temp {0.4, 0.7} -> straddle rate vs temperature.

MANDATORY CONTROL (applies at EVERY temperature): a straddle produced by
degeneration is not decision variance. For every sample we report
  n_chars, n_tokens, trigram_rep_rate, distinct_word_ratio, collapse
and, inside each straddling A-id, we compare those between the delivered and
not-delivered samples. If the not-delivered side is systematically degenerate,
the "flip" is an artifact and is reported as such.

  python study1/analyze_tempsweep.py
"""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import defaultdict

WORD = re.compile(r"\w+")

# a sample is called degenerate if it trips either bar
REP_BAR = 0.50          # >50% of its word trigrams are repeats
DISTINCT_BAR = 0.30     # <30% of its words are distinct


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def degen(text):
    w = WORD.findall(text.lower())
    if len(w) < 4:
        return {"trigram_rep_rate": 0.0, "distinct_word_ratio": 1.0, "collapse": False,
                "n_words": len(w)}
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    rep = 1.0 - len(set(tri)) / len(tri)
    dis = len(set(w)) / len(w)
    return {"trigram_rep_rate": rep, "distinct_word_ratio": dis,
            "collapse": bool(rep > REP_BAR or dis < DISTINCT_BAR), "n_words": len(w)}


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
    ap.add_argument("--ids", default="A000,A001,A002,A003,A004")
    args = ap.parse_args()

    ids = [s.strip() for s in args.ids.split(",")]
    S = {r["sid"]: r for r in (json.loads(l) for l in open(args.samples) if l.strip())}
    J = {r["sid"]: r for r in (json.loads(l) for l in open(args.judged) if l.strip())}

    # (id, temp) -> list of (delivered, sample, degen)
    cell = defaultdict(list)
    for sid, s in S.items():
        if s["id"] not in ids or s["temperature"] == 0.0:
            continue
        j = J.get(sid)
        if j is None:
            continue
        cell[(s["id"], s["temperature"])].append(
            (j["harm_delivered_inclusive"] is True, s, degen(s["response"])))

    temps = sorted({t for _, t in cell})

    print("=" * 88)
    print("C1/C2. STRADDLE RATE vs TEMPERATURE  (valence-inclusive; 5 one-sided A-ids)")
    print("=" * 88)
    per_temp = {}
    for t in temps:
        print(f"\n--- temperature {t} ---")
        hdr = (f"  {'id':6} {'K':>3} {'deliv':>5} {'not':>4} {'straddle':>8} {'uniq':>4} "
               f"{'chars mu':>8} {'tok mu':>6} {'rep3 mu':>7} {'distinct':>8} {'collapse':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        n_st = 0
        rows = []
        for cid in ids:
            g = cell.get((cid, t), [])
            if not g:
                continue
            K = len(g)
            pos = sum(1 for d, _, _ in g if d)
            st = 0 < pos < K
            n_st += st
            u = len({s["response"] for _, s, _ in g})
            rows.append((cid, g, pos, K, st))
            print(f"  {cid:6} {K:3d} {pos:5d} {K-pos:4d} {str(st):>8} {u:4d} "
                  f"{mean([s['n_chars'] for _,s,_ in g]):8.0f} "
                  f"{mean([s['n_tokens'] for _,s,_ in g]):6.0f} "
                  f"{mean([x['trigram_rep_rate'] for _,_,x in g]):7.2f} "
                  f"{mean([x['distinct_word_ratio'] for _,_,x in g]):8.2f} "
                  f"{sum(1 for _,_,x in g if x['collapse']):4d}/{K}")
        n_ids = len(rows)
        lo, hi = wilson(n_st, n_ids)
        allpos = sum(p for _, _, p, _, _ in rows)
        allK = sum(k for _, _, _, k, _ in rows)
        print(f"\n  straddle rate = {n_st}/{n_ids} = {n_st/max(1,n_ids):.2f} "
              f"Wilson-95 [{lo:.2f}, {hi:.2f}] | pooled delivery {allpos}/{allK} "
              f"= {allpos/max(1,allK):.2f} | collapse "
              f"{sum(1 for _,g,_,_,_ in rows for _,_,x in g if x['collapse'])}/{allK}")
        per_temp[t] = (n_st, n_ids, rows, allpos, allK)

    # ---------------- degeneration control ----------------
    print()
    print("=" * 88)
    print("MANDATORY CONTROL — is any observed flip a DEGENERATION artifact?")
    print("=" * 88)
    print(f"  degenerate := trigram_rep_rate > {REP_BAR} OR distinct_word_ratio < {DISTINCT_BAR}\n")
    any_st = False
    for t in temps:
        for cid, g, pos, K, st in per_temp[t][2]:
            if not st:
                continue
            any_st = True
            a = [(s, x) for d, s, x in g if d]          # delivered
            b = [(s, x) for d, s, x in g if not d]      # not delivered
            print(f"  T={t} {cid}:  delivered {len(a)} / not {len(b)}")
            for nm, arr in (("delivered    ", a), ("not-delivered", b)):
                print(f"    {nm}: chars {mean([s['n_chars'] for s,_ in arr]):6.0f} "
                      f"tok {mean([s['n_tokens'] for s,_ in arr]):5.0f} "
                      f"rep3 {mean([x['trigram_rep_rate'] for _,x in arr]):.2f} "
                      f"distinct {mean([x['distinct_word_ratio'] for _,x in arr]):.2f} "
                      f"collapse {sum(1 for _,x in arr if x['collapse'])}/{len(arr)}")
            dch = mean([s["n_chars"] for s, _ in a]) - mean([s["n_chars"] for s, _ in b])
            drep = mean([x["trigram_rep_rate"] for _, x in a]) - \
                mean([x["trigram_rep_rate"] for _, x in b])
            col_b = sum(1 for _, x in b if x["collapse"])
            col_a = sum(1 for _, x in a if x["collapse"])
            verdict = ("ARTIFACT: the minority side is degenerate"
                       if (col_b == len(b) and col_a == 0) or
                          (col_a == len(a) and col_b == 0) or abs(drep) > 0.25
                       else "NOT explained by degeneration")
            print(f"    delta chars {dch:+.0f}, delta rep3 {drep:+.2f} -> {verdict}\n")
    if not any_st:
        print("  (no straddlers at any temperature — control has nothing to adjudicate)\n")

    # ---------------- updated pooled bound ----------------
    print("=" * 88)
    print("UPDATED POOLED RULE-OF-THREE BOUND (one-sided A-ids, temp 0.2)")
    print("=" * 88)
    g02 = [(cid, g) for cid, g, _, _, _ in per_temp.get(0.2, (0, 0, [], 0, 0))[2]]
    n_draw = minority = 0
    for cid, g in g02:
        K = len(g)
        pos = sum(1 for d, _, _ in g if d)
        side = pos if pos <= K - pos else K - pos
        n_draw += K
        minority += side
    print(f"  draws at temp 0.2 over these A-ids: n = {n_draw}, minority outcomes = {minority}")
    if minority == 0:
        print(f"  rule-of-three 95% upper bound on per-draw flip prob p <= 3/{n_draw} "
              f"= {3/n_draw:.4f}")
        for K in (8, 16, 32, 64):
            p = 3 / n_draw
            P = 1 - (1 - p) ** K - p ** K
            print(f"    at that UPPER bound: P(straddle at K={K:2d}) = {P:.2f} -> "
                  f"{math.ceil(10/P) if P>0 else float('inf'):.0f} A-ids for 10 groups")
        print("  point estimate p = 0 -> P(straddle) = 0 at every K.")
    else:
        p = minority / n_draw
        lo, hi = wilson(minority, n_draw)
        print(f"  observed flip rate p-hat = {minority}/{n_draw} = {p:.4f} "
              f"Wilson-95 [{lo:.4f}, {hi:.4f}] -- p is NOT 0, the bound is now an estimate")


if __name__ == "__main__":
    main()
