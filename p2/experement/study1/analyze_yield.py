#!/usr/bin/env python3
"""study1/analyze_yield.py — PART F2 report: unselected straddler yield, 25 A-ids x K=32.

Reports exactly what the gate asks for and nothing more:
  * total minority draws and the population flip rate (UNSELECTED, so this is a
    central estimate, unlike Part C's 0.044 lower bound)
  * per-A-id outcome split
  * A-ids with >=2 minority samples  = the USABLE groups for a within-prompt probe
  * total within-prompt pairs available
  * projection to K=64 at temp 0.2, and to temp 0.4 (different-regime caveat)
  * the pre-registered length / trigram_rep_rate / collapse control
  * the judge-duplicate check (same text judged both ways => judge noise)

No capture, no probe. Numbers only.

  python study1/analyze_yield.py
"""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import defaultdict

WORD = re.compile(r"\w+")
REP_BAR, DISTINCT_BAR = 0.50, 0.30


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def degen(text):
    w = WORD.findall(text.lower())
    if len(w) < 4:
        return {"rep3": 0.0, "distinct": 1.0, "collapse": False}
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    rep = 1.0 - len(set(tri)) / len(tri)
    dis = len(set(w)) / len(w)
    return {"rep3": rep, "distinct": dis,
            "collapse": bool(rep > REP_BAR or dis < DISTINCT_BAR)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def p_at_least(m, p, K):
    """P(at least m minority outcomes in K draws), minority prob p."""
    if p <= 0:
        return 0.0
    tot = 0.0
    for i in range(m):
        tot += math.comb(K, i) * p ** i * (1 - p) ** (K - i)
    return max(0.0, 1.0 - tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/scratch/ore99/study1_straddle/samples.jsonl")
    ap.add_argument("--judged", default="/scratch/ore99/study1_straddle/judged.jsonl")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=32)
    args = ap.parse_args()

    S = {r["sid"]: r for r in (json.loads(l) for l in open(args.samples) if l.strip())}
    J = {r["sid"]: r for r in (json.loads(l) for l in open(args.judged) if l.strip())}

    cell = defaultdict(list)
    for sid, s in S.items():
        if s["temperature"] != args.temp or sid not in J:
            continue
        cell[s["id"]].append((J[sid]["harm_delivered_inclusive"] is True, s))

    ids = sorted(cell)
    print("=" * 92)
    print(f"PART F2 — UNSELECTED STRADDLER YIELD  ({len(ids)} A-ids, temp {args.temp}, "
          f"target K={args.k})")
    print("=" * 92)
    hdr = (f"  {'id':6} {'K':>3} {'deliv':>5} {'not':>4} {'minority':>8} {'p_min':>6} "
           f"{'usable':>7} {'uniq':>4} {'chars mu':>8} {'rep3':>5} {'collapse':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for cid in ids:
        g = cell[cid]
        K = len(g)
        pos = sum(1 for d, _ in g if d)
        mino = min(pos, K - pos)
        dg = [degen(s["response"]) for _, s in g]
        r = {"id": cid, "K": K, "pos": pos, "mino": mino, "g": g, "dg": dg,
             "usable": mino >= 2}
        rows.append(r)
        print(f"  {cid:6} {K:3d} {pos:5d} {K-pos:4d} {mino:8d} {mino/K:6.3f} "
              f"{str(r['usable']):>7} {len({s['response'] for _, s in g}):4d} "
              f"{mean([s['n_chars'] for _, s in g]):8.0f} "
              f"{mean([d['rep3'] for d in dg]):5.2f} "
              f"{sum(1 for d in dg if d['collapse']):4d}/{K}")

    n_draw = sum(r["K"] for r in rows)
    n_mino = sum(r["mino"] for r in rows)
    lo, hi = wilson(n_mino, n_draw)
    straddlers = [r for r in rows if r["mino"] >= 1]
    usable = [r for r in rows if r["usable"]]
    pairs_bal = sum(r["mino"] for r in usable)
    pairs_prod = sum(r["mino"] * (r["K"] - r["mino"]) for r in usable)

    print()
    print("=" * 92)
    print("HEADLINE")
    print("=" * 92)
    print(f"  total draws                          : {n_draw}")
    print(f"  total minority draws                 : {n_mino}")
    print(f"  population flip rate (UNSELECTED)    : {n_mino/n_draw:.4f} "
          f"Wilson-95 [{lo:.4f}, {hi:.4f}]")
    print(f"  A-ids with >=1 minority (straddlers) : {len(straddlers)}/{len(rows)} "
          f"= {len(straddlers)/len(rows):.2f}")
    print(f"  A-ids with >=2 minority (USABLE)     : {len(usable)}/{len(rows)} "
          f"= {len(usable)/len(rows):.2f}   <- groups for a within-prompt probe")
    print(f"  usable A-ids                         : {[r['id'] for r in usable]}")
    print(f"  within-prompt pairs, balanced        : {pairs_bal} "
          f"(minority-side samples in usable groups; the binding constraint)")
    print(f"  within-prompt pairs, all pos x neg   : {pairs_prod}")

    # ---------------- controls ----------------
    print()
    print("=" * 92)
    print("PRE-REGISTERED CONTROLS")
    print("=" * 92)
    n_col = sum(1 for r in rows for d in r["dg"] if d["collapse"])
    print(f"  degeneration: collapsed samples = {n_col}/{n_draw} "
          f"(rep3 > {REP_BAR} or distinct < {DISTINCT_BAR})")
    print(f"  {'id':6} {'chars(maj)':>10} {'chars(min)':>10} {'d':>6} "
          f"{'rep3(maj)':>9} {'rep3(min)':>9} {'verdict':>28}")
    for r in straddlers:
        maj_is_pos = r["pos"] >= r["K"] - r["pos"]
        maj = [(s, d) for (dd, s), d in zip(r["g"], r["dg"]) if dd is maj_is_pos]
        mino = [(s, d) for (dd, s), d in zip(r["g"], r["dg"]) if dd is not maj_is_pos]
        if not maj or not mino:
            continue
        dch = mean([s["n_chars"] for s, _ in mino]) - mean([s["n_chars"] for s, _ in maj])
        drep = mean([d["rep3"] for _, d in mino]) - mean([d["rep3"] for _, d in maj])
        col_m = sum(1 for _, d in mino if d["collapse"])
        v = ("ARTIFACT: minority degenerate"
             if (col_m == len(mino) and col_m > 0) or abs(drep) > 0.25
             else "not degeneration")
        print(f"  {r['id']:6} {mean([s['n_chars'] for s,_ in maj]):10.0f} "
              f"{mean([s['n_chars'] for s,_ in mino]):10.0f} {dch:+6.0f} "
              f"{mean([d['rep3'] for _,d in maj]):9.2f} "
              f"{mean([d['rep3'] for _,d in mino]):9.2f} {v:>28}")

    amb = dupes = 0
    for cid in ids:
        by = defaultdict(set)
        cnt = defaultdict(int)
        for d, s in cell[cid]:
            by[s["response"]].add(d)
            cnt[s["response"]] += 1
        amb += sum(1 for k, v in by.items() if len(v) > 1)
        dupes += sum(1 for k, c in cnt.items() if c > 1)
    print(f"\n  judge-duplicate check: {dupes} duplicated texts, {amb} judged BOTH ways")
    print("  => " + ("JUDGE NOISE PRESENT" if amb else
                     "no judge disagreement on identical text; every flip is a real"
                     " generation difference"))

    # ---------------- projections ----------------
    print()
    print("=" * 92)
    print("PROJECTIONS (reported for the capture decision — nothing started)")
    print("=" * 92)
    ps_raw = [r["mino"] / r["K"] for r in rows]
    ps_sm = [(r["mino"] + 0.5) / (r["K"] + 1.0) for r in rows]
    print(f"  {'scenario':34} {'E[usable A-ids]':>16} {'E[minority samples]':>20}")
    for name, K, ps in (
        (f"temp {args.temp}, K=32 (observed)", 32, None),
        (f"temp {args.temp}, K=64 (raw p-hat)", 64, ps_raw),
        (f"temp {args.temp}, K=64 (smoothed p)", 64, ps_sm),
    ):
        if ps is None:
            print(f"  {name:34} {len(usable):16d} {pairs_bal:20d}")
            continue
        eu = sum(p_at_least(2, p, K) for p in ps)
        em = sum(p * K for p in ps)
        print(f"  {name:34} {eu:16.1f} {em:20.1f}")

    # temp 0.4 extrapolation from the 5 A-ids measured at both temperatures
    both = [cid for cid in ids
            if any(S[s]["temperature"] == 0.4 for s in S if S[s]["id"] == cid)]
    if both:
        m02 = m04 = n02 = n04 = 0
        for cid in both:
            g2 = [(J[s]["harm_delivered_inclusive"] is True)
                  for s in S if S[s]["id"] == cid and S[s]["temperature"] == args.temp and s in J]
            g4 = [(J[s]["harm_delivered_inclusive"] is True)
                  for s in S if S[s]["id"] == cid and S[s]["temperature"] == 0.4 and s in J]
            m02 += min(sum(g2), len(g2) - sum(g2)); n02 += len(g2)
            m04 += min(sum(g4), len(g4) - sum(g4)); n04 += len(g4)
        r02, r04 = m02 / max(1, n02), m04 / max(1, n04)
        ratio = r04 / r02 if r02 > 0 else float("nan")
        print(f"\n  temp 0.4 extrapolation (from the {len(both)} A-ids measured at both temps):")
        print(f"    those A-ids: flip rate {r02:.4f} @ 0.2 vs {r04:.4f} @ 0.4  -> ratio {ratio:.2f}x")
        p04 = [min(0.5, p * ratio) for p in ps_raw]
        for K in (16, 32):
            eu = sum(p_at_least(2, p, K) for p in p04)
            print(f"    projected temp 0.4, K={K:2d}: E[usable A-ids] = {eu:.1f}/{len(rows)}"
                  f"   (DIFFERENT DECODING REGIME from the paper's config)")
        print("    caveat: extrapolated from 5 A-ids, applied as a constant multiplier;")
        print("    treat as an order-of-magnitude trade, not an estimate.")


if __name__ == "__main__":
    main()
