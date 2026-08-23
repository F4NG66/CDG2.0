#!/usr/bin/env python3
"""region_steer/analyze40.py — powered analysis.
PRIMARY: paired per-case specificity delta (arm - E), bootstrap 95% CI + Wilcoxon signed-rank.
SELECTIVITY: same paired delta on B (harmful spec) vs C (benign spec). SECONDARY: binary
neutralization + Wilson. Plus collapse, strata (strict/disclaimer), dedup stats."""
import os, sys, json, math, random
from collections import defaultdict
import numpy as np

EXP = "/home/ore99/experement"
ALPHAS = [21, 42, 84, 126]


def phi(z): return 0.5*(1+math.erf(z/math.sqrt(2)))


def wilcoxon(d):
    """Two-sided Wilcoxon signed-rank, normal approx w/ continuity + tie correction."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n < 6: return (None, n)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0]*n
    i = 0
    while i < n:
        j = i
        while j+1 < n and abs(d[order[j+1]]) == abs(d[order[i]]): j += 1
        avg = (i+1 + j+1)/2.0
        for k in range(i, j+1): ranks[order[k]] = avg
        i = j+1
    Wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    mean = n*(n+1)/4.0
    from collections import Counter
    ties = Counter(abs(x) for x in d)
    tie_corr = sum(t**3 - t for t in ties.values())/48.0
    var = n*(n+1)*(2*n+1)/24.0 - tie_corr
    if var <= 0: return (None, n)
    z = (Wp - mean - math.copysign(0.5, Wp-mean))/math.sqrt(var)
    return (2*(1-phi(abs(z))), n)


def boot_ci(d, B=10000, seed=0):
    if not d: return (None, None, None)
    rng = np.random.default_rng(seed)
    arr = np.array(d, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(B, len(arr)))].mean(1)
    return (round(float(arr.mean()), 3), round(float(np.percentile(means, 2.5)), 3),
            round(float(np.percentile(means, 97.5)), 3))


def wilson(k, n, z=1.96):
    if n == 0: return (None, None, None)
    p = k/n; dd = 1+z*z/n; c = (p+z*z/(2*n))/dd
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/dd
    return (round(p, 3), round(max(0, c-h), 3), round(min(1, c+h), 3))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=os.path.join(EXP, "region_steer/results/judged40.jsonl"))
    args = ap.parse_args()
    J = args.judged
    rows = [json.loads(l) for l in open(J) if l.strip()]
    # index: (group, id, arm, alpha) -> row  (E has alpha 0)
    idx = {(r["group"], r["id"], r["arm"], r["alpha"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    stratum = {r["id"]: r["stratum"] for r in rows}
    strict = [c for c in ids if stratum[c] == "strict"]
    disc = [c for c in ids if stratum[c] == "disclaimer"]

    dd = json.load(open(os.path.splitext(J)[0]+"_dedup.json")) if os.path.exists(os.path.splitext(J)[0]+"_dedup.json") else {}
    print("="*104)
    print(f"POWERED CONFIRMATION  n={len(ids)}  (strict={len(strict)}, disclaimer={len(disc)})   dedup={dd}")
    print("="*104)

    def spec(group, cid, arm, alpha):
        r = idx.get((group, cid, arm, alpha))
        if not r: return None
        if group == "B":
            v = r["graded"]; return v.get("specificity_score") if v else None
        return r["benign"].get("specificity_score")

    def paired_delta(group, arm, alpha, case_pool):
        d = []
        for cid in case_pool:
            e = spec(group, cid, "E", 0.0); a = spec(group, cid, arm, alpha)
            if e is not None and a is not None: d.append(a - e)
        return d

    # ---------- PRIMARY: paired specificity delta (B), all arms ----------
    print("\n### PRIMARY — B harmful-specificity paired delta (arm - E) | bootstrap 95% CI | Wilcoxon p")
    print(f"{'arm':<4}{'alpha':>6}{'n':>4}{'mean_d':>9}{'  boot95CI':>18}{'  Wilcoxon_p':>13}{'  sig':>5}")
    for arm in ["T", "O", "R", "Rw"]:
        for a in ALPHAS:
            d = paired_delta("B", arm, a, ids)
            m, lo, hi = boot_ci(d, seed=42)
            p, n = wilcoxon(d)
            sig = "***" if (p is not None and p < 0.001) else "**" if (p is not None and p < 0.01) else "*" if (p is not None and p < 0.05) else "ns"
            ps = f"{p:.4g}" if p is not None else "NA"
            print(f"{arm:<4}{a:>6}{n:>4}{m:>9.3f}{f'[{lo},{hi}]':>18}{ps:>13}{sig:>5}")

    # ---------- SELECTIVITY: B vs C paired delta for T and O ----------
    print("\n### SELECTIVITY — paired specificity delta on B (harmful) vs C (benign): T and O")
    print(f"{'arm':<4}{'alpha':>6} | {'B mean_d':>9}{'  B_CI':>16}{'  B_p':>10} | {'C mean_d':>9}{'  C_CI':>16}{'  C_p':>10}  verdict")
    for arm in ["T", "O"]:
        for a in ALPHAS:
            db = paired_delta("B", arm, a, ids); mb, lob, hib = boot_ci(db, seed=1); pb, _ = wilcoxon(db)
            dc = paired_delta("C", arm, a, ids); mc, loc, hic = boot_ci(dc, seed=1); pc, _ = wilcoxon(dc)
            selective = (pb is not None and pb < 0.05 and mb < 0 and (pc is None or pc >= 0.05 or abs(mc) < abs(mb)/2))
            verdict = "SELECTIVE" if selective else ("both-drop" if (mc is not None and mc < -0.05) else "-")
            pbs = f"{pb:.3g}" if pb is not None else "NA"; pcs = f"{pc:.3g}" if pc is not None else "NA"
            print(f"{arm:<4}{a:>6} | {mb:>9.3f}{f'[{lob},{hib}]':>16}{pbs:>10} | {mc:>9.3f}{f'[{loc},{hic}]':>16}{pcs:>10}  {verdict}")

    # ---------- STRATA: T primary delta within strict vs disclaimer ----------
    print("\n### STRATA — T harmful-specificity paired delta by baseline harm type")
    print(f"{'stratum':<12}{'alpha':>6}{'n':>4}{'mean_d':>9}{'  boot95CI':>18}{'  Wilcoxon_p':>13}")
    for name, pool in [("strict", strict), ("disclaimer", disc)]:
        for a in ALPHAS:
            d = paired_delta("B", "T", a, pool); m, lo, hi = boot_ci(d, seed=7); p, n = wilcoxon(d)
            ps = f"{p:.4g}" if p is not None else "NA"
            print(f"{name:<12}{a:>6}{n:>4}{m:>9.3f}{f'[{lo},{hi}]':>18}{ps:>13}")

    # ---------- SECONDARY: binary neutralization + Wilson; collapse ----------
    print("\n### SECONDARY — B harm_delivered_inclusive (Wilson 95% CI) + collapse")
    def brate(arm, a):
        rs = [idx[("B", c, arm, a)] for c in ids if ("B", c, arm, a) in idx]
        rs = [r for r in rs if r["harm_inc"] is not None]
        k = sum(1 for r in rs if r["harm_inc"]); n = len(rs)
        cj = np.mean([r["graded"]["collapse_score"] for r in rs]) if rs else float("nan")
        ch = np.mean([1.0 if r["metrics"]["collapse_heur"] else 0.0 for r in rs]) if rs else float("nan")
        return k, n, cj, ch
    kE, nE, _, _ = brate("E", 0.0); pE = wilson(kE, nE)
    print(f"  E baseline: harm_inc {kE}/{nE} = {pE[0]} CI[{pE[1]},{pE[2]}]")
    print(f"{'arm':<4}{'alpha':>6}{'harm_inc':>10}{'  95% CI':>16}{'coll_J':>8}{'coll_H':>8}")
    for arm in ["T", "O", "R", "Rw"]:
        for a in ALPHAS:
            k, n, cj, ch = brate(arm, a); p, lo, hi = wilson(k, n)
            print(f"{arm:<4}{a:>6}{f'{k}/{n}':>10}{f'[{lo},{hi}]':>16}{cj:>8.2f}{ch:>8.2f}")

    # ---------- C helpfulness (utility guard) ----------
    print("\n### GROUP-C benign utility (guard)")
    print(f"{'arm':<4}{'alpha':>6}{'helpful':>9}{'on_topic':>9}{'benign_spec':>12}{'coll_J':>8}")
    for arm, alist in [("E", [0.0]), ("T", ALPHAS), ("O", ALPHAS)]:
        for a in alist:
            rs = [idx[("C", c, arm, a)] for c in ids if ("C", c, arm, a) in idx]
            rs = [r for r in rs if r["benign"]["helpful_score"] is not None]
            if not rs: continue
            hp = np.mean([r["benign"]["helpful_score"] for r in rs])
            ot = np.mean([1.0 if r["benign"]["on_topic"] else 0.0 for r in rs])
            sp = np.mean([r["benign"]["specificity_score"] for r in rs])
            cj = np.mean([r["benign"]["collapse_score"] for r in rs])
            print(f"{arm:<4}{a:>6g}{hp:>9.2f}{ot:>9.2f}{sp:>12.2f}{cj:>8.2f}")


if __name__ == "__main__":
    main()
