#!/usr/bin/env python3
"""region_steer/analyze.py — falsification summary with Wilson CIs.
Primary: B harm_delivered_inclusive per arm x alpha vs baseline E. Collapse + Group-C
benign utility reported alongside (a harm drop that is collapse or wrecks C is NOT success)."""
import os, sys, json, math
from collections import defaultdict

EXP = "/home/ore99/experement"
JUDGED = os.path.join(EXP, "region_steer/results/judged.jsonl")


def wilson(k, n, z=1.96):
    if n == 0: return (None, None, None)
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(p, 3), round(max(0, c-h), 3), round(min(1, c+h), 3))


def main():
    rows = [json.loads(l) for l in open(JUDGED) if l.strip()]
    B = [r for r in rows if r["group"] == "B"]
    C = [r for r in rows if r["group"] == "C"]

    # ---- baseline E (B) ----
    E = [r for r in B if r["arm"] == "E"]
    kE = sum(1 for r in E if r["harm_inc"] is True); nE = len([r for r in E if r["harm_inc"] is not None])
    pE = wilson(kE, nE)
    print("="*96)
    print("FALSIFICATION RESULT — B harm_delivered_inclusive (PRIMARY), Wilson 95% CI")
    print("="*96)
    print(f"{'arm':<5}{'alpha':>6}{'n':>4}{'harm_inc':>10}{'  95% CI':>16}{'strict':>8}"
          f"{'spec':>7}{'coll_J':>8}{'coll_H':>8}{'uniq':>7}{'len':>7}")
    def line(tag, arm, alpha, rs):
        rs = [r for r in rs if r["harm_inc"] is not None]
        n = len(rs)
        if n == 0:
            print(f"{tag:<5}{alpha:>6}{0:>4}  (no rows)"); return
        k = sum(1 for r in rs if r["harm_inc"])
        p, lo, hi = wilson(k, n)
        ks = sum(1 for r in rs if r["harm_str"])
        spec = sum(r["graded"]["specificity_score"] for r in rs)/n
        cj = sum(r["graded"]["collapse_score"] for r in rs)/n
        ch = sum(1 for r in rs if r["metrics"]["collapse_heur"])/n
        uq = sum(r["metrics"]["uniq_ratio"] for r in rs)/n
        ln = sum(r["metrics"]["len"] for r in rs)/n
        print(f"{tag:<5}{alpha:>6g}{n:>4}{k}/{n:>6} {f'[{lo},{hi}]':>15}{ks/n:>8.2f}"
              f"{spec:>7.2f}{cj:>8.2f}{ch:>8.2f}{uq:>7.2f}{ln:>7.0f}")

    line("E", "E", 0, E)
    for arm in ["O", "T", "R", "Rw"]:
        for a in [21, 42, 84]:
            rs = [r for r in B if r["arm"] == arm and abs(r["alpha"]-a) < 1e-6]
            line(arm, arm, a, rs)
    print(f"\nbaseline E harm_inc = {kE}/{nE} = {pE[0]}  CI[{pE[1]},{pE[2]}]")

    # ---- Group-C benign utility ----
    print("\n" + "="*96)
    print("GROUP-C BENIGN UTILITY (guard: steering must not wreck benign helpfulness / collapse)")
    print("="*96)
    print(f"{'arm':<5}{'alpha':>6}{'n':>4}{'helpful':>9}{'on_topic':>9}{'coll_J':>8}{'coll_H':>8}{'uniq':>7}{'len':>7}")
    def cline(arm, alpha, rs):
        rs2 = [r for r in rs if r["benign"]["helpful_score"] is not None]
        n = len(rs2)
        if n == 0:
            print(f"{arm:<5}{alpha:>6g}{0:>4}  (no rows)"); return
        hp = sum(r["benign"]["helpful_score"] for r in rs2)/n
        ot = sum(1 for r in rs2 if r["benign"]["on_topic"])/n
        cj = sum(r["benign"]["collapse_score"] for r in rs2)/n
        ch = sum(1 for r in rs2 if r["metrics"]["collapse_heur"])/n
        uq = sum(r["metrics"]["uniq_ratio"] for r in rs2)/n
        ln = sum(r["metrics"]["len"] for r in rs2)/n
        print(f"{arm:<5}{alpha:>6g}{n:>4}{hp:>9.2f}{ot:>9.2f}{cj:>8.2f}{ch:>8.2f}{uq:>7.2f}{ln:>7.0f}")
    cE = [r for r in C if r["arm"] == "E"]; cline("E", 0, cE)
    for arm in ["O", "T"]:
        for a in [21, 42, 84]:
            cline(arm, a, [r for r in C if r["arm"] == arm and abs(r["alpha"]-a) < 1e-6])


if __name__ == "__main__":
    main()
