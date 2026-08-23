#!/usr/bin/env python3
"""clockv2_refusal/test1_filter_jaccard.py -- STRICT held-out view for Test 1.

Reads test1_projections.json (already computed, READ-ONLY; loads no model) and re-runs the
form-matched paired analysis after ALSO dropping any held-out prompt within Jaccard>0.4 of the
Phase-1 fit set. XSTest's matched-pair design leaves near-twins across the fit/held-out split
(maxJ up to 0.71), so this is the genuinely-held-out, near-dup-free version of Test 1(A).

    /home/ore99/env_llada/bin/python clockv2_refusal/test1_filter_jaccard.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "clockv2"))
import common  # noqa: E402
sys.path.insert(0, HERE)
from build_vrefusal_v2 import match_pairs, fp, nw  # noqa: E402

OUT = os.path.join(HERE, "data", "test1_projections.json")
FIT = os.path.join(HERE, "data", "balanced_fit.json")
LAYERS = [25, 26, 27]


def paired(d):
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan
    try:
        w = stats.wilcoxon(d)[1]
    except ValueError:
        w = np.nan
    return dz, w, float((d > 0).mean())


def auc(safe, unsafe):
    u = stats.mannwhitneyu(unsafe, safe, alternative="two-sided")[0]
    return float(u / (len(safe) * len(unsafe)))


def main():
    d = json.load(open(OUT))
    fit = json.load(open(FIT))
    fit_all = list(fit["harmless_safe"]) + list(fit["harmful_unsafe"])
    hs, hu = d["held_safe"], d["held_unsafe"]

    # strict held-out: drop near-dups (J>0.4) of the fit set, then re-form-match
    hs_c = [s for s in hs if common.max_jaccard_vs(s, fit_all) <= 0.4]
    hu_c = [u for u in hu if common.max_jaccard_vs(u, fit_all) <= 0.4]
    ps, pu = match_pairs(hs_c, hu_c, kw=2, win=3)
    print("=" * 74)
    print("TEST 1(A) STRICT -- form-matched AND near-dup-filtered (J<=0.4 vs fit)")
    print("=" * 74)
    print(f"held-out after J<=0.4 filter: safe {len(hs_c)}/{len(hs)}  unsafe {len(hu_c)}/{len(hu)}")
    print(f"form-matched pairs (kw=2,win=3): n={len(ps)} | "
          f"FP {sum(fp(x) for x in ps)}/{len(ps)} vs {sum(fp(x) for x in pu)}/{len(pu)} | "
          f"words {np.mean([nw(x) for x in ps]):.1f}/{np.mean([nw(x) for x in pu]):.1f}")
    if len(ps) < 6:
        print("n too small for a meaningful paired estimate -- reporting anyway, flagged.")

    si = [hs.index(x) for x in ps]
    ui = [hu.index(x) for x in pu]
    Ps, Pu = d["proj_v2"]["safe"], d["proj_v2"]["unsafe"]
    Ns, Nu = d["proj_null"]["safe"], d["proj_null"]["unsafe"]

    for rp in ("last", "pmean"):
        print(f"\n--- read point = {rp} ---")
        print(f"{'layer':>5} | {'v_refusal_v2':>26} | {'v_random_null':>18} | {'ratio':>7}")
        print(f"{'':>5} | {'dz':>7} {'AUC':>6} {'%u>s':>5} {'p':>5} | {'dz':>7} {'AUC':>6} | {'x null':>7}")
        for L in LAYERS:
            Ls = str(L)
            d2 = np.array([Pu[rp][Ls][j] for j in ui]) - np.array([Ps[rp][Ls][i] for i in si])
            dn = np.array([Nu[rp][Ls][j] for j in ui]) - np.array([Ns[rp][Ls][i] for i in si])
            dz2, p2, fpos = paired(d2)
            dzn, _, _ = paired(dn)
            a2 = auc([Ps[rp][Ls][i] for i in si], [Pu[rp][Ls][j] for j in ui])
            an = auc([Ns[rp][Ls][i] for i in si], [Nu[rp][Ls][j] for j in ui])
            r = abs(dz2) / abs(dzn) if dzn and not np.isnan(dzn) else np.inf
            print(f"L{L:>4} | {dz2:>+7.2f} {a2:>6.3f} {100*fpos:>4.0f}% {p2:>5.2f} | "
                  f"{dzn:>+7.2f} {an:>6.3f} | {r:>7.1f}")


if __name__ == "__main__":
    main()
