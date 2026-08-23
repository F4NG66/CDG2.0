#!/usr/bin/env python3
"""clockv2/analyze_twoarm_seed.py — SEED-VERIFIED TWO-ARM AGREEMENT (§7 mitigation).

The §7 authored-vs-derived confound is mitigated by one argument: arm 2 (benign, DATASET-
sourced) and arm 3 (benign_op, AUTHORED) share no items, differ in register by construction,
and yet land on nearly the same content component. If an authorship signature were driving
the content term, the authored arm would have to coincidentally reproduce the dataset-sourced
one. This script puts that argument on the same seed footing as everything else.

  content via arm 2 = dija - benign        (dataset-sourced benign; stance MISmatched)
  content via arm 3 = dija - benign_op     (authored benign;        stance matched)

The gap between them is the STANCE effect; the agreement is what survives it.

  seed 0   = temp=0.0 (the deterministic record)
  seed 1,2 = temp=0.2 (Gumbel sampling; the DIJA paper's own config)

Read point = prompt_mean (PRIMARY). last_prompt is not used: the axes are +0.65 aligned
there. v_refusal and tau are untouched and never loaded.

CAVEAT ON PER-ID CORRELATION: content_arm2 and content_arm3 share the `dija` term, so a
per-id correlation between them is inflated by that common term and is NOT evidence of
agreement. The agreement claim is about the MAGNITUDE of the two content components, which
is what this script reports. Per-id r is shown for the SEED stability of each arm separately
(same arm, different seeds), where no such artifact exists.

    python clockv2/analyze_twoarm_seed.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

from common import RESULTS_DIR

LAYERS = [25, 26, 27]
ARMS = ["clean", "dija", "benign", "benign_op"]


def _acc(rows_iter, keep):
    acc = {}
    for r in rows_iter:
        if r["cond"] not in keep or r["layer"] not in LAYERS:
            continue
        acc.setdefault((r["cond"], r["layer"]), {}).setdefault(r["id"], []).append(r["vshared_pmean"])
    return {k: {i: float(np.mean(v)) for i, v in d.items()} for k, d in acc.items()}


def load_seed0():
    out = {}
    for f, keep in (("probe_readings.jsonl", {"clean", "dija"}),
                    ("probe_readings_benign.jsonl", {"benign"}),
                    ("probe_readings_benign_op.jsonl", {"benign_op"})):
        out.update(_acc((json.loads(l) for l in open(os.path.join(RESULTS_DIR, f))), keep))
    return out


def load_seed(s):
    out = {}
    for cond in ARMS:
        p = os.path.join(RESULTS_DIR, f"probe_readings_s{s}_{cond}.jsonl")
        out.update(_acc((json.loads(l) for l in open(p)), {cond}))
    return out


def vec(D, cond, L, ids):
    return np.array([D[(cond, L)][i] for i in ids])


def dz(d):
    return d.mean() / d.std(ddof=1)


def main():
    src = {0: load_seed0(), 1: load_seed(1), 2: load_seed(2)}
    ids = sorted(set(src[1][("benign", 25)]))
    for s in src:
        for c in ARMS:
            for L in LAYERS:
                assert set(src[s][(c, L)]) >= set(ids), (s, c, L)

    print("=" * 108)
    print(f"TWO-ARM AGREEMENT — seed-verified | n={len(ids)} paired ids | READ POINT = prompt_mean (PRIMARY)")
    print("  content via arm 2 = dija - benign     (DATASET-sourced benign, stance mismatched)")
    print("  content via arm 3 = dija - benign_op  (AUTHORED benign,        stance matched)")
    print("  gap = arm2 - arm3 = the stance effect;  agreement = how close the two land despite")
    print("  sharing no items and coming from different sources (the §7 confound mitigation).")
    print("=" * 108)

    for L in LAYERS:
        print(f"\n--- L{L} (prompt_mean) ---")
        print(f"{'seed':>16} | {'content via arm2':>18} | {'content via arm3':>18} | "
              f"{'gap':>8} {'arm3/arm2':>10} {'gap % of arm2':>14}")
        print(f"{'':>16} | {'mean_d':>9} {'dz':>7} | {'mean_d':>9} {'dz':>7} |")
        rows = []
        for s in (0, 1, 2):
            dj = vec(src[s], "dija", L, ids)
            c2 = dj - vec(src[s], "benign", L, ids)
            c3 = dj - vec(src[s], "benign_op", L, ids)
            gap = c2.mean() - c3.mean()
            ratio = c3.mean() / c2.mean()
            gpct = 100 * gap / c2.mean()
            rows.append((c2.mean(), c3.mean(), gap, ratio, gpct))
            lab = "0 (temp=0.0)" if s == 0 else f"{s} (temp=0.2)"
            print(f"{lab:>16} | {c2.mean():>+9.2f} {dz(c2):>+7.2f} | {c3.mean():>+9.2f} {dz(c3):>+7.2f} | "
                  f"{gap:>+8.2f} {ratio:>10.3f} {gpct:>13.1f}%")
        a = np.array(rows[1:])
        print(f"{'temp=0.2 spread':>16} | {a[:,0].mean():>+9.2f}{'':>8} | {a[:,1].mean():>+9.2f}{'':>8} | "
              f"{a[:,2].mean():>+8.2f} {a[:,3].mean():>7.3f}±{a[:,3].std(ddof=1):.3f} "
              f"{a[:,4].mean():>8.1f}%±{a[:,4].std(ddof=1):.1f}")
        print(f"{'temp=0 ref':>16} | {rows[0][0]:>+9.2f}{'':>8} | {rows[0][1]:>+9.2f}{'':>8} | "
              f"{rows[0][2]:>+8.2f} {rows[0][3]:>10.3f} {rows[0][4]:>13.1f}%")

    print("\n" + "=" * 108)
    print("SPLIT VIA BOTH ARMS, PER SEED. READ POINT = prompt_mean (PRIMARY).")
    print("  (dija-clean) == (benign_x - clean) + (dija - benign_x), exact per id, asserted below.")
    print("=" * 108)
    print(f"{'layer':>5} | {'seed':>16} | {'via arm2 struct%/content%':>26} | {'via arm3 struct%/content%':>26}")
    for L in LAYERS:
        pct2, pct3 = [], []
        for s in (0, 1, 2):
            cl, dj = vec(src[s], "clean", L, ids), vec(src[s], "dija", L, ids)
            tot = dj - cl
            out = []
            for arm in ("benign", "benign_op"):
                bx = vec(src[s], arm, L, ids)
                st, ct = bx - cl, dj - bx
                assert np.allclose(tot, st + ct, atol=1e-9), "identity violated"
                out.append((100 * st.mean() / tot.mean(), 100 * ct.mean() / tot.mean()))
            pct2.append(out[0]); pct3.append(out[1])
            lab = "0 (temp=0.0)" if s == 0 else f"{s} (temp=0.2)"
            print(f"L{L:>4} | {lab:>16} | {out[0][0]:>11.1f}% / {out[0][1]:<12.1f} | "
                  f"{out[1][0]:>11.1f}% / {out[1][1]:<12.1f}")
        A2, A3 = np.array(pct2[1:]), np.array(pct3[1:])
        print(f"L{L:>4} | {'temp=0.2 spread':>16} | {A2[:,0].mean():>7.1f}±{A2[:,0].std(ddof=1):.1f}% / "
              f"{A2[:,1].mean():.1f}±{A2[:,1].std(ddof=1):.1f}   | "
              f"{A3[:,0].mean():>7.1f}±{A3[:,0].std(ddof=1):.1f}% / {A3[:,1].mean():.1f}±{A3[:,1].std(ddof=1):.1f}")
        print()

    print("=" * 108)
    print("PER-ID SEED STABILITY OF EACH ARM'S CONTENT TERM (same arm, across the two temp=0.2 seeds).")
    print("  NOT a cross-arm correlation: arm2 and arm3 content share the `dija` term, so a")
    print("  cross-arm per-id r would be inflated by that and is deliberately not reported.")
    print("=" * 108)
    print(f"{'layer':>5} | {'r(s1,s2) arm2':>14} {'sign agree':>11} | {'r(s1,s2) arm3':>14} {'sign agree':>11}")
    for L in LAYERS:
        o = []
        for arm in ("benign", "benign_op"):
            c1 = vec(src[1], "dija", L, ids) - vec(src[1], arm, L, ids)
            c2 = vec(src[2], "dija", L, ids) - vec(src[2], arm, L, ids)
            o.append((stats.pearsonr(c1, c2)[0], 100 * float(((c1 > 0) == (c2 > 0)).mean())))
        print(f"L{L:>4} | {o[0][0]:>14.3f} {o[0][1]:>10.0f}% | {o[1][0]:>14.3f} {o[1][1]:>10.0f}%")

    print("\n" + "=" * 108)
    print("NO VERDICT WRITTEN. v_refusal and tau untouched.")
    print("=" * 108)


if __name__ == "__main__":
    main()
