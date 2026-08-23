#!/usr/bin/env python3
"""clockv2/analyze_split_seed.py — SEED VARIANCE OF THE STRUCTURE/CONTENT SPLIT (§2 core).

Puts the paper's quantitative core on the same seed footing as the headline. The split is
an exact per-id identity, not a fit:

    (dija - clean)  ==  (benign_op - clean)  +  (dija - benign_op)
       total                structure                content

so it is recomputed wholesale at each seed and the identity is ASSERTED, not assumed.

Arms: clean, dija, benign_op (arm 3). benign (arm 2) is not involved — it is a reference
contrast, not part of the arm-3 split.
  seed 0   = temp=0.0 (the deterministic record)
  seed 1,2 = temp=0.2 (Gumbel sampling; the DIJA paper's own config)

Read point = prompt_mean (PRIMARY) throughout. last_prompt is NOT used for the split: the
axes are +0.65 aligned there, and a split computed on non-separable axes is meaningless.
v_refusal and tau are untouched.

    python clockv2/analyze_split_seed.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

from common import RESULTS_DIR

LAYERS = [25, 26, 27]
ARMS = ["clean", "dija", "benign_op"]


def _acc(rows_iter, keep):
    acc = {}
    for r in rows_iter:
        if r["cond"] not in keep or r["layer"] not in LAYERS:
            continue
        acc.setdefault((r["cond"], r["layer"]), {}).setdefault(r["id"], []).append(r["vshared_pmean"])
    return {k: {i: float(np.mean(v)) for i, v in d.items()} for k, d in acc.items()}


def load_seed0():
    """temp=0.0 record: clean+dija from probe_readings.jsonl, benign_op from its own file."""
    out = {}
    for f, keep in (("probe_readings.jsonl", {"clean", "dija"}),
                    ("probe_readings_benign_op.jsonl", {"benign_op"})):
        gen = (json.loads(l) for l in open(os.path.join(RESULTS_DIR, f)))
        out.update(_acc(gen, keep))
    return out


def load_seed(s):
    out = {}
    for cond in ARMS:
        p = os.path.join(RESULTS_DIR, f"probe_readings_s{s}_{cond}.jsonl")
        gen = (json.loads(l) for l in open(p))
        out.update(_acc(gen, {cond}))
    return out


def terms(D, L, ids):
    """-> per-id (total, structure, content) delta vectors. Identity holds exactly."""
    cl = np.array([D[("clean", L)][i] for i in ids])
    dj = np.array([D[("dija", L)][i] for i in ids])
    bo = np.array([D[("benign_op", L)][i] for i in ids])
    total, struct, content = dj - cl, bo - cl, dj - bo
    assert np.allclose(total, struct + content, atol=1e-9), "identity violated"
    return total, struct, content


def dz(d):
    return d.mean() / d.std(ddof=1)


def main():
    src = {0: load_seed0(), 1: load_seed(1), 2: load_seed(2)}
    ids = sorted(set(src[1][("clean", 25)]))
    for s in src:
        for c in ARMS:
            for L in LAYERS:
                assert set(src[s][(c, L)]) >= set(ids), (s, c, L)

    print("=" * 106)
    print(f"STRUCTURE / CONTENT SPLIT — seed variance | n={len(ids)} paired ids | via arm 3 (benign_op)")
    print("  identity (exact, per id):  (dija-clean) == (benign_op-clean) + (dija-benign_op)")
    print("  READ POINT = prompt_mean (PRIMARY). seed 0 = temp=0.0; seeds 1,2 = temp=0.2 resamples.")
    print("=" * 106)

    for L in LAYERS:
        print(f"\n--- L{L} (prompt_mean) ---")
        print(f"{'seed':>16} | {'total':>17} | {'structure':>17} | {'content':>17} | {'struct%':>8} {'content%':>9}")
        print(f"{'':>16} | {'mean_d':>8} {'dz':>7} | {'mean_d':>8} {'dz':>7} | {'mean_d':>8} {'dz':>7} |")
        pct = []
        for s in (0, 1, 2):
            t, st, ct = terms(src[s], L, ids)
            sp, cp = 100 * st.mean() / t.mean(), 100 * ct.mean() / t.mean()
            pct.append((sp, cp))
            lab = "0 (temp=0.0)" if s == 0 else f"{s} (temp=0.2)"
            print(f"{lab:>16} | {t.mean():>+8.2f} {dz(t):>+7.2f} | {st.mean():>+8.2f} {dz(st):>+7.2f} | "
                  f"{ct.mean():>+8.2f} {dz(ct):>+7.2f} | {sp:>7.1f}% {cp:>8.1f}%")
        a = np.array(pct[1:])
        print(f"{'temp=0.2 spread':>16} | {'':>17} | {'':>17} | {'':>17} | "
              f"{a[:,0].mean():>6.1f}%±{a[:,0].std(ddof=1):.1f} {a[:,1].mean():>6.1f}%±{a[:,1].std(ddof=1):.1f}")
        print(f"{'temp=0 ref':>16} | {'':>17} | {'':>17} | {'':>17} | "
              f"{pct[0][0]:>7.1f}% {pct[0][1]:>8.1f}%")

    print("\n" + "=" * 106)
    print("PER-ID STABILITY OF EACH TERM. READ POINT = prompt_mean (PRIMARY).")
    print("  r = Pearson corr of the per-id delta vector across the two temp=0.2 seeds.")
    print("=" * 106)
    print(f"{'layer':>5} | {'r total':>8} {'r struct':>9} {'r content':>10} | "
          f"{'sign agree: total':>18} {'struct':>8} {'content':>8}")
    for L in LAYERS:
        t1, s1, c1 = terms(src[1], L, ids)
        t2, s2, c2 = terms(src[2], L, ids)
        r = lambda a, b: stats.pearsonr(a, b)[0]
        ag = lambda a, b: 100 * float(((a > 0) == (b > 0)).mean())
        print(f"L{L:>4} | {r(t1,t2):>8.3f} {r(s1,s2):>9.3f} {r(c1,c2):>10.3f} | "
              f"{ag(t1,t2):>17.0f}% {ag(s1,s2):>7.0f}% {ag(c1,c2):>7.0f}%")

    print("\n" + "=" * 106)
    print("NO VERDICT WRITTEN. v_refusal and tau untouched.")
    print("=" * 106)


if __name__ == "__main__":
    main()
