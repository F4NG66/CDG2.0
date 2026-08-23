#!/usr/bin/env python3
"""clockv2/analyze_seed.py — SEED ROBUSTNESS of the final contrast (aggregate only).

Scope: dija - benign_op at L25-L27 only. Read point = prompt_mean (PRIMARY); last_prompt
reported separately (axes +0.65 aligned there, secondary only). v_refusal is not loaded --
REPORT.md §8 Q1 stays logged and not run. tau untouched.

  seed 0  = the temp=0.0 reference already in the record (deterministic argmax).
  seed 1,2 = fresh temp=0.2 captures (Gumbel sampling; the DIJA paper's own config).

The question: does the content component (dz / AUC vs the ~zero null) hold when the
denoising trajectory is resampled? Reports both the aggregate per seed AND the per-id
delta correlation across seeds, which is the stricter test -- two seeds can agree on the
mean while disagreeing completely about which items separate.

    python clockv2/analyze_seed.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

from common import BAND, RESULTS_DIR

LAYERS = [25, 26, 27]
ARMS = ["dija", "benign_op"]


def load_temp0():
    """-> {(cond, layer): [n_ids] step-averaged proj} for vshared/vnull at pmean+last."""
    out = {}
    for f, conds in (("probe_readings.jsonl", {"dija"}),
                     ("probe_readings_benign_op.jsonl", {"benign_op"})):
        acc = {}
        for l in open(os.path.join(RESULTS_DIR, f)):
            r = json.loads(l)
            if r["cond"] not in conds or r["layer"] not in LAYERS:
                continue
            for p in ("vshared", "vnull"):
                for rp in ("pmean", "last"):
                    acc.setdefault((r["cond"], r["layer"], p, rp), {}).setdefault(r["id"], []).append(r[f"{p}_{rp}"])
        for k, per_id in acc.items():
            out[k] = {i: float(np.mean(v)) for i, v in per_id.items()}
    return out


def load_seed(seed):
    acc = {}
    for cond in ARMS:
        p = os.path.join(RESULTS_DIR, f"probe_readings_s{seed}_{cond}.jsonl")
        for l in open(p):
            r = json.loads(l)
            for pr in ("vshared", "vnull"):
                for rp in ("pmean", "last"):
                    acc.setdefault((r["cond"], r["layer"], pr, rp), {}).setdefault(r["id"], []).append(r[f"{pr}_{rp}"])
    return {k: {i: float(np.mean(v)) for i, v in d.items()} for k, d in acc.items()}


def stat(D, L, probe, rp, ids):
    a = np.array([D[("dija", L, probe, rp)][i] for i in ids])
    b = np.array([D[("benign_op", L, probe, rp)][i] for i in ids])
    d = a - b
    dz = d.mean() / d.std(ddof=1)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")[0]
    return {"mean": d.mean(), "dz": dz, "auc": u / (len(a) * len(b)),
            "frac_pos": float((d > 0).mean()), "d": d}


def main():
    src = {0: load_temp0()}
    for s in (1, 2):
        src[s] = load_seed(s)
    ids = sorted(set(src[1][("dija", 25, "vshared", "pmean")]))
    for s in src:
        for L in LAYERS:
            assert set(src[s][("dija", L, "vshared", "pmean")]) >= set(ids), (s, L)
    print("=" * 104)
    print(f"SEED ROBUSTNESS — final contrast (dija - benign_op) | n={len(ids)} paired ids | L25-L27")
    print("  seed 0 = temp=0.0 (deterministic argmax; the existing record)")
    print("  seed 1,2 = temp=0.2 (Gumbel sampling, the DIJA paper's config) — fresh trajectories")
    print("=" * 104)

    for rp, tag in (("pmean", "PRIMARY  read point = prompt_mean"),
                    ("last", "SECONDARY read point = last_prompt (axes +0.65 aligned; secondary only)")):
        print("\n" + "=" * 104)
        print(tag)
        print("=" * 104)
        print(f"{'layer':>5} | {'seed':>16} | {'v_injection_svd':>32} | {'v_random_null':>22}| {'ratio':>6}")
        print(f"{'':>5} | {'':>16} | {'mean_d':>9} {'dz':>7} {'AUC':>6} {'%>0':>5} | "
              f"{'dz':>7} {'AUC':>6} {'%>0':>5}| {'x null':>6}")
        for L in LAYERS:
            rows = []
            for s in (0, 1, 2):
                v = stat(src[s], L, "vshared", rp, ids)
                n = stat(src[s], L, "vnull", rp, ids)
                r = abs(v["dz"]) / abs(n["dz"]) if n["dz"] else np.inf
                lab = "0 (temp=0.0)" if s == 0 else f"{s} (temp=0.2)"
                print(f"L{L:>4} | {lab:>16} | {v['mean']:>+9.2f} {v['dz']:>+7.2f} {v['auc']:>6.3f} "
                      f"{100*v['frac_pos']:>4.0f}% | {n['dz']:>+7.2f} {n['auc']:>6.3f} "
                      f"{100*n['frac_pos']:>4.0f}%| {r:>6.1f}")
                rows.append(v)
            st = np.array([r["dz"] for r in rows[1:]])
            sa = np.array([r["auc"] for r in rows[1:]])
            print(f"{'':>5} | {'temp=0.2 spread':>16} | {'':>9} dz {st.mean():+.2f}+-{st.std(ddof=1):.2f}"
                  f"   AUC {sa.mean():.3f}+-{sa.std(ddof=1):.3f}   "
                  f"(temp=0 ref: dz {rows[0]['dz']:+.2f} AUC {rows[0]['auc']:.3f})")
            print()

    print("=" * 104)
    print("PER-ID STABILITY — the stricter test. READ POINT = prompt_mean (PRIMARY).")
    print("  Two seeds can agree on the mean while disagreeing about WHICH items separate.")
    print("  r = Pearson corr of the per-id delta vector (dija_i - benign_op_i) across seeds.")
    print("=" * 104)
    print(f"{'layer':>5} | {'r(s1,s2)':>9} | {'r(s0,s1)':>9} {'r(s0,s2)':>9} | {'sign agree s1/s2':>17}")
    for L in LAYERS:
        d = {s: stat(src[s], L, "vshared", "pmean", ids)["d"] for s in (0, 1, 2)}
        r12 = stats.pearsonr(d[1], d[2])[0]
        r01 = stats.pearsonr(d[0], d[1])[0]
        r02 = stats.pearsonr(d[0], d[2])[0]
        agree = float(((d[1] > 0) == (d[2] > 0)).mean())
        print(f"L{L:>4} | {r12:>9.3f} | {r01:>9.3f} {r02:>9.3f} | {100*agree:>16.0f}%")

    print("\n" + "=" * 104)
    print("VERDICT INPUTS ONLY — no verdict written here.")
    print("=" * 104)


if __name__ == "__main__":
    main()
