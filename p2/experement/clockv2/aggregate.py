#!/usr/bin/env python3
"""clockv2/aggregate.py — PHASE 3 (aggregate only; writes NO verdict).

Consumes results/probe_readings.jsonl (100 ids x 2 cond x 16 layers x 128 steps) and
emits the aggregate tables. Analysis is pinned to three locked decisions:

  1. PAIRED PER-ID DELTAS.  Raw projections are un-centered and dominated by a large
     common-mode component (h . v where h has a big mean), which is why harmful clean
     prompts can read negative on the refusal axis. Only  d_i = dija_i - clean_i  is
     interpretable. Every headline number here is a paired delta.
  2. tau_nn WHEREVER CALIBRATION MATTERS.  tau_read (softmax-weighted) is compressed to
     ~[0.28,0.77] by soft-argmax averaging; tau_nn (nearest mu_step) is uncompressed.
     disc_* is therefore computed from tau_nn. tau_read is carried for continuity only.
  3. REGRESS disc ON n_inject BEFORE ANY DETECTION CLAIM.  DIJA raises total masks to
     128+n_inject, so the per-step transfer budget (~1.5 tok/step) drains the response
     canvas FASTER by arithmetic alone -- nothing internal. If n_inject explains the
     discrepancy, there is no clock finding.

READ POINTS ARE NEVER MIXED IN A COLUMN. prompt_mean is PRIMARY (v_refusal/v_injection
are near-orthogonal there, cos=-0.05). last_prompt is SECONDARY and every last-token
table restates that the two axes are +0.65 aligned there, so a "refusal" move at `last`
is not separable from an "injection" move. resp_mean is exploratory only.

    python clockv2/aggregate.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

from common import BAND, RESULTS_DIR, STEPS

D_MODEL = 4096
PROBES = ("vshared", "vrefusal", "vnull")      # headline, honest-report, floor
LABEL = {"vshared": "v_injection_svd", "vrefusal": "v_refusal", "vnull": "v_random_null"}


def load(path):
    """-> arrays[metric] of shape [n_ids, 2, n_layers, STEPS]; cond axis 0=clean 1=dija."""
    rows = [json.loads(l) for l in open(path)]
    ids = sorted({r["id"] for r in rows})
    lay = {L: i for i, L in enumerate(BAND)}
    idx = {c: i for i, c in enumerate(ids)}
    cnd = {"clean": 0, "dija": 1}
    metrics = [k for k in rows[0] if k not in ("id", "cond", "layer", "step")]
    A = {m: np.full((len(ids), 2, len(BAND), STEPS), np.nan) for m in metrics}
    for r in rows:
        i, c, l, t = idx[r["id"]], cnd[r["cond"]], lay[r["layer"]], r["step"]
        for m in metrics:
            A[m][i, c, l, t] = r[m]
    miss = {m: int(np.isnan(A[m]).sum()) for m in metrics if np.isnan(A[m]).any()}
    return ids, A, miss


def paired(d):
    """d: [n] paired deltas -> dict of effect size, sign consistency, Wilcoxon p."""
    n = len(d)
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan            # Cohen's dz (paired)
    try:
        w_p = stats.wilcoxon(d)[1]                       # no normality assumption
    except ValueError:
        w_p = np.nan
    return {"mean": d.mean(), "sd": sd, "dz": dz, "p": w_p,
            "frac_pos": float((d > 0).mean()), "n": n}


def auc(a, b):
    """P(b > a) via Mann-Whitney U; 0.5 = no separation, unpaired view."""
    u = stats.mannwhitneyu(b, a, alternative="two-sided")[0]
    return u / (len(a) * len(b))


def fmt_p(p):
    if np.isnan(p):
        return "  n/a"
    return "<1e-16" if p < 1e-16 else f"{p:.2g}"


def main():
    path = os.path.join(RESULTS_DIR, "probe_readings.jsonl")
    ids, A, miss = load(path)
    n = len(ids)
    print("=" * 100)
    print(f"PHASE 3 AGGREGATE — n={n} paired ids | {len(BAND)} layers L{BAND[0]}-L{BAND[-1]} "
          f"| {STEPS} steps | temp=0")
    print(f"missing cells: {miss if miss else 'none'}")
    print("=" * 100)

    n_inj = A["n_inject"][:, 1, 0, 0]                    # per id (dija); clean is 0 by defn
    plen_c, plen_d = A["prompt_len"][:, 0, 0, 0], A["prompt_len"][:, 1, 0, 0]
    print(f"\nn_inject over {n} ids: mean={n_inj.mean():.1f} sd={n_inj.std():.1f} "
          f"min={n_inj.min():.0f} max={n_inj.max():.0f}")
    print(f"prompt_len: clean {plen_c.mean():.1f}+-{plen_c.std():.1f}  "
          f"dija {plen_d.mean():.1f}+-{plen_d.std():.1f}")

    # ---------------------------------------------------------------- physical canvas
    print("\n" + "-" * 100)
    print("TABLE 1 — physical canvas clock (mask_ratio_t, response-only). No read point: "
          "this is token bookkeeping, not activations.")
    print("-" * 100)
    mr = A["mask_ratio_t"][:, :, 0, :]                    # layer-invariant
    print(f"{'step':>5} | {'clean mask_ratio':>17} {'dija mask_ratio':>17} | "
          f"{'clean tau_from_mask':>19} {'dija tau_from_mask':>18}")
    for t in (0, 16, 32, 64, 96, 112, 127):
        print(f"{t:>5} | {mr[:,0,t].mean():>17.3f} {mr[:,1,t].mean():>17.3f} | "
              f"{1-mr[:,0,t].mean():>19.3f} {1-mr[:,1,t].mean():>18.3f}")
    done_c = [np.argmax(mr[i, 0] <= 0) if (mr[i, 0] <= 0).any() else STEPS for i in range(n)]
    done_d = [np.argmax(mr[i, 1] <= 0) if (mr[i, 1] <= 0).any() else STEPS for i in range(n)]
    print(f"\nfirst step where response canvas is FULLY filled (mask_ratio==0):")
    print(f"  clean: median={np.median(done_c):.0f}  (never-fills count={sum(1 for x in done_c if x==STEPS)}/{n})")
    print(f"  dija : median={np.median(done_d):.0f}  (never-fills count={sum(1 for x in done_d if x==STEPS)}/{n})")
    r_dn = stats.pearsonr(n_inj, np.array(done_d, float))
    print(f"  corr(n_inject, dija fill-completion step): r={r_dn[0]:+.3f} p={fmt_p(r_dn[1])}")
    print("  ^ if strongly negative: more injected blanks => canvas finishes EARLIER, by arithmetic.")

    # ------------------------------------------------- headline: paired deltas @ pmean
    for rp, tag in (("pmean", "PRIMARY  read point = prompt_mean"),
                    ("last", "SECONDARY read point = last_prompt")):
        print("\n" + "-" * 100)
        ttl = "TABLE 2" if rp == "pmean" else "TABLE 3"
        print(f"{ttl} — paired delta (DIJA - clean), step-averaged. {tag}")
        if rp == "last":
            print("  CAVEAT: at last_prompt, cos(v_refusal, v_injection_svd) = +0.65 — the two axes")
            print("  are NOT separable here. A 'refusal' move at this read point may be an injection")
            print("  move bleeding through. This table is secondary; prompt_mean (Table 2) is primary.")
        print("-" * 100)
        print(f"{'layer':>5} | " + " | ".join(
            f"{LABEL[p]:>28}" for p in PROBES))
        print(f"{'':>5} | " + " | ".join(f"{'mean_d':>8} {'dz':>6} {'%>0':>5} {'p':>6}"
                                         for _ in PROBES))
        best = {}
        for li, L in enumerate(BAND):
            cells = []
            for p in PROBES:
                d = (A[f"{p}_{rp}"][:, 1, li, :] - A[f"{p}_{rp}"][:, 0, li, :]).mean(1)
                s = paired(d)
                best.setdefault(p, []).append((abs(s["dz"]), L, s))
                cells.append(f"{s['mean']:>+8.2f} {s['dz']:>+6.2f} {100*s['frac_pos']:>4.0f}% "
                             f"{fmt_p(s['p']):>6}")
            print(f"L{L:>4} | " + " | ".join(cells))

        print(f"\n  strongest layer per probe ({rp}):")
        for p in PROBES:
            dz, L, s = max(best[p])
            print(f"    {LABEL[p]:<18} L{L}: mean_d={s['mean']:+.2f} dz={s['dz']:+.2f} "
                  f"{100*s['frac_pos']:.0f}% of ids same-sign  p={fmt_p(s['p'])}")

        # floor comparison + isotropic shift-norm sanity bound
        dzs = {p: max(best[p])[0] for p in PROBES}
        print(f"\n  FLOOR CHECK ({rp}): |dz| ratio vs v_random_null")
        for p in ("vshared", "vrefusal"):
            r = dzs[p] / dzs["vnull"] if dzs["vnull"] > 0 else np.inf
            print(f"    {LABEL[p]:<18} {r:>6.1f}x the null floor")
        dn = abs(max(best["vnull"])[2]["mean"])
        shift = dn * np.sqrt(np.pi * D_MODEL / 2)
        ds = abs(max(best["vshared"])[2]["mean"])
        print(f"\n  isotropic sanity bound ({rp}): a random unit axis absorbs |shift|*sqrt(2/(pi*d)).")
        print(f"    null |mean_d|={dn:.2f} => implied total shift norm ~{shift:.0f}")
        print(f"    v_injection_svd |mean_d|={ds:.2f} => captures ~{100*ds/shift:.0f}% of that shift in ONE of {D_MODEL} dims")
        print(f"    (heuristic only — assumes isotropy, which the shift is not.)")

    # ---------------------------------------------------------- v_refusal honest call
    print("\n" + "-" * 100)
    print("TABLE 4 — v_refusal, per-step paired delta. READ POINT = prompt_mean (PRIMARY).")
    print("  Question: does DIJA move the prompt OFF the refusal axis? Sign convention:")
    print("  +v = harmful direction (mean_harmful - mean_harmless). Only the DELTA is meaningful.")
    print("-" * 100)
    print(f"{'layer':>5} | " + " ".join(f"{'t='+str(t):>13}" for t in (0, 32, 64, 96, 127)))
    for li, L in enumerate(BAND):
        cells = []
        for t in (0, 32, 64, 96, 127):
            d = A[f"vrefusal_pmean"][:, 1, li, t] - A[f"vrefusal_pmean"][:, 0, li, t]
            s = paired(d)
            star = "*" if (not np.isnan(s["p"]) and s["p"] < 0.001) else " "
            cells.append(f"{s['mean']:>+8.2f}/{s['dz']:>+.2f}{star}")
        print(f"L{L:>4} | " + " ".join(f"{c:>13}" for c in cells))
    print("  (cell = mean_delta/dz ; * = Wilcoxon p<0.001)")

    # ------------------------------------------------------------------ tau, secondary
    print("\n" + "-" * 100)
    print("TABLE 5 — tau (SECONDARY). READ POINT = resp_mean (the only valid one: mu_step was")
    print("  fit there). disc uses tau_nn, NOT tau_read.")
    print("  CIRCULARITY CAVEAT: mu_step was fit on step index, so tau_read/tau_nn recovering")
    print("  step is near-tautological. A 'tau shift' under DIJA is expected from the canvas")
    print("  fill-level / n_inject confound and is NOT evidence of a nontrivial internal clock.")
    print("  COMPRESSION CAVEAT: tau_read (soft) saturates ~[0.28,0.77]; disc_read therefore")
    print("  carries a large systematic bias PRESENT IN CLEAN TOO. tau_nn is uncompressed.")
    print("-" * 100)
    for li, L in enumerate(BAND):
        if L not in (25, 29):
            continue
        print(f"\n  --- L{L} (resp_mean) ---")
        print(f"  {'step':>5} | {'tau_nn clean':>12} {'tau_nn dija':>12} | "
              f"{'tau_read clean':>14} {'tau_read dija':>13} | {'disc_nn clean':>13} {'disc_nn dija':>12}")
        for t in (0, 32, 64, 96, 127):
            tn_c, tn_d = A["tau_nn"][:, 0, li, t].mean(), A["tau_nn"][:, 1, li, t].mean()
            tr_c, tr_d = A["tau_read"][:, 0, li, t].mean(), A["tau_read"][:, 1, li, t].mean()
            dc = (A["tau_nn"][:, 0, li, t] - A["tau_from_mask"][:, 0, li, t]).mean()
            dd = (A["tau_nn"][:, 1, li, t] - A["tau_from_mask"][:, 1, li, t]).mean()
            print(f"  {t:>5} | {tn_c:>12.3f} {tn_d:>12.3f} | {tr_c:>14.3f} {tr_d:>13.3f} | "
                  f"{dc:>+13.3f} {dd:>+12.3f}")

    # ------------------------------------- THE gate: regress disc on n_inject
    print("\n" + "-" * 100)
    print("TABLE 6 — THE CONFOUND GATE. Regress paired delta-disc on n_inject across ids.")
    print("  delta_disc_i = mean_t(disc_nn dija_i) - mean_t(disc_nn clean_i);  x = n_inject_i")
    print("  If R^2 is high, the 'clock discrepancy' IS the n_inject arithmetic — no finding.")
    print("-" * 100)
    print(f"{'layer':>5} | {'slope':>10} {'R^2':>7} {'r':>7} {'p':>8} | "
          f"{'resid mean':>10} {'resid dz':>9} {'resid p':>8}")
    for li, L in enumerate(BAND):
        dd = ((A["tau_nn"][:, 1, li, :] - A["tau_from_mask"][:, 1, li, :]).mean(1)
              - (A["tau_nn"][:, 0, li, :] - A["tau_from_mask"][:, 0, li, :]).mean(1))
        lr = stats.linregress(n_inj, dd)
        resid = dd - (lr.slope * n_inj + lr.intercept)
        rs = paired(resid)
        print(f"L{L:>4} | {lr.slope:>+10.5f} {lr.rvalue**2:>7.3f} {lr.rvalue:>+7.3f} "
              f"{fmt_p(lr.pvalue):>8} | {rs['mean']:>+10.4f} {rs['dz']:>+9.2f} {fmt_p(rs['p']):>8}")
    print("  resid* = delta_disc with the n_inject trend removed. Residual mean is ~0 BY")
    print("  CONSTRUCTION (OLS centres it), so resid dz/p here are diagnostics of spread,")
    print("  NOT an independent test of a leftover clock effect.")

    # ----------------------------- does the HEADLINE survive the same confound?
    print("\n" + "-" * 100)
    print("TABLE 7 — does the headline survive n_inject? Regress paired delta on n_inject.")
    print("  READ POINT = prompt_mean (PRIMARY). Longer DIJA prompts (n_inject blanks) could")
    print("  drive a common-mode shift; v_random_null is the control for exactly that.")
    print("-" * 100)
    print(f"{'probe':>18} {'layer':>5} | {'r(delta,n_inject)':>18} {'R^2':>7} | "
          f"{'partial |dz| after':>18}")
    for p in PROBES:
        for li, L in enumerate(BAND):
            if L not in (25, 29):
                continue
            d = (A[f"{p}_pmean"][:, 1, li, :] - A[f"{p}_pmean"][:, 0, li, :]).mean(1)
            lr = stats.linregress(n_inj, d)
            resid = d - (lr.slope * n_inj + lr.intercept)
            # intercept = the delta at n_inject=0, i.e. the part NOT explained by blank count
            print(f"{LABEL[p]:>18} L{L:>4} | {lr.rvalue:>+18.3f} {lr.rvalue**2:>7.3f} | "
                  f"intercept={lr.intercept:>+8.2f} (delta at n_inject=0)")

    print("\n" + "=" * 100)
    print("AGGREGATE COMPLETE — no verdict written. REPORT.md pending review.")
    print("=" * 100)


if __name__ == "__main__":
    main()
