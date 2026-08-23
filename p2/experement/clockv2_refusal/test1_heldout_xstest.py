#!/usr/bin/env python3
"""clockv2_refusal/test1_heldout_xstest.py -- TEST 1 (path C): does v_refusal_v2 detect harm?

READ-ONLY. Prompt-only forward passes (same method the axis was fit with); zero steering; no
activation modified (assert_readonly_hooks before every forward); no authored text. New folder
only: imports clockv2/common read-only, reads clockv2/probes read-only, writes only under
clockv2_refusal/.

DESIGN. Project v_refusal_v2 on HELD-OUT XSTest (the 163 safe + 113 unsafe NOT used in the
Phase-1 fit -- verified disjoint here) and ask whether UNSAFE reads higher than SAFE on the
axis, vs the v_random_null floor, at L25-L27.

FORM CONTROL -- reported honestly, not assumed. The Phase-1 fit consumed XSTest SAFE's
operational/first-person prompts, so the FULL held-out split is NOT form-matched (safe ~10%
first-person vs unsafe ~68%). Two views, both reported:
  (A) FORM-MATCHED held-out pairs (opener+FP+word-len matched, all marginals p>0.9): the
      form-CLEAN test. Small n (~22 pairs) -- flagged. Paired per-id deltas remove the large
      common-mode component (aggregate.py note 1), so dz is interpretable.
  (B) FULL held-out (163 vs 113), unpaired: form-CONFOUNDED ceiling, shown only so the gap
      between (A) and (B) makes any form-dependence visible.

FLOOR. v_random_null (seed-1337 unit Gaussian, clockv2/probes) projected the SAME way. A
genuine harm detector must clear it. Pre-registered bar: clears the floor by clockv2's margin
(v_injection_svd reached |dz|/|dz_null| ~ x11-x53 at L25-L26 on its decisive contrast).

DISTRIBUTION-SHIFT NOTE. The axis was fit on short (~8-word) unscaffolded XSTest prompts and
is READ here on the same short unscaffolded distribution -- so Test 1 carries NO scaffolding
shift (that shift is Test 2's concern). Reported for completeness.

Read points are never mixed in a column:
  last  = final prompt token        (v_refusal_v2 PRIMARY extraction point)
  pmean = mean over prompt tokens   (auxiliary)

    /home/ore99/env_llada/bin/python clockv2_refusal/test1_heldout_xstest.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "clockv2"))
import common  # noqa: E402  (read-only)
sys.path.insert(0, HERE)
from build_vrefusal_v2 import match_pairs, fp, op, nw, nc  # noqa: E402

XSTEST = os.path.join(HERE, "data", "xstest.json")
FIT = os.path.join(HERE, "data", "balanced_fit.json")
V2 = os.path.join(HERE, "probes", "v_refusal_v2.pt")
VNULL = os.path.join(EXP, "clockv2", "probes", "v_random_null.pt")
OUT = os.path.join(HERE, "data", "test1_projections.json")
LAYERS = [25, 26, 27]


def paired(d):
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan
    try:
        w = stats.wilcoxon(d)[1]
    except ValueError:
        w = np.nan
    return {"mean": float(d.mean()), "dz": float(dz), "p": float(w),
            "frac_pos": float((d > 0).mean()), "n": len(d)}


def auc(safe, unsafe):
    """P(unsafe > safe); 0.5 = no separation."""
    u = stats.mannwhitneyu(unsafe, safe, alternative="two-sided")[0]
    return float(u / (len(safe) * len(unsafe)))


def cohen_d(safe, unsafe):
    safe, unsafe = np.asarray(safe, float), np.asarray(unsafe, float)
    ns, nu = len(safe), len(unsafe)
    sp = np.sqrt(((ns - 1) * safe.var(ddof=1) + (nu - 1) * unsafe.var(ddof=1)) / (ns + nu - 2))
    return float((unsafe.mean() - safe.mean()) / sp) if sp > 0 else np.nan


def collect(model, tok, blocks, texts, v2, vnull, device, label):
    """-> per-prompt projections {rp: {L: [proj_v2], 'null':{L:[proj_null]}}}."""
    cap = common.PromptOnlyCapture(blocks)
    common.assert_readonly_hooks(blocks)
    vr = vnull.double()
    P = {"last": {L: [] for L in common.BAND}, "pmean": {L: [] for L in common.BAND}}
    N = {"last": {L: [] for L in common.BAND}, "pmean": {L: [] for L in common.BAND}}
    try:
        for i, t in enumerate(texts):
            common.prompt_forward(model, tok, t, cap, device=device)
            for L in common.BAND:
                hl, hm = cap.last[L].double(), cap.mean[L].double()
                P["last"][L].append(float(hl @ v2["v_last"][L].double()))
                P["pmean"][L].append(float(hm @ v2["v_mean"][L].double()))
                N["last"][L].append(float(hl @ vr))
                N["pmean"][L].append(float(hm @ vr))
            if (i + 1) % 40 == 0:
                print(f"  [{label}] {i+1}/{len(texts)}", flush=True)
    finally:
        cap.close()
    return P, N


def main():
    device = "cuda"
    xs = json.load(open(XSTEST))
    fit = json.load(open(FIT))
    used_safe, used_unsafe = set(fit["harmless_safe"]), set(fit["harmful_unsafe"])
    held_safe = [s for s in xs["safe"] if s not in used_safe]
    held_unsafe = [u for u in xs["unsafe"] if u not in used_unsafe]

    # ---- zero-overlap verification (exact + Jaccard vs the fit set) ----
    fit_all = list(used_safe) + list(used_unsafe)
    ex_s = len(set(held_safe) & used_safe)
    ex_u = len(set(held_unsafe) & used_unsafe)
    mj_s = max(common.max_jaccard_vs(s, fit_all) for s in held_safe)
    mj_u = max(common.max_jaccard_vs(u, fit_all) for u in held_unsafe)
    print("=" * 78)
    print("TEST 1 (path C) -- v_refusal_v2 on HELD-OUT XSTest (safe vs unsafe)")
    print("=" * 78)
    print(f"\nheld-out: safe={len(held_safe)} unsafe={len(held_unsafe)} "
          f"(disjoint from fit: exact overlap safe={ex_s} unsafe={ex_u}; "
          f"maxJ vs fit safe={mj_s:.2f} unsafe={mj_u:.2f})")
    assert ex_s == 0 and ex_u == 0, "held-out overlaps the fit set"

    # ---- form-matched held-out pair set (the form-CLEAN view) ----
    ps, pu = match_pairs(held_safe, held_unsafe, kw=2, win=3)
    fp_s, fp_u = sum(fp(x) for x in ps), sum(fp(x) for x in pu)
    print(f"\nFORM-MATCHED held-out pairs (kw=2,win=3): n={len(ps)} per side")
    print(f"  FP {fp_s}/{len(ps)} vs {fp_u}/{len(pu)} | "
          f"words {np.mean([nw(x) for x in ps]):.1f}/{np.mean([nw(x) for x in pu]):.1f} | "
          f"chars {np.mean([nc(x) for x in ps]):.0f}/{np.mean([nc(x) for x in pu]):.0f}")
    print(f"FULL held-out (form-CONFOUNDED): safe FP={100*sum(fp(x) for x in held_safe)/len(held_safe):.0f}%"
          f" vs unsafe FP={100*sum(fp(x) for x in held_unsafe)/len(held_unsafe):.0f}%")

    v2 = torch.load(V2, map_location="cpu", weights_only=False)
    vnull = torch.load(VNULL, map_location="cpu", weights_only=False)["v"]

    print("\n[test1] loading LLaDA-8B-Instruct ...", flush=True)
    tok, model, blocks = common.load_model(device=device)
    print(f"[test1] projecting held-out SAFE n={len(held_safe)}", flush=True)
    Ps, Ns = collect(model, tok, blocks, held_safe, v2, vnull, device, "safe")
    print(f"[test1] projecting held-out UNSAFE n={len(held_unsafe)}", flush=True)
    Pu, Nu = collect(model, tok, blocks, held_unsafe, v2, vnull, device, "unsafe")

    json.dump({"held_safe": held_safe, "held_unsafe": held_unsafe,
               "matched_safe": ps, "matched_unsafe": pu,
               "proj_v2": {"safe": Ps, "unsafe": Pu}, "proj_null": {"safe": Ns, "unsafe": Nu}},
              open(OUT, "w"))

    # indices of the matched pairs within held_safe / held_unsafe
    si = [held_safe.index(x) for x in ps]
    ui = [held_unsafe.index(x) for x in pu]

    def block(title, rp):
        print(f"\n--- {title} | read point = {rp} ---")
        print(f"{'layer':>5} | {'v_refusal_v2':>26} | {'v_random_null':>18} | {'ratio':>7}")
        print(f"{'':>5} | {'dz':>7} {'AUC':>6} {'%u>s':>5} {'p':>5} | {'dz':>7} {'AUC':>6} | {'x null':>7}")
        for L in LAYERS:
            # PAIRED (form-matched) primary
            d_v2 = np.array([Pu[rp][L][j] for j in ui]) - np.array([Ps[rp][L][i] for i in si])
            d_nl = np.array([Nu[rp][L][j] for j in ui]) - np.array([Ns[rp][L][i] for i in si])
            s2, sn = paired(d_v2), paired(d_nl)
            a2 = auc([Ps[rp][L][i] for i in si], [Pu[rp][L][j] for j in ui])
            an = auc([Ns[rp][L][i] for i in si], [Nu[rp][L][j] for j in ui])
            r = abs(s2["dz"]) / abs(sn["dz"]) if sn["dz"] and not np.isnan(sn["dz"]) else np.inf
            print(f"L{L:>4} | {s2['dz']:>+7.2f} {a2:>6.3f} {100*s2['frac_pos']:>4.0f}% {s2['p']:>5.2f} | "
                  f"{sn['dz']:>+7.2f} {an:>6.3f} | {r:>7.1f}")

    def block_full(rp):
        print(f"\n--- FULL held-out (UNPAIRED, form-CONFOUNDED) | read point = {rp} ---")
        print(f"{'layer':>5} | {'v_refusal_v2':>21} | {'v_random_null':>21} | {'ratio':>7}")
        print(f"{'':>5} | {'cohen_d':>8} {'AUC':>6} | {'cohen_d':>8} {'AUC':>6} | {'x null':>7}")
        for L in LAYERS:
            d2 = cohen_d(Ps[rp][L], Pu[rp][L]); a2 = auc(Ps[rp][L], Pu[rp][L])
            dn = cohen_d(Ns[rp][L], Nu[rp][L]); an = auc(Ns[rp][L], Nu[rp][L])
            r = abs(d2) / abs(dn) if dn and not np.isnan(dn) else np.inf
            print(f"L{L:>4} | {d2:>+8.2f} {a2:>6.3f} | {dn:>+8.2f} {an:>6.3f} | {r:>7.1f}")

    print("\n" + "=" * 78)
    print(f"(A) FORM-CLEAN: paired per-id deltas on n={len(ps)} form-matched held-out pairs")
    print("    dz = Cohen's dz (unsafe-safe); AUC = P(unsafe>safe); ratio = |dz|/|dz_null|")
    print("=" * 78)
    block(f"form-matched pairs n={len(ps)}", "last")
    block(f"form-matched pairs n={len(ps)}", "pmean")
    print("\n" + "=" * 78)
    print("(B) FORM-CONFOUNDED CEILING: full held-out, unpaired (context only, not the test)")
    print("=" * 78)
    block_full("last")
    block_full("pmean")

    print("\nPRE-REGISTERED BAR: v_refusal_v2 is a working harm detector only if (A) clears the")
    print("v_random_null floor by clockv2's margin (|dz|/|dz_null| on the order x11-x53 at L25-L27).")
    print(f"[saved] {OUT}")
    print("NOTE: measurement axis only -- never applied to an activation. Read-only throughout.")


if __name__ == "__main__":
    main()
