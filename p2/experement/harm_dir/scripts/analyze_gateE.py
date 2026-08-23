#!/usr/bin/env python3
"""PHASE 2'' - the VALID Gate E: does v_harm_ds transfer to real LLaDA harm/safe?

v_harm is rebuilt exactly as in Phase 2' (DeepSeek TRAIN, per layout/layer). It is
then scored on LLaDA-authored harm/safe pairs where BOTH arms are LLaDA and the
harm/safe contrast is real:

  matched  harm = LLaDA DIJA comply, safe = LLaDA-generated matched-length refusal
           -> real content contrast AND length controlled. THE DECISION TARGET.
  natural  harm = LLaDA DIJA comply, safe = LLaDA bare-behaviour refusal (~8 words)
           -> real content contrast but 12.5x length confound. COMPARISON ONLY.

For each target we report, per layout/layer:
  transfer  AUC of <h_LLaDA, v_harm> (harm vs safe) + 95% CI      DECISION: lo > 0.50
  oracle    best LLaDA-only direction on the SAME pairs (grouped CV) + CI
            -> confirms the target actually carries separable signal
  length diagnostics so a positive transfer can't be an unexamined length effect:
    cos(v_harm, v_len_target)   v_harm is length-orthogonal by construction; this
                                checks it stays so in the LLaDA target space
    AUC(v_len_target)           length-only baseline on this target
    r(proj_vharm, len)          correlation of the v_harm score with length

Gate E PASS iff transfer-AUC lower CI > 0.50 on the MATCHED target. A pass is only
credible if the length diagnostics are small (transfer is harm, not length).
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from analyze_gates import (FAIL, INCONC, PASS, auc, auc_ci, cos, fmt_ci,
                           pearson, unit)
from analyze_gates_ds import kfold_auc

DATA = "/home/ore99/experement/harm_dir/data"


def build_vharm(ds, tr, layout, L):
    Hh = np.stack([ds[(c, layout, "harm")]["h"][L].numpy().astype(np.float64) for c in tr])
    Hs = np.stack([ds[(c, layout, "safe")]["h"][L].numpy().astype(np.float64) for c in tr])
    return unit(Hh.mean(0) - Hs.mean(0))


def eval_target(tgt, ids, layout, L, v_harm, n_boot, seed=0):
    def H(c, side):
        return tgt[(c, layout, side)]["h"][L].numpy().astype(np.float64)

    def T(c, side):
        return tgt[(c, layout, side)]["n_resp_tokens"]

    ph = np.array([float(np.dot(H(c, "harm"), v_harm)) for c in ids])
    ps = np.array([float(np.dot(H(c, "safe"), v_harm)) for c in ids])
    A, A_ci = auc(ph, ps), auc_ci(ph, ps, n_boot)

    Hh = np.stack([H(c, "harm") for c in ids])
    Hs = np.stack([H(c, "safe") for c in ids])

    # ---- PRIMARY metric: directional alignment ----
    # When the harm/safe activation CLUSTERS sit far apart, AUC saturates - almost
    # any direction (a RANDOM one, or the norm axis) separates them, so a transfer
    # AUC of 1.0 says little. The discriminating question is whether v_harm points
    # the SAME WAY as LLaDA's own harm axis. Measure cos(v_harm, v_llada_oracle)
    # against a random-direction null; cos ~ N(0, 1/sqrt(d)) so |cos|>~0.05 is >3 sigma.
    v_llada = unit(Hh.mean(0) - Hs.mean(0))          # LLaDA's own harm/safe axis
    cos_align = float(np.dot(unit(v_harm), v_llada))
    rng = np.random.default_rng(seed)
    d = v_harm.shape[0]
    null_cos = np.array([abs(float(np.dot(unit(rng.standard_normal(d)), v_llada)))
                         for _ in range(2000)])
    null_p95, null_p99 = float(np.percentile(null_cos, 95)), float(np.percentile(null_cos, 99))
    # saturation witnesses: how easily does a *random* direction separate the clusters,
    # and does raw norm separate them? (ONE shared random direction per draw - project
    # both classes onto it, else the AUC is meaningless per-element noise.)
    def rand_auc_once():
        rv = unit(rng.standard_normal(d))
        return auc([float(np.dot(H(c, "harm"), rv)) for c in ids],
                   [float(np.dot(H(c, "safe"), rv)) for c in ids])
    null_auc = np.array([rand_auc_once() for _ in range(500)])
    rand_auc_p95 = float(np.percentile(null_auc, 95))
    auc_norm = auc([float(np.linalg.norm(H(c, "harm"))) for c in ids],
                   [float(np.linalg.norm(H(c, "safe"))) for c in ids])

    # length diagnostics
    pool = np.concatenate([Hh, Hs])
    tks = np.array([T(c, "harm") for c in ids] + [T(c, "safe") for c in ids])
    med = float(np.median(tks))
    if (tks > med).sum() >= 2 and (tks <= med).sum() >= 2:
        v_len = unit(pool[tks > med].mean(0) - pool[tks <= med].mean(0))
        c_vl = abs(cos(v_harm, v_len))
    else:
        c_vl = float("nan")
    r_len = pearson(np.concatenate([ph, ps]), tks)

    orc, orc_ci = kfold_auc(tgt, ids, layout, L, 5, 0)
    return dict(auc=A, auc_ci=list(A_ci), oracle=orc, oracle_ci=list(orc_ci),
                cos_align=cos_align, null_p95=null_p95, null_p99=null_p99,
                rand_auc_p95=rand_auc_p95, auc_norm=auc_norm,
                cos_vlen=c_vl, r_len=r_len, n=len(ids))


def verdict_align(cos_align, null_p99):
    """Gate E on alignment: v_harm must point along LLaDA's harm axis well above the
    random-direction null. 0.20 is a practical floor for 'meaningful shared axis'."""
    if math.isnan(cos_align) or cos_align <= null_p99:
        return FAIL
    return PASS if cos_align >= 0.20 else INCONC


def run(name, tgt, ids, ds, tr, layouts, layers, n_boot):
    print()
    print("=" * 132)
    print(f"GATE E TARGET: {name}   ({len(ids)} LLaDA pairs, both arms LLaDA)")
    print("  PRIMARY = cos_align: cos(v_harm_DeepSeek, v_harm_LLaDA_oracle) vs random null.")
    print("  transferAUC is SATURATED here (see rand_AUC_p95 / AUC_norm) so it is diagnostic only.")
    print("=" * 132)
    print(f"{'layout':7s} {'L':>3s} {'cos_align':>9s} {'null_p99':>8s} {'E':>5s}  "
          f"{'xferAUC':>7s} {'randAUC95':>9s} {'AUC_norm':>8s} {'cos_vlen':>8s} {'r_len':>6s} "
          f"{'oracle':>7s}")
    print("-" * 132)
    rows = []
    for lo in layouts:
        for L in layers:
            v = build_vharm(ds, tr, lo, L)
            r = eval_target(tgt, ids, lo, L, v, n_boot)
            r["layout"], r["layer"] = lo, L
            r["gate_e"] = verdict_align(r["cos_align"], r["null_p99"])
            rows.append(r)
            print(f"{lo:7s} {L:3d} {r['cos_align']:9.3f} {r['null_p99']:8.3f} {r['gate_e']:>5s}  "
                  f"{r['auc']:7.3f} {r['rand_auc_p95']:9.3f} {r['auc_norm']:8.3f} "
                  f"{r['cos_vlen']:8.3f} {r['r_len']:6.2f} {r['oracle']:7.3f}")
    print("-" * 132)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-ds", default=os.path.join(DATA, "states_ds.pt"))
    ap.add_argument("--split", default=os.path.join(DATA, "split_ds.json"))
    ap.add_argument("--states-matched", default=os.path.join(DATA, "states_gateE.pt"))
    ap.add_argument("--states-natural", default=os.path.join(DATA, "states_gateE_natural.pt"))
    ap.add_argument("--out", default=os.path.join(DATA, "gateE.json"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    bd = torch.load(args.states_ds, map_location="cpu", weights_only=False)
    ds = {(r["case_id"], r["layout"], r["side"]): r for r in bd["records"]}
    layers, layouts = bd["layers"], bd["layouts"]
    split = json.load(open(args.split))
    ds_cases = sorted({r["case_id"] for r in bd["records"]})
    tr = [c for c in ds_cases if c in set(split["train"])]

    targets = {}
    for label, path in (("matched", args.states_matched), ("natural", args.states_natural)):
        if os.path.exists(path):
            b = torch.load(path, map_location="cpu", weights_only=False)
            assert b["layers"] == layers and b["layouts"] == layouts, \
                f"{label} capture must share layers/layouts with DeepSeek states"
            tgt = {(r["case_id"], r["layout"], r["side"]): r for r in b["records"]}
            ids = sorted({r["case_id"] for r in b["records"]})
            targets[label] = (tgt, ids)

    if "matched" not in targets:
        raise SystemExit(f"missing {args.states_matched} - run the Gate E capture first")

    print(f"v_harm built on DeepSeek TRAIN ({len(tr)} cases). boot={args.boot}")
    all_rows = {}
    for label in ("matched", "natural"):
        if label in targets:
            tgt, ids = targets[label]
            all_rows[label] = run(label, tgt, ids, ds, tr, layouts, layers, args.boot)

    # ---- verdict on the MATCHED target (decided on ALIGNMENT, not saturated AUC) ----
    m = all_rows["matched"]
    passed = [r for r in m if r["gate_e"] == PASS]
    inconc = [r for r in m if r["gate_e"] == INCONC]
    best = max(m, key=lambda r: r["cos_align"])
    saturated = any(r["rand_auc_p95"] > 0.85 for r in m)

    print()
    print("thresholds  E (alignment): cos(v_harm, v_llada_oracle) > null_p99 AND >= 0.20")
    print("=" * 132)
    if saturated:
        print(f"NOTE: transfer-AUC is SATURATED (a random direction reaches AUC "
              f"{max(r['rand_auc_p95'] for r in m):.2f}); the harm/safe clusters sit far "
              "apart so AUC is uninformative. Gate E is decided on directional alignment.")
    if passed:
        b = best
        clean = b["cos_vlen"] < 0.35 and abs(b["r_len"]) < 0.35
        print(f"GATE E: PASS - {len(passed)}/{len(m)} combos align with LLaDA's own harm axis "
              f"far above the random null.")
        print(f"  best: {b['layout']} L{b['layer']}  cos_align={b['cos_align']:.3f} "
              f"(null_p99={b['null_p99']:.3f}; ~{b['cos_align']/max(b['null_p99'],1e-6):.0f}x)  "
              f"cos_vlen={b['cos_vlen']:.2f}  r_len={b['r_len']:.2f}  "
              f"{'(not length)' if clean else '(length not fully controlled)'}")
        print("  -> the DeepSeek harm direction genuinely points along LLaDA's harm axis "
              "(shared ~cos), not merely 'everything separates'.")
        print("  Necessary condition for steering met. NOTE: alignment ~0.5 is a partial (not "
              "perfect) proxy, and this validates a DIRECTION, not yet a causal steering effect.")
    elif inconc:
        b = best
        print(f"GATE E: INCONCLUSIVE - alignment above null but below the 0.20 floor.")
        print(f"  best: {b['layout']} L{b['layer']}  cos_align={b['cos_align']:.3f} "
              f"(null_p99={b['null_p99']:.3f})")
        print("  -> weak shared axis; steering would be unreliable.")
    else:
        print(f"GATE E: FAIL - v_harm does NOT align with LLaDA's harm axis "
              f"(best cos_align={best['cos_align']:.3f} <= null).")
        print("  -> the DeepSeek harm axis is not LLaDA's harm axis. Do not steer with it.")
    print("=" * 132)

    json.dump({"n_boot": args.boot, "train": tr, "results": all_rows},
              open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
