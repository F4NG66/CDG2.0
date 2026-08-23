#!/usr/bin/env python3
"""PHASE 2' - gates on the DeepSeek-BOTH direction. Gate E leads. BLOCKING.

v_harm is fitted on the DeepSeek TRAIN split only; harm and safe are now the same
author, so the authorship confound that killed Phase 2 is gone by construction.

Verdicts are three-valued and judged on the 95% CI, never the point estimate:
    PASS          the CI clears the bar outright
    INCONCLUSIVE  the point clears it but the CI straddles it
    FAIL          the point itself misses

Gates
  A  separation        held-out AUC on DeepSeek TEST.                lo >= 0.90
  B  not length        len-matched AUC lo >= 0.85 AND |cos(v_harm,v_len)| < 0.30
  D  authorship        VACUOUS by construction (both sides DeepSeek). Reported as a
                       sanity check only: cos(v_harm, v_source) and the residual
                       AUC of v_harm on a DeepSeek-text vs LLaDA-text split.
  E  cross-author      BLOCKING, LEADS. Build v_harm on DeepSeek, score it on the
     transfer          28 LLaDA-authored pairs (LLaDA harm vs LLaDA safe).
                       PASS if that transfer AUC's lower CI bound > 0.50.
                       If it separates DeepSeek but is chance on LLaDA it will not
                       steer LLaDA, so E fails/inconclusive and we stop.

Gate E caveat, surfaced not hidden: the 28 LLaDA pairs are the same near-identical
templated text whose harm/safe contrast was already chance-separable in Phase 2
(word-set Jaccard 0.503; grouped-CV AUC ~0.50). So a chance Gate E can mean two
different things. To tell them apart we ALSO fit the best-possible LLaDA-only
direction on those pairs by grouped k-fold CV (the "oracle"): if even the oracle
is at chance, the transfer target carries no detectable harm contrast and Gate E
cannot be passed by ANY direction - that is an uninformative target, not a clean
refutation of v_harm. Both numbers are printed.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from analyze_gates import (FAIL, INCONC, PASS, auc, auc_ci, boot_ci, cos,
                           fmt_ci, pearson, unit, verdict_ge, verdict_lt, worst)

DATA = "/home/ore99/experement/harm_dir/data"
PROBES = "/home/ore99/experement/harm_dir/probes"


def kfold_auc(idx, ids, layout, L, k, seed):
    """Oracle: best LLaDA-only v_harm on these pairs, held out via grouped k-fold."""
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    folds = [ids[i::k] for i in range(k)]
    ph, ps = [], []
    for f in range(k):
        te = folds[f]
        tr = [c for j, fold in enumerate(folds) if j != f for c in fold]
        if len(tr) < 4 or not te:
            continue
        Hh = np.stack([idx[(c, layout, "harm")]["h"][L].numpy().astype(np.float64) for c in tr])
        Hs = np.stack([idx[(c, layout, "safe")]["h"][L].numpy().astype(np.float64) for c in tr])
        v = unit(Hh.mean(0) - Hs.mean(0))
        for c in te:
            ph.append(float(np.dot(idx[(c, layout, "harm")]["h"][L].numpy().astype(np.float64), v)))
            ps.append(float(np.dot(idx[(c, layout, "safe")]["h"][L].numpy().astype(np.float64), v)))
    return auc(ph, ps), auc_ci(np.array(ph), np.array(ps), 2000)


def evaluate(ds, ll, tr, te, nat_ids, layout, L, n_boot):
    def Hd(c, side):
        return ds[(c, layout, side)]["h"][L].numpy().astype(np.float64)

    def Td(c, side):
        return ds[(c, layout, side)]["n_resp_tokens"]

    def Hl(c, side):
        return ll[(c, layout, side)]["h"][L].numpy().astype(np.float64)

    # ---- fit v_harm on DeepSeek TRAIN ----
    Htr_h = np.stack([Hd(c, "harm") for c in tr])
    Htr_s = np.stack([Hd(c, "safe") for c in tr])
    v_harm = unit(Htr_h.mean(0) - Htr_s.mean(0))

    # length axis from both sides pooled (length, not harm in disguise)
    pool = np.concatenate([Htr_h, Htr_s])
    pt = np.array([Td(c, "harm") for c in tr] + [Td(c, "safe") for c in tr])
    med = float(np.median(pt))
    v_len = unit(pool[pt > med].mean(0) - pool[pt <= med].mean(0))

    # authorship axis: same STANCE (safe), different AUTHOR - DeepSeek vs LLaDA
    ll_safe = np.stack([Hl(c, "safe") for c in nat_ids])
    v_src = unit(Htr_s.mean(0) - ll_safe.mean(0))

    # ---- Gate A ----
    ph = np.array([float(np.dot(Hd(c, "harm"), v_harm)) for c in te])
    ps = np.array([float(np.dot(Hd(c, "safe"), v_harm)) for c in te])
    A, A_ci = auc(ph, ps), auc_ci(ph, ps, n_boot)

    # ---- Gate B ----
    r_len = pearson(np.concatenate([ph, ps]),
                    np.array([Td(c, "harm") for c in te] + [Td(c, "safe") for c in te]))
    keep = [c for c in te
            if abs(math.log(max(Td(c, "safe"), 1) / max(Td(c, "harm"), 1))) <= math.log(1.10)]
    kh = np.array([float(np.dot(Hd(c, "harm"), v_harm)) for c in keep])
    ks = np.array([float(np.dot(Hd(c, "safe"), v_harm)) for c in keep])
    A_lm, A_lm_ci = auc(kh, ks), auc_ci(kh, ks, n_boot)

    # ---- cosine CIs: refit v_harm on resampled TRAIN, cos against fixed target ----
    def cos_boot(target):
        def one(rng):
            i = rng.integers(0, len(tr), len(tr))
            return abs(cos(unit(Htr_h[i].mean(0) - Htr_s[i].mean(0)), target))
        return one

    c_len = cos(v_harm, v_len)
    c_len_ci = boot_ci(cos_boot(v_len), n_boot)
    c_src = cos(v_harm, v_src)
    c_src_ci = boot_ci(cos_boot(v_src), n_boot)

    # residual authorship AUC (Gate D sanity): does v_harm sort DeepSeek vs LLaDA text?
    auth_ds = np.array([float(np.dot(Hd(c, "safe"), v_harm)) for c in te])
    auth_ll = np.array([float(np.dot(Hl(c, "safe"), v_harm)) for c in nat_ids])
    A_auth = auc(auth_ds, auth_ll)

    # ---- Gate E: transfer to LLaDA pairs ----
    eh = np.array([float(np.dot(Hl(c, "harm"), v_harm)) for c in nat_ids])
    es = np.array([float(np.dot(Hl(c, "safe"), v_harm)) for c in nat_ids])
    E, E_ci = auc(eh, es), auc_ci(eh, es, n_boot)

    nat_te = [c for c in nat_ids if c in set(te)]      # group-clean subset
    if len(nat_te) >= 3:
        seh = np.array([float(np.dot(Hl(c, "harm"), v_harm)) for c in nat_te])
        ses = np.array([float(np.dot(Hl(c, "safe"), v_harm)) for c in nat_te])
        E_strict, E_strict_ci = auc(seh, ses), auc_ci(seh, ses, n_boot)
    else:
        E_strict, E_strict_ci = float("nan"), (float("nan"), float("nan"))

    # oracle: is there ANY harm signal in the LLaDA pairs to transfer TO?
    E_oracle, E_oracle_ci = kfold_auc(ll, nat_ids, layout, L, 5, 0)

    # ---- verdicts ----
    g_a = verdict_ge(A, A_ci[0], 0.90)
    g_b = worst(verdict_ge(A_lm, A_lm_ci[0], 0.85) if len(keep) >= 8 else INCONC,
                verdict_lt(abs(c_len), c_len_ci[1], 0.30))
    # Gate E: PASS iff transfer AUC lower CI strictly above chance.
    if math.isnan(E) or E <= 0.50:
        g_e = FAIL
    elif math.isnan(E_ci[0]) or E_ci[0] <= 0.50:
        g_e = INCONC
    else:
        g_e = PASS

    return dict(
        layout=layout, layer=L,
        auc=A, auc_ci=list(A_ci), r_len=r_len,
        auc_lenmatched=A_lm, auc_lenmatched_ci=list(A_lm_ci), n_lenmatched=len(keep),
        cos_v_length=c_len, cos_v_length_ci=list(c_len_ci),
        cos_v_source=c_src, cos_v_source_ci=list(c_src_ci),
        auc_authorship=A_auth, n_llada=len(nat_ids),
        auc_transfer=E, auc_transfer_ci=list(E_ci),
        auc_transfer_strict=E_strict, auc_transfer_strict_ci=list(E_strict_ci),
        n_transfer_strict=len(nat_te),
        auc_oracle=E_oracle, auc_oracle_ci=list(E_oracle_ci),
        gate_a=g_a, gate_b=g_b, gate_e=g_e,
        verdict=worst(g_a, g_b, g_e),
        v_harm=v_harm)


def print_table(rows):
    print()
    print("=" * 140)
    print("PHASE 2' VERDICT TABLE  -  v_harm built on DeepSeek TRAIN (both sides same author)")
    print("  Gate E LEADS: transfer of the DeepSeek direction to LLaDA-authored pairs")
    print("=" * 140)
    print(f"{'layout':7s} {'L':>3s} {'AUC':>6s} {'95%CI':>12s} "
          f"{'AUC_lenm':>8s} {'95%CI':>12s} {'cos_len':>7s} {'cos_src':>7s} "
          f"{'AUCxfer':>7s} {'95%CI':>12s} {'A':>5s} {'B':>5s} {'E':>5s} {'OVERALL':>7s}")
    print("-" * 140)
    for r in rows:
        print(f"{r['layout']:7s} {r['layer']:3d} {r['auc']:6.3f} {fmt_ci(r['auc_ci']):>12s} "
              f"{r['auc_lenmatched']:8.3f} {fmt_ci(r['auc_lenmatched_ci']):>12s} "
              f"{r['cos_v_length']:7.3f} {r['cos_v_source']:7.3f} "
              f"{r['auc_transfer']:7.3f} {fmt_ci(r['auc_transfer_ci']):>12s} "
              f"{r['gate_a']:>5s} {r['gate_b']:>5s} {r['gate_e']:>5s} {r['verdict']:>7s}")
    print("-" * 140)
    print("  Gate E diagnostics (group-clean transfer subset  vs  oracle = best LLaDA-only "
          "direction on the same pairs):")
    for r in rows:
        print(f"    {r['layout']:4s} L{r['layer']:<2d}  "
              f"xfer_full={r['auc_transfer']:.3f}{fmt_ci(r['auc_transfer_ci'])} (n={r['n_llada']})  "
              f"xfer_clean={r['auc_transfer_strict']:.3f}{fmt_ci(r['auc_transfer_strict_ci'])} "
              f"(n={r['n_transfer_strict']})  "
              f"oracle={r['auc_oracle']:.3f}{fmt_ci(r['auc_oracle_ci'])}")
    print("  residual authorship AUC (v_harm sorting DeepSeek-text vs LLaDA-text; ~0.5 = "
          "v_harm is NOT an authorship axis):")
    print("    " + "  ".join(
        f"{r['layout'][:4]}L{r['layer']}={r['auc_authorship']:.2f}" for r in rows))
    print("  B1 r(projection,len): " + "  ".join(
        f"{r['layout'][:4]}L{r['layer']}={r['r_len']:+.2f}" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-ds", default=os.path.join(DATA, "states_ds.pt"))
    ap.add_argument("--states-llada", default=os.path.join(DATA, "states.pt"))
    ap.add_argument("--split", default=os.path.join(DATA, "split_ds.json"))
    ap.add_argument("--out", default=os.path.join(DATA, "gates_ds.json"))
    ap.add_argument("--probes-out", default=os.path.join(PROBES, "v_harm_ds.pt"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    bd = torch.load(args.states_ds, map_location="cpu", weights_only=False)
    bl = torch.load(args.states_llada, map_location="cpu", weights_only=False)
    ds = {(r["case_id"], r["layout"], r["side"]): r for r in bd["records"]}
    ll = {(r["case_id"], r["layout"], r["side"]): r for r in bl["records"]}
    layers, layouts = bd["layers"], bd["layouts"]
    assert layers == bl["layers"] and layouts == bl["layouts"], \
        "DeepSeek and LLaDA captures must share layers/layouts for transfer to be valid"

    split = json.load(open(args.split))
    ds_cases = sorted({r["case_id"] for r in bd["records"]})
    tr = [c for c in ds_cases if c in set(split["train"])]
    te = [c for c in ds_cases if c in set(split["test"])]
    nat_ids = sorted({r["case_id"] for r in bl["records"]
                      if r.get("safe_source") == "llada_prefill_arm"})
    if len(tr) < 5 or len(te) < 3 or len(nat_ids) < 8:
        raise SystemExit(f"need full captures: train={len(tr)} test={len(te)} "
                         f"llada_pairs={len(nat_ids)} - this looks like a smoke file")

    print(f"DeepSeek states: {len(bd['records'])} rows  cases={len(ds_cases)}  "
          f"train={len(tr)} test={len(te)}")
    print(f"LLaDA transfer pairs (both sides LLaDA): {len(nat_ids)}   "
          f"layers={layers} layouts={layouts} boot={args.boot}")

    rows = [evaluate(ds, ll, tr, te, nat_ids, lo, L, args.boot)
            for lo in layouts for L in layers]
    probes = {}
    for r in rows:
        probes[f"{r['layout']}/L{r['layer']}"] = torch.tensor(r.pop("v_harm"),
                                                              dtype=torch.float32)
    print_table(rows)

    print()
    print("thresholds  A: AUC lo>=0.90 | B: AUC_lenm lo>=0.85 AND |cos_len|<0.30 | "
          "E: transfer AUC lo>0.50 (all judged on the CI)")
    print("PASS = CI clears bar | INCONC = point clears, CI straddles | FAIL = point misses")

    # ---- overall, led by Gate E ----
    passed = [r for r in rows if r["verdict"] == PASS]
    inconc = [r for r in rows if r["verdict"] == INCONC]
    e_pass = [r for r in rows if r["gate_e"] == PASS]
    oracle_best = max(rows, key=lambda r: -1 if math.isnan(r["auc_oracle"]) else r["auc_oracle"])
    oracle_dead = (math.isnan(oracle_best["auc_oracle"])
                   or oracle_best["auc_oracle_ci"][0] <= 0.50)

    print()
    print("=" * 140)
    if passed:
        b = max(passed, key=lambda r: r["auc_transfer"])
        print(f"OVERALL: PASS - {len(passed)}/{len(rows)} combos clear A, B and E on the CI.")
        print(f"  best: {b['layout']} L{b['layer']}  AUC={b['auc']:.3f}{fmt_ci(b['auc_ci'])}  "
              f"transfer={b['auc_transfer']:.3f}{fmt_ci(b['auc_transfer_ci'])}  "
              f"cos_len={b['cos_v_length']:.2f}  cos_src={b['cos_v_source']:.2f}")
        print("  -> the DeepSeek direction separates AND transfers to LLaDA. Phase 3 unblocked.")
    elif not e_pass and oracle_dead:
        print("OVERALL: INCONCLUSIVE - Gate E is uninformative on this target.")
        print(f"  The transfer target (the {rows[0]['n_llada']} LLaDA pairs) carries no "
              f"detectable harm contrast:")
        print(f"    oracle (best LLaDA-only direction, grouped CV) tops out at "
              f"{oracle_best['auc_oracle']:.3f}"
              f"{fmt_ci(oracle_best['auc_oracle_ci'])} - chance.")
        print("  No direction can pass Gate E on these pairs, so E neither confirms nor")
        print("  refutes v_harm. Report Gate A/B (below) but do NOT steer on E's basis.")
        bA = max(rows, key=lambda r: r["auc"])
        print(f"  DeepSeek-side separation: best {bA['layout']} L{bA['layer']} "
              f"AUC={bA['auc']:.3f}{fmt_ci(bA['auc_ci'])}  (A={bA['gate_a']})")
    elif inconc:
        b = max(inconc, key=lambda r: r["auc_transfer"])
        print(f"OVERALL: INCONCLUSIVE - {len(inconc)}/{len(rows)} combos clear the bars on "
              f"point estimates but CIs straddle.")
        print(f"  best: {b['layout']} L{b['layer']}  AUC={b['auc']:.3f}{fmt_ci(b['auc_ci'])}  "
              f"transfer={b['auc_transfer']:.3f}{fmt_ci(b['auc_transfer_ci'])}")
        print("  -> Do NOT steer yet.")
    else:
        b = max(rows, key=lambda r: r["auc_transfer"])
        print("OVERALL: FAIL - no combo clears A, B and E.")
        print(f"  Gate E best transfer: {b['layout']} L{b['layer']} "
              f"{b['auc_transfer']:.3f}{fmt_ci(b['auc_transfer_ci'])}  "
              f"(oracle on same pairs {oracle_best['auc_oracle']:.3f})")
        print("  -> Do NOT proceed to Phase 3.")
    print("=" * 140)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"n_boot": args.boot, "n_llada_transfer": len(nat_ids),
               "train": tr, "test": te, "results": rows},
              open(args.out, "w"), indent=2, default=float)
    os.makedirs(os.path.dirname(args.probes_out), exist_ok=True)
    torch.save({"kind": "harm_vs_safe_direction_deepseek_both",
                "method": "unit(mean(harm)-mean(safe)) over DeepSeek TRAIN response states",
                "pooling": bd["pooling"], "layers": layers, "layouts": layouts,
                "v": probes}, args.probes_out)
    print(f"\nwrote {args.out}\nwrote {args.probes_out}")


if __name__ == "__main__":
    main()
