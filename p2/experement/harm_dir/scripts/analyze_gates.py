#!/usr/bin/env python3
"""PHASE 2 - the cheap gate. BLOCKING: if this fails we do not steer.

v_harm is fitted on TRAIN only; every number below is held-out TEST.

Verdicts are THREE-VALUED and judged against confidence intervals, not point
estimates. For a gate with threshold t on a metric with 95% CI [lo, hi]:

    PASS          the CI clears the bar outright
    INCONCLUSIVE  the point estimate clears it but the CI straddles it
    FAIL          the point estimate itself misses

A gate that cannot come back INCONCLUSIVE is a gate that launders small-n noise
into a green light.

  Gate A  separation      ROC-AUC of <h, v_harm>.        t = 0.90
  Gate B  length control  B2 AUC on a length-matched subset,  t = 0.85
                          B3 |cos(v_harm, v_length)|,        t < 0.30
                          (B1 Pearson r(projection, length) reported)
  Gate C  bonus           cos(v_harm, v_injection_svd). Reported only.
  Gate D  authorship      BLOCKING. AUC on pairs where BOTH sides are LLaDA,
                          t = 0.85, and additionally its lower CI bound must sit
                          strictly above chance (> 0.50) - a direction that only
                          works because the safe half was written by a different
                          model is useless for steering, since generation is
                          always LLaDA. Plus |cos(v_harm, v_source)| < 0.30.

Split of record for Gate D is the natural-heavy split (20 natural pairs held
out). The stratified split is reported alongside it so its underpowering (n=8)
stays visible rather than hidden.

All cosine gates get bootstrap CIs too, by refitting v_harm on resampled TRAIN.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

DATA = "/home/ore99/experement/harm_dir/data"
PROBES = "/home/ore99/experement/harm_dir/probes"
V_INJ = "/scratch/ore99/clockv2/probes/v_injection_svd.pt"

PASS, INCONC, FAIL = "PASS", "INCONC", "FAIL"
RANK = {FAIL: 0, INCONC: 1, PASS: 2}


# ----------------------------------------------------------------- statistics
def auc(pos, neg):
    """ROC-AUC via the Mann-Whitney U statistic (ties at 0.5)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def boot_ci(one, n_boot, seed=0):
    vals = [v for v in (one(np.random.default_rng(seed + i)) for i in range(n_boot))
            if v is not None and not math.isnan(v)]
    if len(vals) < max(8, n_boot // 4):
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def auc_ci(pos, neg, n_boot, seed=0):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return (float("nan"), float("nan"))

    def one(rng):
        return auc(pos[rng.integers(0, len(pos), len(pos))],
                   neg[rng.integers(0, len(neg), len(neg))])
    return boot_ci(one, n_boot, seed)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def verdict_ge(point, lo, thresh):
    """Metric must be >= thresh, judged on the CI."""
    if math.isnan(point) or point < thresh:
        return FAIL
    return INCONC if (math.isnan(lo) or lo < thresh) else PASS


def verdict_lt(point, hi, thresh):
    """Absolute cosine must be < thresh, judged on the CI."""
    if math.isnan(point) or point >= thresh:
        return FAIL
    return INCONC if (math.isnan(hi) or hi >= thresh) else PASS


def worst(*vs):
    return min(vs, key=lambda v: RANK[v])


# ----------------------------------------------------------------- evaluation
def evaluate(idx, tr, te, layout, L, v_inj, n_boot):
    def H(c, side):
        return idx[(c, layout, side)]["h"][L].numpy().astype(np.float64)

    def T(c, side):
        return idx[(c, layout, side)]["n_resp_tokens"]

    def src(c):
        return idx[(c, layout, "safe")]["safe_source"]

    Htr_h = np.stack([H(c, "harm") for c in tr])
    Htr_s = np.stack([H(c, "safe") for c in tr])
    v_harm = unit(Htr_h.mean(0) - Htr_s.mean(0))

    # length axis, from BOTH sides pooled so it is length and not harm in disguise
    pool_h = np.concatenate([Htr_h, Htr_s])
    pool_t = np.array([T(c, "harm") for c in tr] + [T(c, "safe") for c in tr])
    med = float(np.median(pool_t))
    v_len = unit(pool_h[pool_t > med].mean(0) - pool_h[pool_t <= med].mean(0))

    # authorship axis, safe side only
    nat_i = [i for i, c in enumerate(tr) if src(c) == "llada_prefill_arm"]
    gen_i = [i for i, c in enumerate(tr) if src(c) == "deepseek_generated"]
    have_src = len(nat_i) >= 3 and len(gen_i) >= 3
    v_src = unit(Htr_s[nat_i].mean(0) - Htr_s[gen_i].mean(0)) if have_src else None

    # ---- Gate A ----
    ph = np.array([float(np.dot(H(c, "harm"), v_harm)) for c in te])
    ps = np.array([float(np.dot(H(c, "safe"), v_harm)) for c in te])
    A = auc(ph, ps)
    A_ci = auc_ci(ph, ps, n_boot)

    # ---- Gate B ----
    r_len = pearson(np.concatenate([ph, ps]),
                    np.array([T(c, "harm") for c in te] + [T(c, "safe") for c in te]))
    keep = [c for c in te
            if abs(math.log(max(T(c, "safe"), 1) / max(T(c, "harm"), 1))) <= math.log(1.10)]
    kh = np.array([float(np.dot(H(c, "harm"), v_harm)) for c in keep])
    ks = np.array([float(np.dot(H(c, "safe"), v_harm)) for c in keep])
    A_lm = auc(kh, ks)
    A_lm_ci = auc_ci(kh, ks, n_boot)

    # ---- Gate D ----
    nat_te = [c for c in te if src(c) == "llada_prefill_arm"]
    gen_te = [c for c in te if src(c) == "deepseek_generated"]
    nh = np.array([float(np.dot(H(c, "harm"), v_harm)) for c in nat_te])
    ns = np.array([float(np.dot(H(c, "safe"), v_harm)) for c in nat_te])
    A_nat = auc(nh, ns)
    A_nat_ci = auc_ci(nh, ns, n_boot)
    A_gen = auc(np.array([float(np.dot(H(c, "harm"), v_harm)) for c in gen_te]),
                np.array([float(np.dot(H(c, "safe"), v_harm)) for c in gen_te]))

    # ---- cosine CIs: refit v_harm on resampled TRAIN ----
    def cos_boot(target):
        def one(rng):
            i = rng.integers(0, len(tr), len(tr))
            return abs(cos(unit(Htr_h[i].mean(0) - Htr_s[i].mean(0)), target))
        return one

    c_len = cos(v_harm, v_len)
    c_len_ci = boot_ci(cos_boot(v_len), n_boot)
    c_src = cos(v_harm, v_src) if have_src else float("nan")
    c_src_ci = boot_ci(cos_boot(v_src), n_boot) if have_src else (float("nan"),) * 2

    c_inj = float("nan")
    if v_inj is not None and L in v_inj.get("v", {}).get("mean", {}):
        c_inj = cos(v_harm, v_inj["v"]["mean"][L].numpy().astype(np.float64))

    # ---- verdicts ----
    g_a = verdict_ge(A, A_ci[0], 0.90)
    g_b = worst(verdict_ge(A_lm, A_lm_ci[0], 0.85) if len(keep) >= 8 else INCONC,
                verdict_lt(abs(c_len), c_len_ci[1], 0.30))
    # Gate D additionally demands the natural-only AUC sit strictly above chance.
    d_chance = (PASS if (not math.isnan(A_nat_ci[0]) and A_nat_ci[0] > 0.50)
                else FAIL if (math.isnan(A_nat) or A_nat <= 0.50) else INCONC)
    g_d = worst(verdict_ge(A_nat, A_nat_ci[0], 0.85) if len(nat_te) >= 8 else INCONC,
                d_chance,
                verdict_lt(abs(c_src), c_src_ci[1], 0.30))

    return dict(
        layout=layout, layer=L,
        auc=A, auc_ci=list(A_ci), r_len=r_len,
        auc_lenmatched=A_lm, auc_lenmatched_ci=list(A_lm_ci), n_lenmatched=len(keep),
        cos_v_length=c_len, cos_v_length_ci=list(c_len_ci),
        auc_natural=A_nat, auc_natural_ci=list(A_nat_ci), n_natural=len(nat_te),
        auc_generated=A_gen, n_generated=len(gen_te),
        cos_v_source=c_src, cos_v_source_ci=list(c_src_ci),
        cos_v_injection_svd=c_inj,
        gate_a=g_a, gate_b=g_b, gate_d=g_d,
        verdict=worst(g_a, g_b, g_d),
        v_harm=v_harm)


# ----------------------------------------------------------------- reporting
def fmt_ci(ci):
    return "      [-]     " if math.isnan(ci[0]) else f"[{ci[0]:.2f},{ci[1]:.2f}]"


def print_table(name, rows, of_record):
    tag = "   <-- SPLIT OF RECORD (Gate D decision)" if of_record else ""
    print()
    print("=" * 132)
    print(f"SPLIT: {name}{tag}")
    print("=" * 132)
    print(f"{'layout':7s} {'L':>3s} {'AUC':>6s} {'95% CI':>13s} "
          f"{'AUC_lenm':>9s} {'95% CI':>13s} {'cos_len':>8s} "
          f"{'AUC_nat':>8s} {'95% CI':>13s} {'n':>3s} {'cos_src':>8s} "
          f"{'cos_inj':>8s} {'A':>6s} {'B':>6s} {'D':>6s} {'VERDICT':>7s}")
    print("-" * 132)
    for r in rows:
        print(f"{r['layout']:7s} {r['layer']:3d} {r['auc']:6.3f} {fmt_ci(r['auc_ci']):>13s} "
              f"{r['auc_lenmatched']:9.3f} {fmt_ci(r['auc_lenmatched_ci']):>13s} "
              f"{r['cos_v_length']:8.3f} "
              f"{r['auc_natural']:8.3f} {fmt_ci(r['auc_natural_ci']):>13s} {r['n_natural']:3d} "
              f"{r['cos_v_source']:8.3f} {r['cos_v_injection_svd']:8.3f} "
              f"{r['gate_a']:>6s} {r['gate_b']:>6s} {r['gate_d']:>6s} {r['verdict']:>7s}")
    print("-" * 132)
    print("  B1 r(projection,len): " + "  ".join(
        f"{r['layout'][:4]}L{r['layer']}={r['r_len']:+.2f}" for r in rows))
    print("  AUC_gen (DeepSeek-safe pairs, context only): " + "  ".join(
        f"{r['layout'][:4]}L{r['layer']}={r['auc_generated']:.2f}" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join(DATA, "states.pt"))
    ap.add_argument("--splits", default=f"natural_heavy:{DATA}/split_natheavy.json,"
                                        f"stratified:{DATA}/split.json")
    ap.add_argument("--of-record", default="natural_heavy")
    ap.add_argument("--out", default=os.path.join(DATA, "gates.json"))
    ap.add_argument("--probes-out", default=os.path.join(PROBES, "v_harm.pt"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    blob = torch.load(args.states, map_location="cpu", weights_only=False)
    recs = blob["records"]
    layers, layouts = blob["layers"], blob["layouts"]
    idx = {(r["case_id"], r["layout"], r["side"]): r for r in recs}
    cases = sorted({r["case_id"] for r in recs})

    v_inj = (torch.load(V_INJ, map_location="cpu", weights_only=False)
             if os.path.exists(V_INJ) else None)
    if v_inj is None:
        print(f"NOTE: {V_INJ} missing - Gate C column will read nan")

    print(f"states: {len(recs)} records  cases={len(cases)}  "
          f"layouts={layouts}  layers={layers}  bootstrap={args.boot}")

    all_results, probes = {}, {}
    for spec in args.splits.split(","):
        name, path = spec.split(":", 1)
        split = json.load(open(path))
        tr = [c for c in cases if c in set(split["train"])]
        te = [c for c in cases if c in set(split["test"])]
        if not te or not tr:
            raise SystemExit(f"split {name}: train={len(tr)} test={len(te)} - the gates "
                             "need the full capture, not the smoke file")
        rows = [evaluate(idx, tr, te, lo, L, v_inj, args.boot)
                for lo in layouts for L in layers]
        print(f"  split {name:14s} train={len(tr):3d} test={len(te):3d} "
              f"natural_in_test={rows[0]['n_natural']:3d}")
        for r in rows:
            probes[f"{name}/{r['layout']}/L{r['layer']}"] = torch.tensor(
                r.pop("v_harm"), dtype=torch.float32)
        all_results[name] = rows

    for name, rows in all_results.items():
        print_table(name, rows, name == args.of_record)

    print()
    print("thresholds  A: AUC>=0.90 | B: AUC_lenm>=0.85 AND |cos_len|<0.30 | "
          "D: AUC_nat>=0.85 AND lowerCI>0.50 AND |cos_src|<0.30")
    print("all three BLOCKING; judged on the CI, not the point estimate")
    print("PASS = CI clears the bar | INCONC = point clears it but CI straddles | "
          "FAIL = point misses")

    rows = all_results[args.of_record]
    passed = [r for r in rows if r["verdict"] == PASS]
    inconc = [r for r in rows if r["verdict"] == INCONC]
    print()
    print("=" * 132)
    if passed:
        b = max(passed, key=lambda r: r["auc_natural"])
        print(f"OVERALL ({args.of_record}): PASS - {len(passed)}/{len(rows)} combos clear "
              f"all three gates on the CI.")
        print(f"  best: {b['layout']} L{b['layer']}  AUC={b['auc']:.3f} {fmt_ci(b['auc_ci'])}  "
              f"AUC_nat={b['auc_natural']:.3f} {fmt_ci(b['auc_natural_ci'])}  "
              f"cos_len={b['cos_v_length']:.3f}  cos_src={b['cos_v_source']:.3f}")
        print("  -> Phase 3 unblocked.")
    elif inconc:
        b = max(inconc, key=lambda r: r["auc_natural"])
        print(f"OVERALL ({args.of_record}): INCONCLUSIVE - {len(inconc)}/{len(rows)} combos "
              f"clear the bars on point estimates but their CIs straddle.")
        print(f"  best: {b['layout']} L{b['layer']}  AUC={b['auc']:.3f} {fmt_ci(b['auc_ci'])}  "
              f"AUC_nat={b['auc_natural']:.3f} {fmt_ci(b['auc_natural_ci'])}")
        print("  -> Do NOT steer yet. More natural LLaDA-safe pairs would tighten this.")
    else:
        b = max(rows, key=lambda r: -1 if math.isnan(r["auc_natural"]) else r["auc_natural"])
        why = [g for g, k in (("A", "gate_a"), ("B", "gate_b"), ("D", "gate_d"))
               if all(r[k] == FAIL for r in rows)]
        print(f"OVERALL ({args.of_record}): FAIL - no combo passes.")
        if why:
            print(f"  every row fails gate(s): {', '.join(why)}")
        print(f"  best attempt: {b['layout']} L{b['layer']}  AUC={b['auc']:.3f}  "
              f"AUC_nat={b['auc_natural']:.3f}  cos_len={b['cos_v_length']:.3f}  "
              f"cos_src={b['cos_v_source']:.3f}")
        print("  -> Do NOT proceed to Phase 3.")
    print("=" * 132)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"of_record": args.of_record, "n_boot": args.boot,
               "results": all_results}, open(args.out, "w"), indent=2)
    os.makedirs(os.path.dirname(args.probes_out), exist_ok=True)
    torch.save({"kind": "harm_vs_safe_direction",
                "method": "unit(mean(harm resp states) - mean(safe resp states)), TRAIN only",
                "pooling": blob["pooling"], "layers": layers, "layouts": layouts,
                "v": probes}, args.probes_out)
    print(f"\nwrote {args.out}\nwrote {args.probes_out}")


if __name__ == "__main__":
    main()
