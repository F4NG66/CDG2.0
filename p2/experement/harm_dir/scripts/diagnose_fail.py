#!/usr/bin/env python3
"""PHASE 2 follow-up - why did the gate fail, and is the premise salvageable?

The verdict table says v_harm is largely an authorship axis. This asks the one
question that decides what to do next:

    Is there ANY harm signal in the 28 natural pairs (both sides LLaDA)?

If yes, the fix is more natural LLaDA-safe pairs and the approach stands.
If no, the harm-vs-safe premise itself is wrong for this data.

Uses grouped k-fold CV over the natural pairs only, so all 28 get a held-out
score instead of burning 20 of them on a single fit. No GPU: reuses states.pt.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from analyze_gates import auc, auc_ci, boot_ci, cos, unit

DATA = "/home/ore99/experement/harm_dir/data"


def kfold_auc(idx, ids, layout, L, k, seed, tag):
    """Grouped k-fold: v_harm fitted on the other folds, scored on the held-out."""
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
    return auc(ph, ps), auc_ci(np.array(ph), np.array(ps), 2000), len(ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join(DATA, "states.pt"))
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs_complete.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "diagnosis.json"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(args.states, map_location="cpu", weights_only=False)
    idx = {(r["case_id"], r["layout"], r["side"]): r for r in blob["records"]}
    layers, layouts = blob["layers"], blob["layouts"]
    pairs = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.pairs)}

    nat = sorted(c for c, p in pairs.items() if p["safe_source"] == "llada_prefill_arm")
    gen = sorted(c for c, p in pairs.items() if p["safe_source"] == "deepseek_generated")
    print(f"natural (LLaDA safe) pairs: {len(nat)}   generated (DeepSeek safe): {len(gen)}")

    out = []
    print()
    print("=" * 116)
    print("IS THERE HARM SIGNAL IN THE NATURAL PAIRS ALONE?")
    print(f"grouped {args.folds}-fold CV over the {len(nat)} natural pairs "
          f"(both sides LLaDA); v_harm refitted per fold")
    print("=" * 116)
    print(f"{'layout':7s} {'L':>3s} {'AUC_nat_cv':>11s} {'95% CI':>14s} {'n_scored':>9s} "
          f"{'AUC_gen_cv':>11s} {'cos(all,nat)':>13s} {'cos(nat,src)':>13s} {'cos(nat,len)':>13s}")
    print("-" * 116)
    for lo in layouts:
        for L in layers:
            a_nat, ci_nat, n_nat = kfold_auc(idx, nat, lo, L, args.folds, args.seed, "nat")
            a_gen, _, _ = kfold_auc(idx, gen, lo, L, args.folds, args.seed, "gen")

            def stackh(ids, side):
                return np.stack([idx[(c, lo, side)]["h"][L].numpy().astype(np.float64) for c in ids])

            allids = nat + gen
            v_all = unit(stackh(allids, "harm").mean(0) - stackh(allids, "safe").mean(0))
            v_nat = unit(stackh(nat, "harm").mean(0) - stackh(nat, "safe").mean(0))
            v_src = unit(stackh(nat, "safe").mean(0) - stackh(gen, "safe").mean(0))
            Hh, Hs = stackh(allids, "harm"), stackh(allids, "safe")
            T = np.array([idx[(c, lo, "harm")]["n_resp_tokens"] for c in allids]
                         + [idx[(c, lo, "safe")]["n_resp_tokens"] for c in allids])
            P = np.concatenate([Hh, Hs])
            med = float(np.median(T))
            v_len = unit(P[T > med].mean(0) - P[T <= med].mean(0))

            row = dict(layout=lo, layer=L, auc_natural_cv=a_nat, auc_natural_cv_ci=list(ci_nat),
                       n_scored=n_nat, auc_generated_cv=a_gen,
                       cos_all_nat=cos(v_all, v_nat), cos_nat_src=cos(v_nat, v_src),
                       cos_nat_len=cos(v_nat, v_len))
            out.append(row)
            ci = f"[{ci_nat[0]:.2f},{ci_nat[1]:.2f}]"
            print(f"{lo:7s} {L:3d} {a_nat:11.3f} {ci:>14s} {n_nat:9d} {a_gen:11.3f} "
                  f"{row['cos_all_nat']:13.3f} {row['cos_nat_src']:13.3f} {row['cos_nat_len']:13.3f}")
    print("-" * 116)

    best = max(out, key=lambda r: r["auc_natural_cv"])
    print()
    print("READING:")
    print(f"  best natural-only CV AUC: {best['layout']} L{best['layer']} = "
          f"{best['auc_natural_cv']:.3f} CI[{best['auc_natural_cv_ci'][0]:.2f},"
          f"{best['auc_natural_cv_ci'][1]:.2f}]  (chance = 0.50)")
    if best["auc_natural_cv_ci"][0] > 0.5:
        print("  -> natural pairs DO carry harm signal above chance. The gate failed on")
        print("     DeepSeek contamination, not on the premise. Fix = more natural pairs.")
    else:
        print("  -> natural pairs carry NO reliable harm signal (CI includes chance).")
        print("     More natural pairs of this kind will not rescue it; the harm-vs-safe")
        print("     contrast as currently defined is not what separates these states.")
    print(f"  generated-pair CV AUC ~{np.mean([r['auc_generated_cv'] for r in out]):.3f} "
          f"-> the DeepSeek half is trivially separable (authorship, not harm)")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
