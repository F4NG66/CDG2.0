#!/usr/bin/env python3
"""study1/kdetect_grouped.py — the HONEST k_detect (STEP 4 of the expansion).

The pilot's k_detect (~0, AUC 0.81 already at step 0) was confounded: it separated
7 harmful prompts from 3 benign-outcome prompts across DIFFERENT behaviors, so the
step-0 canvas alone encodes which request it is. That is prompt identity, not a
decision signal.

This script removes that confound by construction:

  LOCK 1  within-behavior contrast : positives are dija arms judged harm-delivered
          (valence-INCLUSIVE), negatives are the MATCHED benign_op arm of the SAME
          A-id (byte-identical <mask:N> blank geometry, benign fill). The probe can
          no longer win by recognising the behavior -- both arms share it.
  LOCK 1  grouped CV by A-id       : leave-one-GROUP-out; every arm of one base
          request is held out together, so no fold ever trains on a sibling.
  control label permutation        : arm labels are swapped WITHIN each group (the
          exact null for "outcome within behavior"). Must collapse to ~0.5.
  threshold FPR is set on the benign_op arm (Group C) -- not on clean, not on the
          pilot's 3 safe cases.

out_unmask is the forbidden read and is absent from the ragged payloads entirely.

  python study1/kdetect_grouped.py --data /scratch/ore99/study1_expansion \
                                   --judged /scratch/ore99/study1_expansion/judged.jsonl
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
for p in (HERE, EXP, os.path.join(EXP, "dija_attack")):
    if p not in sys.path:
        sys.path.insert(0, p)

import span_locator  # noqa: E402
from capture_union import map_span_to_commit, MODEL_ID, R_SCAFFOLD  # noqa: E402


# ---------------------------------------------------------------- small stats
def _auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    c = 0.0
    for p in pos:
        for n in neg:
            c += (p > n) + 0.5 * (p == n)
    return c / (len(pos) * len(neg))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# ---------------------------------------------------------------- features
def pooled_out_mask_ragged(payload, *, scaffold_only=False):
    """Per-step mean of hidden over the stored OUT_MASK cells -> [S, L, D] fp32.

    The ragged payload stores exactly `reply_role_mask & fillable_reply_mask`, so no
    out_unmask position can leak in. valid[i]=False when step i has no cell left."""
    hf = payload["hidden_flat"]                     # [T, L, D] fp16
    ptr = payload["step_ptr"].tolist()              # [S+1]
    cpos = payload["cell_pos"]                      # [T]
    S = len(ptr) - 1
    L, D = hf.shape[1], hf.shape[2]
    keep_pos = None
    if scaffold_only:
        keep_pos = (payload["region_id"] == R_SCAFFOLD)

    pooled = torch.zeros(S, L, D, dtype=torch.float32)
    valid = torch.zeros(S, dtype=torch.bool)
    for i in range(S):
        a, b = ptr[i], ptr[i + 1]
        if b <= a:
            continue
        blk = hf[a:b]                               # [n_i, L, D]
        if keep_pos is not None:
            sel = keep_pos[cpos[a:b].long()]
            if not bool(sel.any()):
                continue
            blk = blk[sel]
        pooled[i] = blk.float().mean(dim=0)
        valid[i] = True
    return pooled, valid


# ---------------------------------------------------------------- probe
def grouped_probe(feat, valid, labels, groups, *, layer_sel, permute=False, rng=None):
    """Leave-one-GROUP-out diff-of-means probe at every denoising step.

    feat  : [N, S, L, D]      labels: list[int]      groups: list[str] (A-id)
    permute: swap the labels WITHIN each group (exact null; needs 2 arms/group).
    Returns auc_per_step[S], heldout_score[N,S].
    """
    N, S, L, D = feat.shape
    lab = list(labels)
    if permute:
        by_g = {}
        for j, g in enumerate(groups):
            by_g.setdefault(g, []).append(j)
        # Within-group permutation is the exact null ONLY when groups hold >=2 arms.
        # For singleton groups (the within-dija design) it is a no-op that silently
        # returns the real labels, so the "null" equals the real AUC. Fall back to a
        # global shuffle there -- groups are singletons, so nothing is broken by it.
        if all(len(v) < 2 for v in by_g.values()):
            vals = list(lab)
            rng.shuffle(vals)
            lab = vals
        else:
            for g, idxs in by_g.items():
                vals = [lab[j] for j in idxs]
                rng.shuffle(vals)
                for j, v in zip(idxs, vals):
                    lab[j] = v

    X = feat[:, :, layer_sel, :].reshape(N, S, -1)          # [N, S, D']
    auc_per_step = [None] * S
    heldout = torch.full((N, S), float("nan"))
    uniq = sorted(set(groups))

    for i in range(S):
        Xi = X[:, i, :]
        for g in uniq:                                       # leave-one-group-out
            held = [j for j in range(N) if groups[j] == g and valid[j, i]]
            train = [j for j in range(N) if groups[j] != g and valid[j, i]]
            if not held:
                continue
            yt = [lab[j] for j in train]
            if sum(yt) < 2 or (len(yt) - sum(yt)) < 2:
                continue
            Xt = Xi[train]
            mu = Xt.mean(0); sd = Xt.std(0).clamp_min(1e-6)  # fit on TRAIN only
            Z = (Xt - mu) / sd
            yy = torch.tensor(yt)
            w = Z[yy == 1].mean(0) - Z[yy == 0].mean(0)
            for j in held:
                heldout[j, i] = float(((Xi[j] - mu) / sd) @ w)
        sc, yv = [], []
        for j in range(N):
            if not torch.isnan(heldout[j, i]):
                sc.append(float(heldout[j, i])); yv.append(lab[j])
        if sum(yv) >= 2 and (len(yv) - sum(yv)) >= 2:
            auc_per_step[i] = _auc(sc, yv)
    return auc_per_step, heldout


def threshold_on_benign(heldout, arms, *, fpr_target=0.05):
    """Per-step threshold with FPR<=target on the benign_op (Group C) held-out scores."""
    N, S = heldout.shape
    th = torch.full((S,), float("nan"))
    for i in range(S):
        neg = sorted(float(heldout[j, i]) for j in range(N)
                     if arms[j] == "benign_op" and not torch.isnan(heldout[j, i]))
        if len(neg) < 3:
            continue
        k = max(0, int(math.ceil((1.0 - fpr_target) * len(neg))) - 1)
        th[i] = neg[min(k, len(neg) - 1)]
    return th


def first_cross(heldout_row, th, *, persist=1):
    """Earliest step whose score exceeds that step's threshold for `persist` steps."""
    S = heldout_row.shape[0]
    run = 0
    for i in range(S):
        t, s = th[i], heldout_row[i]
        if torch.isnan(t) or torch.isnan(s):
            run = 0
            continue
        run = run + 1 if s > t else 0
        if run >= persist:
            return i - persist + 1
    return None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/scratch/ore99/study1_expansion")
    ap.add_argument("--judged", default=None, help="judged.jsonl (valence-inclusive)")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="layer POSITIONS into the stored layer axis (default: all)")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--persist", type=int, default=1, help="steps above threshold to fire")
    ap.add_argument("--nperm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scaffold-only", action="store_true",
                    help="pool only R_SCAFFOLD cells (harm-region readout variant)")
    ap.add_argument("--include-dija-negatives", action="store_true",
                    help="also use harm-NOT-delivered dija arms as negatives")
    ap.add_argument("--contrast", default="harm",
                    choices=["harm", "arm-placebo", "within-dija"],
                    help="harm=dija(harm-delivered) vs matched benign_op (primary); "
                         "arm-placebo=dija(harm NOT delivered) vs its matched benign_op "
                         "-- SAME arm contrast with NO harm difference, so any AUC here is "
                         "pure arm/prompt-text leakage; within-dija=harm-delivered vs "
                         "harm-not-delivered, dija arm only (no arm confound)")
    ap.add_argument("--delta-from-step0", action="store_true",
                    help="feature = pooled[i] - pooled[0]: removes the static prompt "
                         "encoding present before any denoising happens")
    args = ap.parse_args()

    judged_path = args.judged or os.path.join(args.data, "judged.jsonl")
    jud = []
    with open(judged_path) as fh:
        for line in fh:
            if line.strip():
                jud.append(json.loads(line))
    pos_ids = {r["id"] for r in jud
               if r.get("arm") == "dija" and r.get("harm_delivered_inclusive") is True}
    neg_ids = {r["id"] for r in jud
               if r.get("arm") == "dija" and r.get("harm_delivered_inclusive") is not True}
    print(f"[judge] dija inclusive-positive: {len(pos_ids)}  negative: {len(neg_ids)}")

    # ---- assemble the within-behavior design ------------------------------
    rows = []          # (traj_id, A-id, arm, label)
    if args.contrast == "harm":
        for cid in sorted(pos_ids):
            rows.append((f"{cid}__dija", cid, "dija", 1))
            rows.append((f"{cid}__benign_op", cid, "benign_op", 0))
        if args.include_dija_negatives:
            for cid in sorted(neg_ids):
                rows.append((f"{cid}__dija", cid, "dija", 0))
                rows.append((f"{cid}__benign_op", cid, "benign_op", 0))
    elif args.contrast == "arm-placebo":
        # PLACEBO: label the ARM, on the cases where the dija arm delivered NO harm.
        # There is no harm difference to find, so any AUC is arm/prompt-text leakage.
        for cid in sorted(neg_ids):
            rows.append((f"{cid}__dija", cid, "dija", 1))
            rows.append((f"{cid}__benign_op", cid, "benign_op", 0))
    else:  # within-dija: outcome contrast with the arm held fixed
        for cid in sorted(pos_ids):
            rows.append((f"{cid}__dija", cid, "dija", 1))
        for cid in sorted(neg_ids):
            rows.append((f"{cid}__dija", cid, "dija", 0))
    rows = [r for r in rows if os.path.exists(os.path.join(args.data, r[0] + ".pt"))]
    print(f"[design] {len(rows)} arms over {len(set(r[1] for r in rows))} A-id groups "
          f"(pos={sum(r[3] for r in rows)})")
    if sum(r[3] for r in rows) < 4:
        print("!! too few positives for a grouped probe — STOP")
        sys.exit(2)

    feats, valids, labels, groups, arms, payloads = [], [], [], [], [], {}
    for traj, cid, arm, lab in rows:
        p = torch.load(os.path.join(args.data, traj + ".pt"),
                       map_location="cpu", weights_only=False)
        f, v = pooled_out_mask_ragged(p, scaffold_only=args.scaffold_only)
        if args.delta_from_step0 and bool(v[0]):
            f = f - f[0:1]          # trajectory-relative: kill the static prompt encoding
        feats.append(f); valids.append(v)
        labels.append(lab); groups.append(cid); arms.append(arm)
        if arm == "dija" and lab == 1:
            payloads[cid] = p
        else:
            del p
    feat = torch.stack(feats)                                  # [N, S, L, D]
    valid = torch.stack(valids)                                # [N, S]
    N, S, L, D = feat.shape
    layer_sel = args.layers if args.layers is not None else list(range(L))
    print(f"[feat] N={N} steps={S} layers_stored={L} d={D} layer_sel={layer_sel} "
          f"{'SCAFFOLD-ONLY' if args.scaffold_only else 'whole-region'}")

    # ---- real probe --------------------------------------------------------
    auc, heldout = grouped_probe(feat, valid, labels, groups, layer_sel=layer_sel)
    print("\n" + "=" * 78 + "\nPER-STEP AUC — grouped CV by A-id, WITHIN matched behavior\n" + "=" * 78)
    for i in range(0, S, 8):
        cells = " ".join(f"{i+k:3d}:{'  n/a' if auc[i+k] is None else f'{auc[i+k]:.3f}'}"
                         for k in range(8) if i + k < S)
        print("  " + cells)
    got = [(i, a) for i, a in enumerate(auc) if a is not None]
    if got:
        bi, ba = max(got, key=lambda t: t[1])
        print(f"\n  step0 AUC = {auc[0] if auc[0] is not None else float('nan')}")
        print(f"  peak  AUC = {ba:.3f} @ step {bi}")

    # ---- permutation control ----------------------------------------------
    rng = random.Random(args.seed)
    peaks = []
    for _ in range(args.nperm):
        pa, _ = grouped_probe(feat, valid, labels, groups, layer_sel=layer_sel,
                              permute=True, rng=rng)
        vals = [a for a in pa if a is not None]
        if vals:
            peaks.append(max(vals))
    if peaks:
        print(f"\n[control] label-permutation WITHIN group, {len(peaks)} draws: "
              f"peak AUC mean={st.mean(peaks):.3f} max={max(peaks):.3f} "
              f"(expect ~0.5-0.6; the real peak must clear this)")

    # ---- k_detect ----------------------------------------------------------
    th = threshold_on_benign(heldout, arms, fpr_target=args.fpr)
    print(f"\n[k_detect] threshold FPR<={args.fpr} on the benign_op arm, "
          f"persist={args.persist} step(s)")
    kdet = {}
    for j, (traj, cid, arm, lab) in enumerate(rows):
        if lab == 1:
            kdet[cid] = first_cross(heldout[j], th, persist=args.persist)

    # ---- k_commit + Delta --------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("\n" + "=" * 78 + "\nDELTA = k_commit - k_detect  (per inclusive-positive)\n" + "=" * 78)
    print(f"{'id':6} {'k_commit(med)':>13} {'min':>5} {'max':>5} {'k_detect':>9} {'Delta':>7}")
    deltas, kcs, kds = [], [], []
    for cid in sorted(payloads):
        dr = map_span_to_commit(payloads[cid], tok, span_locator)
        mmm = dr["commit_of_span"]           # PRIMARY: scaffold-blank anchored
        if not mmm or mmm.get("median") is None:
            print(f"{cid:6} {'—':>13}  (no scaffold commits)")
            continue
        kc = int(round(mmm["median"])); kd = kdet.get(cid)
        kcs.append(kc)
        if kd is None:
            print(f"{cid:6} {kc:13d} {mmm['min']:5d} {mmm['max']:5d} {'never':>9} {'—':>7}")
            continue
        d = kc - kd
        kds.append(kd); deltas.append(d)
        print(f"{cid:6} {kc:13d} {mmm['min']:5d} {mmm['max']:5d} {kd:9d} {d:+7d}")

    if deltas:
        npos = sum(1 for d in deltas if d > 0)
        lo, hi = wilson(npos, len(deltas))
        print(f"\n  k_commit pooled median = {st.median(kcs)}")
        print(f"  k_detect pooled median = {st.median(kds)}")
        print(f"  Delta  median={st.median(deltas):+.1f}  mean={st.mean(deltas):+.1f}  "
              f"min={min(deltas):+d}  max={max(deltas):+d}")
        print(f"  Delta>0 in {npos}/{len(deltas)}  Wilson-95 = [{lo:.2f}, {hi:.2f}]")
        fired = len(deltas); tot = len(payloads)
        flo, fhi = wilson(fired, tot)
        print(f"  probe fired on {fired}/{tot} positives  Wilson-95 = [{flo:.2f}, {fhi:.2f}]")
    else:
        print("\n  no Delta computable — probe never fired on any positive")


if __name__ == "__main__":
    main()
