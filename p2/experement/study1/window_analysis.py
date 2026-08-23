#!/usr/bin/env python3
"""study1/window_analysis.py — VIABILITY test on the 10 pilot cases (analysis only).

Reads the already-saved dija payloads at /scratch/ore99/study1_pilot/ (hidden +
commit trajectory) and asks the one question that decides Study 1: is there a gap
between when harm becomes DETECTABLE (k_detect, from an out_mask-ONLY probe) and
when it becomes COMMITTED (k_commit, when harm-span tokens leave the mask set)?

  1. k_commit : per inclusive-positive case, harm span -> captured positions -> commit_step.
  2. k_detect : leave-one-out diff-of-means probe on the SEPARATED out_mask positions
                (reply_role_mask & fillable_reply_mask; out_unmask EXCLUDED) at each step,
                separating inclusive-positive from safe/refusal. Threshold at a target FPR
                on the safe/refusal (benign-outcome) cases. Per-step AUC + first-cross step.
  3. Delta  = k_commit - k_detect, per positive, pooled. Sign + magnitude.
  4. Verdict (n=10 => direction only).

NO GPU generation, NO capture. Loads only the tokenizer (offline) for span mapping.

  python study1/window_analysis.py --pilot /scratch/ore99/study1_pilot
"""
from __future__ import annotations
import argparse
import json
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
for p in (HERE, EXP, os.path.join(EXP, "dija_attack")):
    if p not in sys.path:
        sys.path.insert(0, p)

import span_locator  # noqa: E402
from capture_union import map_span_to_commit, MODEL_ID  # noqa: E402


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


def pooled_out_mask_features(payload):
    """Per-step mean of hidden over OUT_MASK fillable positions -> [n_steps, n_layers, d].
    valid[i] False when no out_mask position remains at step i."""
    h = payload["hidden"]                                   # [S, L, R, D] fp16
    role = payload["reply_role_mask"] & payload["fillable_reply_mask"]  # [S, R] out_mask only
    S, L, R, D = h.shape
    pooled = torch.zeros(S, L, D, dtype=torch.float32)
    valid = torch.zeros(S, dtype=torch.bool)
    for i in range(S):
        idx = role[i].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        pooled[i] = h[i][:, idx, :].float().mean(dim=1)     # [L, D]
        valid[i] = True
    # aux: mean entropy over out_mask positions (already computed at capture)
    ent = payload["entropy"].float()                        # [S, R]
    ent_om = torch.zeros(S)
    for i in range(S):
        idx = role[i].nonzero(as_tuple=True)[0]
        ent_om[i] = ent[i][idx].mean() if idx.numel() else float("nan")
    return pooled, valid, ent_om


def loo_probe(feat, valid, labels, *, layer_sel, fpr_target=0.0):
    """Leave-one-out diff-of-means probe at every step.

    feat   : [N, S, L, D] fp32 pooled out_mask features.
    layer_sel : list of layer indices to concat (feature = those layers flattened).
    Returns per-step: auc[S], and per-case first-cross step (k_detect) using a
    threshold set so FPR<=fpr_target on the TRAINING negatives (benign-outcome cases).
    """
    N, S, L, D = feat.shape
    y = torch.tensor(labels)
    Xall = feat[:, :, layer_sel, :].reshape(N, S, -1)       # [N, S, D']
    auc_per_step = [None] * S
    heldout_score = torch.full((N, S), float("nan"))
    thresh = torch.full((S,), float("nan"))                 # representative (full-neg) threshold

    for i in range(S):
        vmask = valid[:, i]
        if int(vmask.sum()) < N:                            # require all cases valid at step i
            # still compute over valid subset if enough of each class
            pass
        idx_valid = vmask.nonzero(as_tuple=True)[0].tolist()
        yv = [labels[j] for j in idx_valid]
        if sum(yv) < 2 or (len(yv) - sum(yv)) < 2:
            continue
        Xi = Xall[:, i, :]                                   # [N, D']
        scores = {}
        for held in idx_valid:
            train = [j for j in idx_valid if j != held]
            Xt = Xi[train]
            mu = Xt.mean(0); sd = Xt.std(0).clamp_min(1e-6)
            Z = (Xt - mu) / sd
            yt = torch.tensor([labels[j] for j in train])
            w = Z[yt == 1].mean(0) - Z[yt == 0].mean(0)      # diff-of-means direction
            zheld = (Xi[held] - mu) / sd
            scores[held] = float(zheld @ w)
            heldout_score[held, i] = scores[held]
        sc = [scores[j] for j in idx_valid]
        auc_per_step[i] = _auc(sc, yv)
        # representative threshold: FPR<=target on the (held-out) negatives' scores
        neg_scores = sorted([scores[j] for j in idx_valid if labels[j] == 0])
        if neg_scores:
            # smallest t with (#neg > t)/n_neg <= fpr_target  => pick the ceil((1-fpr)*n)-th
            k = max(0, int((1.0 - fpr_target) * len(neg_scores)) - 1)
            thresh[i] = neg_scores[min(k, len(neg_scores) - 1)]

    # per-case k_detect = earliest step its held-out score exceeds that step's threshold
    kdetect = {}
    for j in range(N):
        first = None
        for i in range(S):
            t = thresh[i]
            s = heldout_score[j, i]
            if not torch.isnan(t) and not torch.isnan(s) and s > t:
                first = i
                break
        kdetect[j] = first
    return auc_per_step, kdetect, heldout_score, thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="/scratch/ore99/study1_pilot")
    ap.add_argument("--layers-sel", type=int, nargs="+", default=None,
                    help="indices into the 5 captured layers to use (default: all)")
    ap.add_argument("--fpr", type=float, default=0.0, help="target FPR on benign-outcome cases")
    args = ap.parse_args()

    judged = [json.loads(l) for l in open(os.path.join(args.pilot, "judged.jsonl")) if l.strip()]
    pos_ids = [j["id"] for j in judged if j["harm_delivered_inclusive"] is True]
    neg_ids = [j["id"] for j in judged if j["harm_delivered_inclusive"] is False]
    print(f"inclusive-positive (harm) n={len(pos_ids)}: {pos_ids}")
    print(f"safe/refusal (benign)     n={len(neg_ids)}: {neg_ids}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    ids_order = pos_ids + neg_ids
    labels = [1] * len(pos_ids) + [0] * len(neg_ids)

    # ---------- 1. k_commit on the harm span of each positive ----------
    print("\n" + "=" * 78 + "\n1. k_commit  (harm-span positions leaving the mask set)\n" + "=" * 78)
    kcommit_median = {}
    pooled_all = []
    pooled_span_commits = []
    for cid, lab in zip(ids_order, labels):
        payload = torch.load(os.path.join(args.pilot, f"{cid}__dija.pt"),
                             map_location="cpu", weights_only=False)
        if lab == 1:
            dr = map_span_to_commit(payload, tok, span_locator)
            hc = dr["hit_commits"]                 # PRIMARY: injected scaffold-blank commits
            sp = dr["span_commit"]                 # secondary: fixed-locator span-based
            if hc:
                kcommit_median[cid] = st.median(hc)
                pooled_span_commits.extend(hc)
                print(f"  {cid}: scaffold_blanks={dr['n_scaffold_blank']:3d}  "
                      f"k_commit(PRIMARY) min={min(hc):3d} med={int(st.median(hc)):3d} max={max(hc):3d}"
                      f"  |  span_pos={dr['n_span_positions']:3d} span_med={sp['median']} "
                      f"(was 157-163 for A003/A006)")
            else:
                print(f"  {cid}: NO scaffold-blank positions mapped")
        feat, valid, ent_om = pooled_out_mask_features(payload)
        pooled_all.append((feat, valid, ent_om))
        del payload

    if pooled_span_commits:
        pc = sorted(pooled_span_commits)
        q = lambda f: pc[min(len(pc) - 1, int(f * len(pc)))]
        print(f"\n  POOLED harm-span k_commit (n={len(pc)} positions): "
              f"p10={q(.10)} p25={q(.25)} median={q(.50)} p75={q(.75)} p90={q(.90)} "
              f"mean={st.mean(pc):.1f}")

    # ---------- 2. k_detect: out_mask-only LOO probe per step ----------
    S = pooled_all[0][0].shape[0]
    L = pooled_all[0][0].shape[1]
    feat = torch.stack([p[0] for p in pooled_all])          # [N, S, L, D]
    valid = torch.stack([p[1] for p in pooled_all])         # [N, S]
    layer_sel = args.layers_sel if args.layers_sel is not None else list(range(L))

    print("\n" + "=" * 78 + f"\n2. k_detect  (out_mask-only LOO diff-of-means probe; layers idx {layer_sel})\n" + "=" * 78)
    auc_step, kdetect, hscore, thresh = loo_probe(feat, valid, labels,
                                                  layer_sel=layer_sel, fpr_target=args.fpr)
    # per-step AUC table at representative steps
    print("  step |  AUC  | n_out_mask(mean)  (out_mask-only separation of harm vs benign)")
    for i in range(0, S, 8):
        a = auc_step[i]
        nmask = float(valid[:, i].sum())
        astr = f"{a:.2f}" if a is not None else " -- "
        print(f"  {i:4d} | {astr} | cases_valid={int(nmask)}")
    # first step AUC crosses 0.8 and 0.9 (stable = also next valid step high)
    def first_cross(th):
        for i in range(S):
            a = auc_step[i]
            if a is not None and a >= th:
                return i
        return None
    print(f"\n  first step AUC>=0.70: {first_cross(0.70)}   "
          f">=0.80: {first_cross(0.80)}   >=0.90: {first_cross(0.90)}")
    best = max((a for a in auc_step if a is not None), default=None)
    print(f"  peak AUC across steps: {best:.2f}" if best is not None else "  peak AUC: n/a")

    # ---------- 3. Delta = k_commit - k_detect ----------
    print("\n" + "=" * 78 + "\n3. Delta = k_commit - k_detect  (per inclusive-positive)\n" + "=" * 78)
    deltas = []
    for n, cid in enumerate(pos_ids):
        kd = kdetect[n]
        kc = kcommit_median.get(cid)
        if kd is None or kc is None:
            print(f"  {cid}: k_detect={kd}  k_commit_median={kc}  Delta= n/a")
            continue
        kc_i = int(round(kc)); d = kc_i - kd
        deltas.append(d)
        print(f"  {cid}: k_detect={kd:3d}  k_commit_median={kc_i:3d}  Delta={d:+d}")
    if deltas:
        print(f"\n  POOLED Delta: n={len(deltas)}  median={int(st.median(deltas)):+d}  "
              f"mean={st.mean(deltas):+.1f}  min={min(deltas):+d} max={max(deltas):+d}  "
              f"positive={sum(d > 0 for d in deltas)}/{len(deltas)}")

    # ---------- 4. entropy aux cross-check ----------
    print("\n" + "=" * 78 + "\n4. AUX: out_mask mean-entropy separation (no training)\n" + "=" * 78)
    ent = torch.stack([p[2] for p in pooled_all])           # [N, S]
    for i in range(0, S, 16):
        col = ent[:, i]
        sc = [-float(col[j]) for j in range(len(labels))]   # lower entropy -> more "decided"
        if any(torch.isnan(col)):
            continue
        a = _auc(sc, labels)
        if a is not None:
            print(f"  step {i:4d}: entropy-AUC(harm=low-ent)={a:.2f}")

    print("\n[done] n=10 -> DIRECTION ONLY. See verdict logic: Delta>0 sizable => viable; "
          "Delta~0/negative => detection not earlier than commit => pivot to pre-gen gating.")


if __name__ == "__main__":
    main()
