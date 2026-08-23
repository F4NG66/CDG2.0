#!/usr/bin/env python3
"""clockv2/build_tau.py — READ-ONLY denoising-progress (tau) probe.

WHY THIS IS REBUILT RATHER THAN REUSED AS-IS
--------------------------------------------
clock_attack/ ships a mu-bank and a *steerer*, but no tau PROBE. StepSteerer's hook
does `h[0, mrow, :] += vec` -- it MODIFIES activations, which this study forbids, so it
is never imported. Only the mu-bank (a read-only data artifact) is conceptually reused.

The shipped bank cannot be used directly either: its meta says block_length=32, i.e. a
4-block restricted schedule (4 x 32 steps, each block restarting from fully-masked).
Phase 2 runs BOTH conditions on the unified fill_all_masks schedule (one 128-step ramp
over every mask), because DIJA's injected prompt blanks are only fillable that way.
Those are different hidden trajectories, so the step->hidden mapping would not transfer.
We therefore rebuild the bank under the exact Phase-2 schedule, and extend it from
{25,29} to the full L16..L31 band.

The bank is built on the SAME held-out neutral seeds as the original (D_neutral_clean,
D000..), which have zero id-overlap and zero behavior-text overlap with the 100 eval
cases (verified). Fit seeds and convention-verification seeds are DISJOINT.

TAU DECODE (read-only): given hidden h at layer L,
    tau_soft = sum_t softmax(-||h - mu_step[t]|| / T_temp) * (t / (steps-1))
    tau_nn   = argmin_t ||h - mu_step[t]|| / (steps - 1)
Both are pure readouts. Nothing is written back.

TAU CONVENTION IS VERIFIED, NOT ASSUMED: on held-out neutral runs we correlate the
decoded tau against the physical canvas clock (1 - mask_ratio). A strong positive
correlation confirms "tau increases with denoising progress". The sign is reported,
never asserted a priori.

    python clockv2/build_tau.py --n-fit 32 --n-verify 8
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from common import (BAND, D_NEUTRAL, GEN_LENGTH, HELDOUT_DIR, STEPS, StepCapture,
                    assert_readonly_hooks, eval_behaviors, load_model, max_jaccard_vs,
                    save_probe)


def neutral_seeds(j_thresh=0.4):
    """D_neutral_clean, minus anything close to an eval behavior."""
    A = eval_behaviors()
    out = []
    for d in json.load(open(D_NEUTRAL)):
        if max_jaccard_vs(d["behavior"], A) <= j_thresh:
            out.append(d["behavior"])
    return out


def run_one(model, tok, blocks, behavior, device):
    from common import capture_denoise
    s = tok.apply_chat_template([{"role": "user", "content": behavior}],
                                add_generation_prompt=True, tokenize=False)
    ids = tok([s], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    cap = StepCapture(blocks, BAND, gen_start=ids.shape[1])
    assert_readonly_hooks(blocks)
    try:
        _, info = capture_denoise(model, ids, cap, steps=STEPS, gen_length=GEN_LENGTH)
        H = {L: torch.stack([st[L] for st in cap.resp_mean]) for L in BAND}   # [T, d]
    finally:
        cap.close()
    assert info["n_inject"] == 0, "neutral seeds must carry no prompt-embedded blanks"
    return H, info["mask_ratio"]


def decode_tau(h, mu_step, t_temp=1.0):
    """READ-ONLY tau decode. h:[d]  mu_step:[T,d] -> (tau_soft, tau_nn) in [0,1]."""
    d = torch.cdist(h.unsqueeze(0).double(), mu_step.double()).squeeze(0)   # [T]
    T = mu_step.shape[0]
    grid = torch.arange(T, dtype=torch.float64) / (T - 1)
    w = torch.softmax(-d / (d.std().clamp_min(1e-6) * t_temp), dim=0)
    return float((w * grid).sum()), float(grid[int(torch.argmin(d))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-fit", type=int, default=32)
    ap.add_argument("--n-verify", type=int, default=8)
    ap.add_argument("--j-thresh", type=float, default=0.4)
    args = ap.parse_args()

    seeds = neutral_seeds(args.j_thresh)
    fit_seeds = seeds[:args.n_fit]
    ver_seeds = seeds[args.n_fit:args.n_fit + args.n_verify]      # DISJOINT from fit
    print(f"[tau] neutral pool={len(seeds)} (after leakage filter) -> "
          f"fit n={len(fit_seeds)}, verify n={len(ver_seeds)} (disjoint)")
    assert not (set(fit_seeds) & set(ver_seeds)), "fit/verify seeds overlap"

    print("[tau] loading model ...", flush=True)
    tok, model, blocks = load_model(device=args.device)

    # ---- fit the mu-bank under the Phase-2 unified schedule ----
    acc = {L: torch.zeros(STEPS, 4096, dtype=torch.float64) for L in BAND}
    sq = {L: torch.zeros(STEPS, 4096, dtype=torch.float64) for L in BAND}
    for k, beh in enumerate(fit_seeds):
        H, mr = run_one(model, tok, blocks, beh, args.device)
        for L in BAND:
            acc[L] += H[L].double()
            sq[L] += H[L].double() ** 2
        if k == 0:
            # the clean canvas clock must be the deterministic 1-per-step ramp
            print(f"[tau] seed0 mask_ratio: t0={mr[0]:.3f} t64={mr[64]:.3f} "
                  f"t127={mr[127]:.3f} (expect 1.000 / 0.500 / 0.008)")
        print(f"  [fit] {k+1}/{len(fit_seeds)}", flush=True)
    n = len(fit_seeds)
    mu_step = {L: (acc[L] / n).float() for L in BAND}
    sd_step = {L: ((sq[L] / n - (acc[L] / n) ** 2).clamp_min(0).sqrt()).float() for L in BAND}

    # ---- verify the tau convention empirically on disjoint held-out seeds ----
    rows = []
    for k, beh in enumerate(ver_seeds):
        H, mr = run_one(model, tok, blocks, beh, args.device)
        tau_from_mask = [1.0 - r for r in mr]
        for L in BAND:
            dec = [decode_tau(H[L][t], mu_step[L]) for t in range(STEPS)]
            soft = [d[0] for d in dec]
            nn = [d[1] for d in dec]
            rows.append((L, soft, nn, tau_from_mask))
        print(f"  [verify] {k+1}/{len(ver_seeds)}", flush=True)

    def pearson(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a) ** 0.5
        vb = sum((x - mb) ** 2 for x in b) ** 0.5
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / max(va * vb, 1e-9)

    per_layer = {}
    for L in BAND:
        rs = [r for r in rows if r[0] == L]
        rsoft = sum(pearson(r[1], r[3]) for r in rs) / len(rs)
        rnn = sum(pearson(r[2], r[3]) for r in rs) / len(rs)
        mae = sum(sum(abs(s - m) for s, m in zip(r[1], r[3])) / STEPS for r in rs) / len(rs)
        per_layer[L] = {"r_soft": rsoft, "r_nn": rnn, "mae_soft": mae}

    save_probe("tau_bank.pt", {
        "kind": "denoising_progress_bank_READ_ONLY_PROBE",
        "mu_step": mu_step, "sd_step": sd_step,
        "layers": BAND, "total_steps": STEPS, "gen_length": GEN_LENGTH,
        "n_fit": n, "fit_schedule": "unified fill_all_masks (single 128-step ramp), temp=0",
        "pool": "mean over response canvas",
        "captured_at": "block_output(out[0])",
        "convention_verified": per_layer,
        "usage": "PROJECT/DECODE ONLY. Never added to an activation. StepSteerer not imported.",
    })

    print("\n=== tau probe sanity (read-only) ===")
    print(f"  bank fit on n={n} held-out neutral seeds; verified on n={len(ver_seeds)} disjoint")
    print(f"  {'layer':>5} {'r(tau_soft, 1-mask)':>20} {'r(tau_nn, 1-mask)':>18} {'MAE_soft':>9} "
          f"{'||mu_step[64]||':>15}")
    for L in BAND:
        p = per_layer[L]
        print(f"  {L:5d} {p['r_soft']:20.3f} {p['r_nn']:18.3f} {p['mae_soft']:9.3f} "
              f"{mu_step[L][64].norm().item():15.2f}")
    best = max(BAND, key=lambda L: per_layer[L]["r_soft"])
    print(f"\n  TAU CONVENTION (empirical): r>0 means tau RISES with denoising progress.")
    print(f"  best layer L{best}: r_soft={per_layer[best]['r_soft']:+.3f}")
    for L in (25, 29):
        print(f"  tau layer L{L}: r_soft={per_layer[L]['r_soft']:+.3f} "
              f"r_nn={per_layer[L]['r_nn']:+.3f}")
    print(f"  all finite: {all(torch.isfinite(mu_step[L]).all().item() for L in BAND)}")


if __name__ == "__main__":
    main()
