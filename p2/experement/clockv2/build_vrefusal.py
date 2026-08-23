#!/usr/bin/env python3
"""clockv2/build_vrefusal.py — refusal direction as a READ-ONLY MEASUREMENT AXIS.

    v_refusal[L] = normalize( mean_harmful[L] - mean_harmless[L] )        (Arditi Eq.1-2)

extracted from PROMPT-ONLY forward passes at the last prompt token.

SCOPE — this probe exists ONLY to be projected onto hidden states and read.
  * It is never added to, subtracted from, or scaled into any activation.
  * It is never used to generate text.
  * It is never used to ablate, bypass, or reduce refusal.
  * No generation happens in this file at all: forward passes only, no denoising loop.
The artifact is a direction we DOT WITH. That is its entire purpose.

LEAKAGE: extracted on a held-out contrastive set disjoint from the 100 eval cases.
  harmful  <- unsafe.jsonl (MedSafetyBench), filtered at Jaccard<=0.4 vs eval behaviors
  harmless <- D_neutral_clean (D000..),      filtered the same way
Both halves are medical, so the difference isolates harm rather than domain.
curated/ and neutral/ are NOT used: they ARE the eval set (99/100 exact match).

    python clockv2/build_vrefusal.py
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from common import (BAND, HELDOUT_DIR, PromptOnlyCapture, assert_readonly_hooks,
                    build_heldout, load_model, prompt_forward, save_probe)


def collect(model, tok, blocks, texts, device, label):
    """Mean hidden state per layer over a prompt set. Prompt-only, read-only."""
    cap = PromptOnlyCapture(blocks)
    assert_readonly_hooks(blocks)                     # guard BEFORE any forward
    acc_last = {L: torch.zeros(4096, dtype=torch.float64) for L in BAND}
    acc_mean = {L: torch.zeros(4096, dtype=torch.float64) for L in BAND}
    n = 0
    try:
        for i, t in enumerate(texts):
            prompt_forward(model, tok, t, cap, device=device)
            for L in BAND:
                acc_last[L] += cap.last[L].double()
                acc_mean[L] += cap.mean[L].double()
            n += 1
            if (i + 1) % 20 == 0:
                print(f"  [{label}] {i+1}/{len(texts)}", flush=True)
    finally:
        cap.close()
    return ({L: (acc_last[L] / n).float() for L in BAND},
            {L: (acc_mean[L] / n).float() for L in BAND}, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--j-thresh", type=float, default=0.4)
    args = ap.parse_args()

    ho = build_heldout(j_thresh=args.j_thresh)
    os.makedirs(HELDOUT_DIR, exist_ok=True)
    with open(os.path.join(HELDOUT_DIR, "contrastive.json"), "w") as f:
        json.dump(ho, f, indent=2)

    print(f"[vrefusal] loading model ...", flush=True)
    tok, model, blocks = load_model(device=args.device)

    print(f"[vrefusal] harmful  n={len(ho['harmful'])}", flush=True)
    mh_last, mh_mean, n_h = collect(model, tok, blocks, ho["harmful"], args.device, "harmful")
    print(f"[vrefusal] harmless n={len(ho['harmless'])}", flush=True)
    ml_last, ml_mean, n_l = collect(model, tok, blocks, ho["harmless"], args.device, "harmless")

    v_last, v_mean, report = {}, {}, []
    for L in BAND:
        d_last = (mh_last[L] - ml_last[L]).double()
        d_mean = (mh_mean[L] - ml_mean[L]).double()
        n_last, n_mean = d_last.norm().item(), d_mean.norm().item()
        v_last[L] = (d_last / max(n_last, 1e-9)).float()
        v_mean[L] = (d_mean / max(n_mean, 1e-9)).float()
        # separation sanity: cosine between the two pooling conventions, and the
        # raw gap size relative to the harmless mean's own norm (scale-free).
        cos_lm = torch.nn.functional.cosine_similarity(
            v_last[L].double(), v_mean[L].double(), dim=0).item()
        rel = n_last / max(ml_last[L].double().norm().item(), 1e-9)
        report.append((L, n_last, rel, cos_lm))

    save_probe("v_refusal.pt", {
        "kind": "refusal_direction_READ_ONLY_PROBE",
        "method": "normalize(mean_harmful - mean_harmless), Arditi Eq.1-2, prompt-only",
        "extraction_point": "last prompt token (primary); mean-over-prompt-tokens (aux)",
        "v_last": v_last, "v_mean": v_mean,
        "n_harmful": n_h, "n_harmless": n_l,
        "layers": BAND,
        "heldout": {k: ho[k] for k in ("j_thresh", "n_dropped", "max_j_kept")},
        "usage": "PROJECT ONLY. Never added to an activation. Never used to steer or "
                 "to reduce refusal.",
    })

    print("\n=== v_refusal sanity (read-only probe) ===")
    print(f"  harmful n={n_h}  harmless n={n_l}  (held-out, J<={args.j_thresh} vs eval)")
    print(f"  {'layer':>5} {'||mean_h - mean_l||':>20} {'rel-to-||mean_l||':>18} {'cos(last,mean)':>15}")
    for L, nl, rel, cos in report:
        print(f"  {L:5d} {nl:20.3f} {rel:18.4f} {cos:15.3f}")
    fin = all(torch.isfinite(v_last[L]).all().item() for L in BAND)
    print(f"\n  all finite: {fin}")
    print(f"  unit norm check L29: {v_last[29].norm().item():.6f} (expect 1.000000)")
    print("  NOTE: this artifact is a measurement axis only — it is never applied.")


if __name__ == "__main__":
    main()
