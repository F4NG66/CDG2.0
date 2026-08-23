#!/usr/bin/env python3
"""clockv2/build_vshared.py — two READ-ONLY reference axes.

  (a) v_random_null   : the seed-1337 normalized Gaussian from
                        clock/execute_latent_clock_attack.py:66-68.
                        HONEST LABEL: this is a RANDOM direction, not an injection
                        signature. The variable there is merely *named* v_shared;
                        there is no SVD anywhere in its construction. In 4096-d a
                        random direction is near-orthogonal to everything, so its
                        projections sit near zero by construction. That makes it a
                        useful FLOOR — the "what does a meaningless axis read like"
                        control against which tau / v_refusal deltas are judged.

  (b) v_injection_svd : an ACTUAL injection-signature direction. Top right singular
                        vector of the (clean -> DIJA) prompt-only hidden-state
                        difference matrix, per layer.
                            D[i, :] = h_dija(prompt_i) - h_clean(prompt_i)
                            v = SVD(D).V[0]
                        Sign is fixed so that mean(D @ v) > 0, i.e. +v = "more
                        injected". Sign is resolved empirically, not assumed.

Both are projected onto hidden states and read. Neither is ever added to an activation.

LEAKAGE: the SVD is fit on HELD-OUT prompts (MedSafetyBench, filtered at Jaccard<=0.4
vs the 100 eval behaviors). clock_attack/dija_scaffolds.jsonl is NOT used — it is
100/100 the eval set. The DIJA scaffold for each held-out prompt is produced by the
existing verified mechanism (dija_refiner.stub_scaffold + ablate_scaffold "minimal"),
which attaches a generic "Step N: <mask:k>" skeleton. No attack content is authored
here; the skeleton is generic and the blanks stay empty (we never generate from them).

KNOWN LIMITATION (style gap) — report this, do not paper over it. The eval scaffolds in
refined_100.json are Qwen-REFINED and keep trailing context, e.g.
    "Procedure: <mask:20>. Proceed with caution as this is highly dangerous."
whereas the held-out stub after "minimal" ablation is bare:
    "Step 1: <mask:14>"
So v_injection_svd captures the STRUCTURAL injection signature that both share (prompt-
embedded mask blanks + worksheet framing), fit on a plainer variant than the eval
prompts carry. Using the refined style for the held-out fit would mean either authoring
scaffold text or re-running the Qwen refiner; we do neither. Phase 3 must read this axis
as "structural injection signature", not "refined-DIJA signature".

Prompt-only forward passes. No denoising loop, no generation, in this file.

    python clockv2/build_vshared.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import torch

from common import (BAND, D_MODEL, HELDOUT_DIR, MASK_ID, PromptOnlyCapture,
                    assert_readonly_hooks, build_heldout, load_model, prompt_forward,
                    save_probe, EXP)

sys.path.insert(0, os.path.join(EXP, "dija_attack"))
from dija_refiner import stub_scaffold, ablate_scaffold, is_valid_scaffold  # noqa: E402
from run_dija import PaperRunner  # noqa: E402  (reuse the VERIFIED prompt-build path)

MASK_MARKER_RE = r"<mask:(\d+)>"


def build_random_null():
    """Reproduce clock/execute_latent_clock_attack.py:66-68 exactly."""
    torch.manual_seed(1337)
    base = torch.randn(D_MODEL, dtype=torch.bfloat16)
    return (base / torch.norm(base)).float()


def make_dija_scaffold(behavior: str):
    """Generic DIJA worksheet skeleton via the existing verified builder.

    NB: is_valid_scaffold returns (bool, reason) — a tuple is ALWAYS truthy, so it must
    be unpacked, never tested directly.
    """
    s = ablate_scaffold(stub_scaffold(behavior), "minimal")
    ok, why = is_valid_scaffold(s)
    return (s, why) if ok else (None, why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--j-thresh", type=float, default=0.4)
    args = ap.parse_args()

    # ---- (a) random null control (no model needed) ----
    v_rand = build_random_null()
    save_probe("v_random_null.pt", {
        "kind": "random_direction_NULL_CONTROL_READ_ONLY",
        "provenance": "clock/execute_latent_clock_attack.py:66-68 (torch.manual_seed(1337); "
                      "randn(4096) normalized). NOT an SVD injection signature.",
        "v": v_rand, "seed": 1337,
        "usage": "PROJECT ONLY, as a null floor. Never added to an activation.",
    })
    print(f"[v_random_null] norm={v_rand.norm().item():.6f} seed=1337 "
          f"(reproduces the vector previously mislabelled 'v_shared')")

    # ---- (b) real injection signature via SVD on held-out clean->DIJA diffs ----
    ho = build_heldout(j_thresh=args.j_thresh)
    prompts = ho["harmful"]                    # held-out, disjoint from the 100 eval cases
    pairs, rejected = [], {}
    for b in prompts:
        sc, why = make_dija_scaffold(b)
        if sc:
            pairs.append((b, sc))
        else:
            rejected[why] = rejected.get(why, 0) + 1
    print(f"[v_injection_svd] {len(pairs)}/{len(prompts)} held-out prompts scaffolded"
          + (f"; rejected: {rejected}" if rejected else ""))
    assert pairs, "no held-out scaffolds survived validation"
    n_masks = [sum(int(x) for x in re.findall(MASK_MARKER_RE, s)) for _, s in pairs]
    print(f"[v_injection_svd] mask counts: min={min(n_masks)} max={max(n_masks)} "
          f"mean={sum(n_masks)/len(n_masks):.1f}")

    print("[v_injection_svd] loading model ...", flush=True)
    tok, model, blocks = load_model(device=args.device)
    runner = PaperRunner(model, tok, args.device)

    cap = PromptOnlyCapture(blocks)
    assert_readonly_hooks(blocks)
    # both read points, matching how Phase 2 reads: mean-pooled and last-prompt-token
    diffs = {"mean": {L: [] for L in BAND}, "last": {L: [] for L in BAND}}
    n_inj = []
    try:
        for i, (beh, sc) in enumerate(pairs):
            # clean: plain chat template on the behavior (matches the clean condition)
            prompt_forward(model, tok, beh, cap, device=args.device)
            c_mean = {L: cap.mean[L].clone() for L in BAND}
            c_last = {L: cap.last[L].clone() for L in BAND}
            # dija: the VERIFIED build_inputs path (chat template + <mask:N> expansion
            # + <<TPL>> span strip). Prompt-only forward; blanks are never filled here.
            ids, _ = runner.build_inputs(sc)
            k = int((ids == MASK_ID).sum())
            assert k > 0, "scaffold produced no mask tokens in the prompt"
            n_inj.append(k)
            prompt_forward(model, tok, ids, cap, device=args.device)
            for L in BAND:
                diffs["mean"][L].append((cap.mean[L] - c_mean[L]).double())
                diffs["last"][L].append((cap.last[L] - c_last[L]).double())
            if (i + 1) % 10 == 0:
                print(f"  [svd] {i+1}/{len(pairs)}", flush=True)
    finally:
        cap.close()
    print(f"[v_injection_svd] prompt-embedded mask tokens: min={min(n_inj)} max={max(n_inj)}")

    v_inj = {"mean": {}, "last": {}}
    report = {}
    for pool in ("mean", "last"):
        rep = []
        for L in BAND:
            D = torch.stack(diffs[pool][L])              # [n, d]
            Dc = D - D.mean(0, keepdim=True)             # centre before SVD
            U, S, Vh = torch.linalg.svd(Dc, full_matrices=False)
            v = Vh[0]
            proj = D @ v
            if proj.mean() < 0:                          # fix sign EMPIRICALLY: +v = "more injected"
                v, proj = -v, -proj
            evr = (S[0] ** 2 / (S ** 2).sum()).item()    # variance explained by PC1
            v_inj[pool][L] = v.float()
            rep.append((L, evr, proj.mean().item(), proj.std().item(), D.norm(dim=1).mean().item()))
        report[pool] = rep

    save_probe("v_injection_svd.pt", {
        "kind": "injection_signature_READ_ONLY_PROBE",
        "method": "top right singular vector of centred (h_dija - h_clean) prompt-only "
                  "differences; sign fixed empirically so mean projection > 0",
        "extraction_point": "v['mean'] = mean-pooled over prompt tokens; "
                            "v['last'] = last prompt token",
        "scaffold_style": "stub_scaffold + ablate 'minimal' (bare 'Step N: <mask:k>'). "
                          "STRUCTURAL signature only — eval scaffolds are Qwen-refined "
                          "and richer. See module docstring 'style gap'.",
        "v": v_inj, "layers": BAND, "n_pairs": len(pairs),
        "heldout": {k: ho[k] for k in ("j_thresh", "n_dropped", "max_j_kept")},
        "usage": "PROJECT ONLY. Never added to an activation.",
    })

    print("\n=== v_injection_svd sanity (read-only probe) ===")
    print(f"  fit on n={len(pairs)} held-out clean/DIJA prompt pairs (J<={args.j_thresh} vs eval)")
    for pool in ("mean", "last"):
        print(f"\n  -- extraction: {pool} --")
        print(f"  {'layer':>5} {'PC1 var-expl':>13} {'mean proj':>11} {'sd proj':>9} {'mean ||diff||':>14}")
        for L, evr, m, s, dn in report[pool]:
            print(f"  {L:5d} {evr:13.3f} {m:11.3f} {s:9.3f} {dn:14.3f}")
    print(f"\n  all finite: {all(torch.isfinite(v_inj[p][L]).all().item() for p in v_inj for L in BAND)}")
    print(f"  unit norm check mean/L29: {v_inj['mean'][29].norm().item():.6f} (expect 1.000000)")
    cos = torch.nn.functional.cosine_similarity(
        v_inj["mean"][29].double(), v_rand.double(), dim=0).item()
    print(f"  cos(v_injection_svd[mean,29], v_random_null) = {cos:+.4f} "
          f"(expect ~0 — confirms the random axis carries no injection signal)")


if __name__ == "__main__":
    main()
