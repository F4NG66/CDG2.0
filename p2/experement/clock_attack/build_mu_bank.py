#!/usr/bin/env python3
"""clock_attack/build_mu_bank.py — Step 0: build the τ mean-vector bank (Eq. 3).

Runs CLEAN (no-injection) LLaDA generations over a NEUTRAL, held-out seed set and
records the block-OUTPUT hidden state at the target layer(s) per denoising step,
using the EXISTING StepActivationCollector hook (out[0]) from ladaAndH.py — i.e. μ
is measured exactly where we will later steer.

    μ_{t,l} = mean over tokens (generated span) and examples, at step-bin t, layer l.

We also store a per-bin diagonal std (σ) as a covariance proxy for the Eq. 5
norm-matched RANDOM control. Saved per layer -> clock_attack/mu_bank_L{L}.pt.

NOT random noise: this is a content-pure direction measured on neutral content, so the
harmful cases never touch the direction (constraint). Default neutral set = the cdg
D_neutral_clean behaviors (100 unique benign items), held out from attack2/source_A.

Run (GPU):
    python clock_attack/build_mu_bank.py --n-seeds 32 --layers 29 25
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)                       # /home/ore99/experement
if EXP not in sys.path:
    sys.path.insert(0, EXP)

from ladaAndH import StepActivationCollector, generate_with_capture, discover_blocks  # noqa: E402

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
DEFAULT_NEUTRAL = "/home/ore99/serverFiles/prompts/cdg_injection/D_neutral_clean/cases.json"


def load_neutral_behaviors(path: str, n: int) -> list[str]:
    """Neutral, benign prompts only. Never the harmful cases."""
    obj = json.load(open(path))
    beh = [c["behavior"] for c in obj if c.get("behavior")]
    return beh[:n]


def bin_edges(total_steps: int, n_bins: int) -> list[list[int]]:
    """Group step indices 0..T-1 into n_bins contiguous bins (round-to-nearest)."""
    groups = [[] for _ in range(n_bins)]
    for i in range(total_steps):
        b = round(i / (total_steps - 1) * (n_bins - 1)) if total_steps > 1 else 0
        groups[b].append(i)
    # every bin is covered when total_steps >= n_bins (monotone step<=1); guard anyway
    for b in range(n_bins):
        if not groups[b]:
            groups[b] = groups[b - 1][-1:] if b > 0 else [0]
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral-seeds", default=DEFAULT_NEUTRAL,
                    help="JSON list with a 'behavior' field (benign/neutral held-out set)")
    ap.add_argument("--n-seeds", type=int, default=32, help="how many neutral seeds for μ")
    ap.add_argument("--layers", type=int, nargs="+", default=[29, 25],
                    help="block indices to build μ for (where we steer)")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--n-bins", type=int, default=101, help="denoising-progress bins (0..100)")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    behaviors = load_neutral_behaviors(args.neutral_seeds, args.n_seeds)
    print(f"[mu] neutral seeds: {len(behaviors)} from {args.neutral_seeds}", flush=True)
    assert behaviors, "no neutral behaviors loaded"

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.padding_side != "left":
        tok.padding_side = "left"
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16)
    model.all_tied_weights_keys = {}            # transformers-compat patch (see generate.py)
    model = model.to(args.device).eval()
    _, blocks = discover_blocks(model)
    n_layers = len(blocks)
    layer_ids = list(range(n_layers))
    print(f"[mu] blocks n={n_layers}; building μ for layers {args.layers}", flush=True)

    T = None
    # running float32 accumulators per target layer: sum and sumsq over examples of the
    # per-example, per-step mean-pooled hidden [T, d]
    acc_sum, acc_sq = {}, {}
    n_done = 0
    for k, beh in enumerate(behaviors):
        formatted = tok.apply_chat_template([{"role": "user", "content": beh}],
                                            add_generation_prompt=True, tokenize=False)
        enc = tok([formatted], add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"].to(args.device)
        attn = enc["attention_mask"].to(args.device)

        coll = StepActivationCollector(blocks, layer_ids, gen_start=ids.shape[1], pool="mean")
        try:
            generate_with_capture(model, ids, coll, attention_mask=attn,
                                  steps=args.steps, gen_length=args.gen_length,
                                  block_length=args.block_length, temperature=0.0,
                                  mask_id=MASK_ID)
        except torch.cuda.OutOfMemoryError:
            coll.close()
            sys.exit("[OOM] CUDA out of memory on MIG slice — switch to a full H100 "
                     "(salloc --gres=gpu:nvidia_h100_80gb_hbm3:1) and rerun.")
        h = coll.stack()                        # [T, n_layers, d] fp16
        coll.close()
        if T is None:
            T = h.shape[0]
            d = h.shape[-1]
            for L in args.layers:
                acc_sum[L] = torch.zeros(T, d, dtype=torch.float32)
                acc_sq[L] = torch.zeros(T, d, dtype=torch.float32)
        for L in args.layers:
            v = h[:, L, :].float()              # [T, d]
            acc_sum[L] += v
            acc_sq[L] += v * v
        n_done += 1
        if n_done % 8 == 0 or n_done == len(behaviors):
            print(f"[mu] captured {n_done}/{len(behaviors)}", flush=True)

    groups = bin_edges(T, args.n_bins)
    os.makedirs(args.out_dir, exist_ok=True)
    for L in args.layers:
        mu_step = acc_sum[L] / n_done                                  # [T, d]
        var = (acc_sq[L] / n_done - mu_step * mu_step).clamp_min(0.0)
        sig_step = var.sqrt()                                          # [T, d]
        mu_bin = torch.stack([mu_step[g].mean(0) for g in groups])     # [n_bins, d]
        sig_bin = torch.stack([sig_step[g].mean(0) for g in groups])   # [n_bins, d]
        path = os.path.join(args.out_dir, f"mu_bank_L{L}.pt")
        torch.save({
            "layer": L, "total_steps": int(T), "n_bins": int(args.n_bins),
            "mu_step": mu_step.half(), "mu_bin": mu_bin.half(), "sigma_bin": sig_bin.half(),
            "n_examples": int(n_done),
            "meta": {"neutral_seeds": args.neutral_seeds, "n_seeds": n_done,
                     "steps": args.steps, "gen_length": args.gen_length,
                     "block_length": args.block_length, "model": MODEL_ID,
                     "pool": "mean", "captured_at": "block_output(out[0])"},
        }, path)
        dnorm = (mu_bin[-1] - mu_bin[T // 4 if False else args.n_bins // 5]).norm().item()
        print(f"[mu] saved {path}  mu_bin{list(mu_bin.shape)}  "
              f"||μ[100]-μ[20]||={dnorm:.2f}", flush=True)

    print("[mu] done.", flush=True)


if __name__ == "__main__":
    main()
