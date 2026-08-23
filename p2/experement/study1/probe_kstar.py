#!/usr/bin/env python3
"""study1/probe_kstar.py — smoke diagnostic (NOT scaling): characterise the k*
window on ONE case. Answers: is the late-k* failure step-budget STARVATION or
genuine harm-return? Re-masks the SCAFFOLD-BLANK worksheet (the real harm site,
avoiding the refusal-tail the span_locator sweeps in) and a matched control, at a
sweep of k*, temp=0. Prints text + a degeneration score per k*."""
from __future__ import annotations
import argparse
import os
import re
import sys

import torch

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIJA = os.path.join(EXP, "dija_attack")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (EXP, DIJA, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_dija import PaperRunner, MODEL_ID, MASK_ID  # noqa: E402
from remask_denoise import build_canvas, run_denoise, compose_response, harm_positions  # noqa: E402

STEPS, GEN = 128, 128
DUP = re.compile(r"\b(\w{3,})\b(?:\s+\1\b)+", re.IGNORECASE)


def degen(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    uniq = len(set(ids)) / max(1, len(ids))
    dup_runs = len(DUP.findall(s))
    return round(1 - uniq, 3), dup_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="A000")
    ap.add_argument("--kstars", default="63,80,95,104,110,118,124,127")
    ap.add_argument("--refined", default=os.path.join(DIJA, "refined_100.json"))
    args = ap.parse_args()
    import json
    row = {r["id"]: r for r in json.load(open(args.refined))}[args.id]
    scaffold = row["Refined_behavior"]
    behavior = row.get("behavior", "")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(dev).eval()
    runner = PaperRunner(model, tok, dev)
    print(f"[probe] {args.id} loaded", flush=True)

    # baseline -> positions + commit
    x, attn, (t0, t1), P, total, f0 = build_canvas(runner, scaffold, gen_length=GEN, mask_id=MASK_ID)
    x, commit, _ = run_denoise(runner, x, attn, steps=STEPS, mask_id=MASK_ID,
                               temperature=0.0, remask="low_confidence", intervention=None)
    hp = harm_positions(runner, x, t0, t1, P, total, f0, behavior)
    scaffold_pos = hp["scaffold_canvas"]        # the worksheet harm site (no refusal tail)
    spanset = set(hp["span_canvas"])
    cont = hp["cont_canvas"]
    sc_commits = sorted(int(commit[g]) for g in scaffold_pos if commit[g] >= 0)
    import statistics as st
    print(f"[probe] scaffold blanks={len(scaffold_pos)}  commit "
          f"min/med/p90/max={sc_commits[0]}/{st.median(sc_commits)}/"
          f"{sc_commits[int(0.9*len(sc_commits))]}/{sc_commits[-1]}")
    print(f"[probe] BASELINE degen={degen(tok, hp['response'])}\n{hp['response']}\n" + "=" * 90)

    # matched control: trailing non-harm positions, count = |scaffold_pos|
    pool = [g for g in cont if g not in spanset]
    ctrl = sorted(pool)[-len(scaffold_pos):]

    for kstar in [int(k) for k in args.kstars.split(",")]:
        for label, pos in [("A=worksheet", scaffold_pos), ("C=control", ctrl)]:
            x2, a2, _, _, _, _ = build_canvas(runner, scaffold, gen_length=GEN, mask_id=MASK_ID)
            x2, _, log = run_denoise(runner, x2, a2, steps=STEPS, mask_id=MASK_ID,
                                     temperature=0.0, remask="low_confidence",
                                     intervention={"kstar": kstar, "positions": pos,
                                                   "seed_post": None, "post_temp": 0.0})
            resp = compose_response(runner, x2, t0, t1, P)
            dr, du = degen(tok, resp)
            print(f"\n### k*={kstar:3} rem={log['remaining_steps']:3} {label:12} "
                  f"wiped={log['n_wiped_committed']}/{log['n_remask']} "
                  f"maxfill/step={max(log['refill_fills_per_step'])} degen={dr} dupruns={du}")
            print(resp)


if __name__ == "__main__":
    main()
