#!/usr/bin/env python3
"""clockv2/capture_seed.py — SEED-ROBUSTNESS re-run of the FINAL CONTRAST only (read-only).

Scope, deliberately narrow (per instruction):
  * ONLY the two arms of the final contrast: dija and benign_op. clean and benign (arm 2)
    are NOT re-run.
  * ONLY L25-L27, the band where the content component lives and the null has collapsed.
  * ONLY v_injection_svd (headline) and v_random_null (floor). v_refusal is NOT projected
    here at all -- REPORT.md §8 Q1 stays logged and not run, by construction rather than by
    convention.
  * Read points: prompt_mean (PRIMARY, the headline) and last_prompt (SECONDARY). tau is
    not touched (NULL/circular, not re-litigated), so resp_mean is not projected.

WHY temperature>0 RATHER THAN EXTRA SEEDS AT temp=0:
  capture_denoise at temperature==0 takes argmax over raw logits (common.py:318-324) --
  the trajectory is a deterministic function of the input. Re-running it under a different
  torch seed would re-measure GPU float non-determinism, not run-to-run stability, and
  would answer a question nobody asked. The stochasticity in this engine enters at
  temperature>0 via Gumbel noise (torch.rand_like, common.py:321-323), which perturbs WHICH
  tokens get committed at each step. Under LLaDA's bidirectional attention the prompt span
  attends to the filling canvas, so a different fill trajectory yields genuinely different
  prompt_mean hidden states -- exactly the variance the single-seed limitation was about.

  temperature=0.2 is the DIJA paper's own config (run_dija.py:159), which this study had
  deviated from for determinism, so this doubles as an ecological check: the finding is
  re-tested under the setting the attack is actually specified at.

NO ACTIVATION IS EVER MODIFIED. assert_readonly_hooks() runs before every forward.

    python clockv2/capture_seed.py --cond dija      --seed 1 --limit 2   # smoke
    python clockv2/capture_seed.py --cond benign_op --seed 1 --limit 0   # full 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from capture import load_probes
from common import (CASES_A, HELDOUT_DIR, REFINED_100, RESULTS_DIR, STEPS, StepCapture,
                    assert_readonly_hooks, capture_denoise, load_model, EXP)

sys.path.insert(0, os.path.join(EXP, "dija_attack"))
from run_dija import PaperRunner  # noqa: E402

SEED_BAND = [25, 26, 27]


def load_arm(cond):
    """-> {id: prompt_string}. NOTE the two engine paths differ, deliberately:
    dija/benign_op are scaffolds fed through PaperRunner.build_inputs (<<TPL>> wrapper +
    <mask:N> expansion); clean is the bare behavior fed through the plain chat template
    with NO wrapper and no blanks. That asymmetry IS the clean arm's definition, and it
    mirrors capture.run_condition() exactly so seeds 1-2 stay comparable to the seed-0
    record. See build_inputs() branch in main().
    """
    if cond == "dija":
        return {r["id"]: r["Refined_behavior"] for r in json.load(open(REFINED_100))}
    if cond == "benign_op":
        p = os.path.join(HELDOUT_DIR, "benign_op_matched.json")
        return {r["id"]: r["benign_scaffold"] for r in json.load(open(p))}
    if cond == "benign":
        # arm 2: dataset-sourced INFORMATIONAL benign. Same wrapper/engine path as arm 3;
        # only the question register differs. Carries the two-arm-agreement argument.
        p = os.path.join(HELDOUT_DIR, "benign_matched.json")
        return {r["id"]: r["benign_scaffold"] for r in json.load(open(p))}
    if cond == "clean":
        return {c["id"]: c["behavior"] for c in json.load(open(CASES_A))}
    raise SystemExit(f"capture_seed.py: unsupported cond={cond!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cond", required=True, choices=["dija", "benign_op", "benign", "clean"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(RESULTS_DIR, f"probe_readings_s{args.seed}_{args.cond}.jsonl")
    smoke = args.limit and args.limit <= 2
    assert args.temperature > 0, "temp=0 is deterministic; a seed sweep there measures nothing"

    scaf = load_arm(args.cond)
    ids = sorted(scaf)
    assert len(ids) == 100, f"expected 100 scaffolds, got {len(ids)}"
    if args.limit:
        ids = ids[:args.limit]

    _, _, vinj, vnul = load_probes()          # v_refusal deliberately not loaded
    print(f"[s{args.seed}/{args.cond}] loading model ...", flush=True)
    tok, model, blocks = load_model(device=args.device)
    runner = PaperRunner(model, tok, args.device)

    done = set()
    if not smoke and not args.overwrite and os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    fout = None if smoke else open(args.out, "w" if args.overwrite else "a", encoding="utf-8")

    for cid in ids:
        if cid in done:
            continue
        if args.cond == "clean":
            # byte-for-byte the capture.run_condition() clean path: bare behavior, plain
            # chat template, NO <<TPL>> wrapper, no injected blanks (n_inject must be 0)
            s = tok.apply_chat_template([{"role": "user", "content": scaf[cid]}],
                                        add_generation_prompt=True, tokenize=False)
            inp = tok([s], add_special_tokens=False, return_tensors="pt")["input_ids"].to(args.device)
        else:
            inp, _ = runner.build_inputs(scaf[cid])
        cap = StepCapture(blocks, SEED_BAND, gen_start=inp.shape[1])
        assert_readonly_hooks(blocks)                     # guard BEFORE any forward
        # trajectory is a reproducible function of (seed, id); distinct across both
        torch.manual_seed(args.seed * 100003 + int(cid[1:]))
        try:
            _, info = capture_denoise(model, inp, cap, steps=STEPS,
                                      temperature=args.temperature)
            H = {"pmean": {L: torch.stack([s[L] for s in cap.prompt_mean]) for L in SEED_BAND},
                 "last": {L: torch.stack([s[L] for s in cap.last_prompt]) for L in SEED_BAND}}
        finally:
            cap.close()

        for L in SEED_BAND:
            proj = {}
            for rp in ("pmean", "last"):
                Hl = H[rp][L].double()
                proj[f"vshared_{rp}"] = Hl @ vinj["v"]["mean" if rp != "last" else "last"][L].double()
                proj[f"vnull_{rp}"] = Hl @ vnul["v"].double()
            for t in range(STEPS):
                row = {"id": cid, "cond": args.cond, "seed": args.seed,
                       "temperature": args.temperature, "layer": L, "step": t,
                       "n_inject": info["n_inject"], "prompt_len": inp.shape[1]}
                for k, v in proj.items():
                    row[k] = round(float(v[t]), 4)
                if fout:
                    fout.write(json.dumps(row) + "\n")
        if fout:
            fout.flush()
        print(f"[s{args.seed}/{args.cond}] {cid}: n_inject={info['n_inject']} "
              f"plen={inp.shape[1]} temp={args.temperature}", flush=True)
    if fout:
        fout.close()
        print(f"[s{args.seed}/{args.cond}] wrote {args.out}")


if __name__ == "__main__":
    main()
