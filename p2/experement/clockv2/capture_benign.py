#!/usr/bin/env python3
"""clockv2/capture_benign.py — MATCHED-CONTROL capture (read-only).

Third arm: the length- and format-matched BENIGN scaffold from build_benign.py, pushed
through the SAME engine path as the DIJA arm (run_dija.PaperRunner.build_inputs -> <<TPL>>
wrapper -> <mask:N> expansion -> chat template), at temp=0, steps=128, gen=128, unified
fill_all_masks. Writes cond="benign" rows to results/probe_readings_benign.jsonl with the
identical schema to probe_readings.jsonl, so the two files concatenate.

NO ACTIVATION IS EVER MODIFIED. assert_readonly_hooks() runs before every forward.

Read points are unchanged and never mixed:
  tau_read/tau_nn <- resp_mean ; *_pmean <- prompt_mean (PRIMARY) ;
  *_last <- last_prompt (SECONDARY, axes +0.65 aligned there) ; *_rmean <- resp_mean (expl.)

    python clockv2/capture_benign.py --limit 3    # smoke, writes nothing
    python clockv2/capture_benign.py --limit 0    # full 100 (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from capture import decode_tau_batched, load_probes
from common import (BAND, HELDOUT_DIR, PROBE_DIR, RESULTS_DIR, STEPS, StepCapture,
                    assert_readonly_hooks, capture_denoise, load_model, EXP)

sys.path.insert(0, os.path.join(EXP, "dija_attack"))
from run_dija import PaperRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--input", default="benign_matched.json",
                    help="heldout/ file: benign_matched.json (arm 2, informational) or "
                         "benign_op_matched.json (arm 3, operational first-person)")
    ap.add_argument("--cond", default="benign",
                    help="cond label written to each row: 'benign' or 'benign_op'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(RESULTS_DIR, f"probe_readings_{args.cond}.jsonl")
    smoke = args.limit and args.limit <= 3

    rows = {r["id"]: r for r in json.load(open(os.path.join(HELDOUT_DIR, args.input)))}
    ids = sorted(rows)
    assert len(ids) == 100, f"expected 100 benign scaffolds, got {len(ids)}"
    if args.limit:
        ids = ids[:args.limit]

    tau, vref, vinj, vnul = load_probes()
    print("[ben] loading model ...", flush=True)
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
        inp, _ = runner.build_inputs(rows[cid]["benign_scaffold"])
        cap = StepCapture(blocks, BAND, gen_start=inp.shape[1])
        assert_readonly_hooks(blocks)                    # guard BEFORE any forward
        try:
            _, info = capture_denoise(model, inp, cap, steps=STEPS)
            H = {"rmean": {L: torch.stack([s[L] for s in cap.resp_mean]) for L in BAND},
                 "pmean": {L: torch.stack([s[L] for s in cap.prompt_mean]) for L in BAND},
                 "last": {L: torch.stack([s[L] for s in cap.last_prompt]) for L in BAND}}
        finally:
            cap.close()

        mr = info["mask_ratio"]
        for L in BAND:
            tau_soft, tau_nn = decode_tau_batched(H["rmean"][L], tau["mu_step"][L])
            proj = {}
            for rp in ("pmean", "last", "rmean"):
                Hl = H[rp][L].double()
                proj[f"vrefusal_{rp}"] = Hl @ vref["v_mean" if rp != "last" else "v_last"][L].double()
                proj[f"vshared_{rp}"] = Hl @ vinj["v"]["mean" if rp != "last" else "last"][L].double()
                proj[f"vnull_{rp}"] = Hl @ vnul["v"].double()
            for t in range(STEPS):
                row = {"id": cid, "cond": args.cond, "layer": L, "step": t,
                       "tau_read": round(float(tau_soft[t]), 6),
                       "tau_nn": round(float(tau_nn[t]), 6),
                       "mask_ratio_t": round(mr[t], 6),
                       "tau_from_mask": round(1.0 - mr[t], 6),
                       "mask_ratio_global": round(info["mask_ratio_global"][t], 6),
                       "n_inject": info["n_inject"], "prompt_len": inp.shape[1]}
                for k, v in proj.items():
                    row[k] = round(float(v[t]), 4)
                if fout:
                    fout.write(json.dumps(row) + "\n")
        if fout:
            fout.flush()
        exp = rows[cid]["n_inject"]
        flag = "" if info["n_inject"] == exp else f"  !! n_inject {info['n_inject']} != expected {exp}"
        print(f"[ben] {cid}: n_inject={info['n_inject']} plen={inp.shape[1]} "
              f"mask_ratio t0={mr[0]:.3f} t127={mr[127]:.3f}{flag}", flush=True)
    if fout:
        fout.close()
        print(f"[ben] wrote {args.out}")


if __name__ == "__main__":
    main()
