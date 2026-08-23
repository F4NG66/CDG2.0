#!/usr/bin/env python3
"""clockv2/capture.py — PHASE 2: read-only capture + projection.

For each of the 100 ids, run the model TWICE with the read-only capture hook:
  (1) clean : cases.json behavior via plain chat template
  (2) dija  : refined_100.json Refined_behavior via the VERIFIED run_dija.PaperRunner
              .build_inputs path (chat template + <mask:N> expansion + <<TPL>> strip)
Paired strictly by "id" (A000..A099). temp=0, steps=128, gen=128, unified fill_all_masks
schedule for BOTH conditions (see README: block_length is ignored on that path).

NO ACTIVATION IS EVER MODIFIED. assert_readonly_hooks() runs before every capture.
No judging, no harm scoring here — representation reading only.

READ POINTS (never mixed in one column; see README "cross-probe geometry"):
  tau_read     <- resp_mean    (the only valid read point: mu_step was fit there)
  *_pmean      <- prompt_mean  PRIMARY   (v_refusal/v_injection axes are near-orthogonal here)
  *_last       <- last_prompt  SECONDARY (axes are +0.65 aligned here — stated, not hidden)
  *_rmean      <- resp_mean    cross-position, exploratory only

Hidden states are projected ON THE FLY and only scalars are written: retaining raw
L16-L31 states would be ~430 GB over 200 runs.

    python clockv2/capture.py --limit 3     # SMOKE (prints readings, writes nothing)
    python clockv2/capture.py --limit 0     # full 100 x 2 (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from build_tau import decode_tau  # noqa: F401  (kept for the scalar/batched equivalence check)
from common import (BAND, CASES_A, MASK_ID, PROBE_DIR, REFINED_100, RESULTS_DIR, STEPS,
                    StepCapture, assert_readonly_hooks, capture_denoise, load_model, EXP)

sys.path.insert(0, os.path.join(EXP, "dija_attack"))
from run_dija import PaperRunner  # noqa: E402


def load_probes():
    p = lambda f: torch.load(os.path.join(PROBE_DIR, f), map_location="cpu", weights_only=False)
    tau = p("tau_bank.pt")
    vref = p("v_refusal.pt")
    vinj = p("v_injection_svd.pt")
    vnul = p("v_random_null.pt")
    return tau, vref, vinj, vnul


def decode_tau_batched(H, mu_step, t_temp=1.0):
    """Batched equivalent of build_tau.decode_tau. H:[T,d] mu_step:[T,d] -> ([T],[T])."""
    d = torch.cdist(H.double(), mu_step.double())                 # [T, T]
    T = mu_step.shape[0]
    grid = torch.arange(T, dtype=torch.float64) / (T - 1)
    w = torch.softmax(-d / (d.std(dim=1, keepdim=True).clamp_min(1e-6) * t_temp), dim=1)
    return (w * grid).sum(1), grid[d.argmin(1)]


def run_condition(model, tok, blocks, runner, row_clean, row_dija, cond, device):
    """One read-only capture. Returns per-step stacked hiddens + physical canvas info."""
    if cond == "clean":
        s = tok.apply_chat_template([{"role": "user", "content": row_clean["behavior"]}],
                                    add_generation_prompt=True, tokenize=False)
        ids = tok([s], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    else:
        ids, _ = runner.build_inputs(row_dija["Refined_behavior"])
    cap = StepCapture(blocks, BAND, gen_start=ids.shape[1])
    assert_readonly_hooks(blocks)                     # guard BEFORE any forward
    try:
        _, info = capture_denoise(model, ids, cap, steps=STEPS)
        H = {
            "rmean": {L: torch.stack([st[L] for st in cap.resp_mean]) for L in BAND},
            "pmean": {L: torch.stack([st[L] for st in cap.prompt_mean]) for L in BAND},
            "last": {L: torch.stack([st[L] for st in cap.last_prompt]) for L in BAND},
        }
    finally:
        cap.close()
    return H, info, ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "probe_readings.jsonl"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    smoke = args.limit and args.limit <= 3

    clean_rows = {c["id"]: c for c in json.load(open(CASES_A))}
    dija_rows = {c["id"]: c for c in json.load(open(REFINED_100))}
    ids = sorted(set(clean_rows) & set(dija_rows))
    assert len(ids) == 100, f"expected 100 paired ids, got {len(ids)}"
    if args.limit:
        ids = ids[:args.limit]

    tau, vref, vinj, vnul = load_probes()
    print(f"[cap] probes: tau_bank(n_fit={tau['n_fit']}) v_refusal(n_h={vref['n_harmful']}) "
          f"v_injection_svd(n={vinj['n_pairs']}) v_random_null", flush=True)

    print("[cap] loading model ...", flush=True)
    tok, model, blocks = load_model(device=args.device)
    runner = PaperRunner(model, tok, args.device)

    done = set()
    if not smoke and not args.overwrite and os.path.exists(args.out):
        for l in open(args.out):
            try:
                r = json.loads(l)
                done.add((r["id"], r["cond"]))
            except Exception:
                pass
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = None if smoke else open(args.out, "a" if not args.overwrite else "w",
                                   encoding="utf-8")

    smoke_dump = []
    for i, cid in enumerate(ids):
        for cond in ("clean", "dija"):
            if (cid, cond) in done:
                continue
            H, info, plen = run_condition(model, tok, blocks, runner,
                                          clean_rows[cid], dija_rows[cid], cond, args.device)
            mr = info["mask_ratio"]
            tfm = [1.0 - r for r in mr]
            for L in BAND:
                tau_soft, tau_nn = decode_tau_batched(H["rmean"][L], tau["mu_step"][L])
                proj = {}
                for rp, key in (("pmean", "pmean"), ("last", "last"), ("rmean", "rmean")):
                    Hl = H[rp][L].double()
                    vr = vref["v_mean" if rp != "last" else "v_last"][L].double()
                    vi = vinj["v"]["mean" if rp != "last" else "last"][L].double()
                    vn = vnul["v"].double()
                    proj[f"vrefusal_{key}"] = (Hl @ vr)
                    proj[f"vshared_{key}"] = (Hl @ vi)
                    proj[f"vnull_{key}"] = (Hl @ vn)
                for t in range(STEPS):
                    row = {
                        "id": cid, "cond": cond, "layer": L, "step": t,
                        "tau_read": round(float(tau_soft[t]), 6),
                        "tau_nn": round(float(tau_nn[t]), 6),
                        "mask_ratio_t": round(mr[t], 6),
                        "tau_from_mask": round(tfm[t], 6),
                        "mask_ratio_global": round(info["mask_ratio_global"][t], 6),
                        "n_inject": info["n_inject"],
                        "prompt_len": plen,
                    }
                    for k, v in proj.items():
                        row[k] = round(float(v[t]), 4)
                    if fout:
                        fout.write(json.dumps(row) + "\n")
                    if smoke and L in (25, 29) and t in (0, 32, 64, 96, 127):
                        smoke_dump.append(row)
            if fout:
                fout.flush()
            print(f"[cap] {cid}/{cond}: n_inject={info['n_inject']} plen={plen} "
                  f"mask_ratio t0={mr[0]:.3f} t127={mr[127]:.3f}", flush=True)
    if fout:
        fout.close()

    if smoke:
        print("\n" + "=" * 118)
        print("SMOKE — three probe readings vs physical canvas clock (READ-ONLY; nothing written)")
        print("read points: tau_read<-resp_mean | vrefusal/vshared/vnull <- prompt_mean (PRIMARY)")
        print("=" * 118)
        hdr = (f"{'id':>5} {'cond':>5} {'L':>3} {'step':>4} | {'mask_ratio_t':>12} "
               f"{'tau_from_mask':>13} {'tau_read':>8} {'disc_t':>7} | "
               f"{'vrefusal':>9} {'vshared':>9} {'vnull':>8}")
        for L in (25, 29):
            print(f"\n--- layer L{L} ---")
            print(hdr)
            for cid in ids:
                for cond in ("clean", "dija"):
                    for r in smoke_dump:
                        if r["id"] == cid and r["cond"] == cond and r["layer"] == L:
                            disc = r["tau_read"] - r["tau_from_mask"]
                            print(f"{r['id']:>5} {r['cond']:>5} {L:>3} {r['step']:>4} | "
                                  f"{r['mask_ratio_t']:>12.3f} {r['tau_from_mask']:>13.3f} "
                                  f"{r['tau_read']:>8.3f} {disc:>+7.3f} | "
                                  f"{r['vrefusal_pmean']:>9.2f} {r['vshared_pmean']:>9.2f} "
                                  f"{r['vnull_pmean']:>8.2f}")
        fin = all(all(torch.isfinite(torch.tensor(v)).item() for k, v in r.items()
                      if isinstance(v, float)) for r in smoke_dump)
        print(f"\n  all readings finite: {fin}")
        print(f"  rows that WOULD be written at --limit 0: 100 ids x 2 cond x "
              f"{len(BAND)} layers x {STEPS} steps = {100*2*len(BAND)*STEPS:,}")


if __name__ == "__main__":
    main()
