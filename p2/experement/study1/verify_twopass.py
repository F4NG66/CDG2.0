#!/usr/bin/env python3
"""study1/verify_twopass.py — PART F STEP 1: does enabling capture perturb the sample?

The two-pass straddler design only works if a seeded GENERATE-ONLY run and a
seeded CAPTURE run produce the SAME generation. Otherwise the straddling seeds
identified in pass 1 would not reproduce in pass 2 and the design collapses.

Risk being tested: at temperature > 0 the sampler draws Gumbel noise from the
global RNG. If the capture path consumes RNG differently from
dija_attack/cdg_denoise.denoise (extra draws, different tensor shape, different
call order), the streams desynchronise and the generations diverge.

Source-level expectation: `capture_union()` now calls the SAME `add_gumbel_noise`
(imported unedited from cdg_denoise) on the SAME full-logits tensor, once per
step, and nothing else in the capture path — softmax, topk, entropy, the
activation hooks — draws from the generator. So the streams should stay in
lockstep. This script checks it empirically, which is the only thing that
settles kernel-level nondeterminism.

Compares against the generate-only samples already on disk, byte for byte.

  python study1/verify_twopass.py --n 3
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
for p in (EXP, HERE, os.path.join(EXP, "dija_attack")):
    if p not in sys.path:
        sys.path.insert(0, p)

from capture_union import capture_union, _load_model, DEFAULT_LAYERS, MODEL_ID  # noqa: E402


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/scratch/ore99/study1_straddle/samples.jsonl")
    ap.add_argument("--refined", default=os.path.join(EXP, "dija_attack", "refined_100.json"))
    ap.add_argument("--n", type=int, default=3, help="how many (A-id, seed) pairs to re-run")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples) if l.strip()]
    cand = [r for r in rows if r["temperature"] == args.temp]
    # spread across A-ids rather than taking three seeds of one case
    picked, seen = [], set()
    for r in cand:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        picked.append(r)
        if len(picked) >= args.n:
            break
    assert picked, "no matching generate-only samples found"

    scaf = {r["id"]: r["Refined_behavior"] for r in json.load(open(args.refined))}

    print(f"[load] {MODEL_ID} ...", flush=True)
    tok, model, blocks = _load_model(args.device)
    from run_dija import PaperRunner  # noqa: E402 (imported unedited)
    runner = PaperRunner(model, tok, args.device)
    print(f"[load] {len(blocks)} blocks | capture layers={args.layers}\n", flush=True)

    # TWO comparisons, because the stored generate-only samples were produced on a
    # DIFFERENT physical GPU (the SLURM allocation moved nodes mid-study). Comparing a
    # capture here against a sample generated there would confound the question we care
    # about ("do the hooks perturb the RNG?") with hardware-level float nondeterminism.
    #   A. same-node generate-only vs same-node capture -> the DECISIVE hook test
    #   B. same-node generate-only vs the STORED sample -> cross-node reproducibility,
    #      which the two-pass design also depends on if the passes run on different nodes
    print(f"[node] {os.uname().nodename}  MIG={os.environ.get('CUDA_VISIBLE_DEVICES','?')}")
    print("[test] A = capture vs generate-only, SAME node (decisive)")
    print("[test] B = generate-only here vs stored sample from the earlier node\n")

    n_a = n_b = 0
    for r in picked:
        cid, seed = r["id"], r["seed"]

        # --- pass 1: generate-only, this node, this seed
        seed_all(seed)
        gen_resp, _ = runner.generate(scaf[cid], steps=128, gen_length=128,
                                      block_length=128, temperature=args.temp,
                                      remask="low_confidence", mask_id=126336)

        # --- pass 2: capture, this node, same seed
        ids, (t0, t1) = runner.build_inputs(scaf[cid])
        seed_all(seed)
        x, payload = capture_union(model, blocks, ids, args.layers,
                                   t_span=(t0, t1), temperature=args.temp)
        P = payload["meta"]["prompt_len"]
        tpl = tok.decode(x[0, payload["meta"]["t0"]:payload["meta"]["t1"]],
                         skip_special_tokens=True)
        cont = tok.decode(x[0, P:], skip_special_tokens=True)
        cap_resp = (tpl + "\n" + cont).strip()

        a_ok = cap_resp == gen_resp
        b_ok = gen_resp == r["response"]
        n_a += a_ok
        n_b += b_ok
        print(f"  {cid} seed={seed} T={args.temp}")
        print(f"    A capture vs generate-only (same node): "
              f"{'IDENTICAL' if a_ok else '*** DIVERGED ***'} "
              f"({len(cap_resp)} vs {len(gen_resp)} chars)")
        print(f"    B generate-only vs stored (other node) : "
              f"{'IDENTICAL' if b_ok else 'DIFFERS'} "
              f"({len(gen_resp)} vs {r['n_chars']} chars)")
        if not a_ok:
            i = next((k for k in range(min(len(cap_resp), len(gen_resp)))
                      if cap_resp[k] != gen_resp[k]), min(len(cap_resp), len(gen_resp)))
            print(f"      first difference at char {i}:")
            print(f"        generate-only: {gen_resp[max(0,i-60):i+60]!r}")
            print(f"        capture      : {cap_resp[max(0,i-60):i+60]!r}")

    print(f"\n  A (hook perturbation)   : {n_a}/{len(picked)} byte-identical")
    print(f"  B (cross-node stability): {n_b}/{len(picked)} byte-identical")
    if n_a == len(picked):
        print("\n  => TWO-PASS REPRODUCIBILITY HOLDS. Capture hooks do not perturb the")
        print("     sample; a straddling seed found in a generate-only pass reproduces")
        print("     under capture ON THE SAME NODE.")
        if n_b < len(picked):
            print("     CAVEAT: cross-node reproducibility FAILED, so both passes of any")
            print("     future capture study must run on the same physical GPU (or pass 1")
            print("     must be re-run on the capture node). This is a scheduling")
            print("     constraint, not a design failure.")
        else:
            print("     Cross-node reproducibility also holds.")
    else:
        print("\n  => TWO-PASS DESIGN FAILS: the hooks change the sample. STOP and report;")
        print("     do not proceed to a capture study on straddling seeds.")
    return 0 if n_a == len(picked) else 2


if __name__ == "__main__":
    sys.exit(main())
