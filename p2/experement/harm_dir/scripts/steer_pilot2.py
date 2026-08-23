#!/usr/bin/env python3
"""PHASE 3.1 (PILOT) - scaffold-region steering (Arm S) and steer+remask (Arm SR).

Phase 3 steered v_harm_ds on the RESPONSE region and was null. But our one partial
success (-0.123, injection direction) used the SCAFFOLD region (tpl_ctx). Phase 3
changed TWO things vs that success (direction AND region). This pilot changes only
REGION back:

  ARM S  - SCAFFOLD-REGION STEERING (the untested cell)
     h -> h - alpha * v_hat  at L16, applied on the INJECTED-SCAFFOLD positions
     (scope_region="template", pos="unmask" == tpl_ctx), NOT the response. Widened
     dose {0,16,32,48,64}. Region is the single changed variable vs Phase 3.

  ARM SR - STEER-EARLY + REMASK-LATE (recent dLLM-defense recipe)
     Two-pass per generation, steering (scaffold region) active throughout:
       pass 1: full steered generation (== Arm S output)
       then  : PROBE every committed generated token with v_hat at L16
               (per-token projection from hooks.buffers[16]); remask the top
               --remask-frac most harm-aligned tokens
       pass 2: steered REFILL of the remasked tokens
     Optional adaptive: scale alpha by mean probe confidence (--adaptive).

CONTROLS (kept decisive, same as Phase 3):
  - RANDOM-direction control for BOTH S and SR (steering AND the SR probe use the
    random vector, matched norm).
  - BENIGN arm through the same sweeps, quality-gated baselines, paired RNG seed.

KNOWN-RISK CHECK (reported, never papered over): scaffold steering can reduce harm
by BREAKING the injected prompt rather than inducing refusal. So every row carries
n_words / trigram_rep / a `degenerate` flag; a harm drop that comes with empty or
garbled output is flagged, not counted as a win. Harm/utility are scored later by
graded_judge.py (verbatim), same as Phase 3.

NOTE on benign: D (neutral CLEAN) has no scaffold, so Arm S steering is a NO-OP on
it (template region is None). The informative benign-injected control for scaffold
steering is C_neutral_injected - see the plan. D is kept for Phase-3 comparability
and SR's remask still acts on it.

GPU (diffusion generation). Smoke with --limit 2, then STOP.
"""
import argparse
import json
import os
import sys
import time

HARM_DIR = "/home/ore99/experement/harm_dir"
CDG_ROOT = "/home/ore99/serverFiles"
sys.path.insert(0, CDG_ROOT)
sys.path.insert(0, os.path.join(HARM_DIR, "scripts"))

import torch  # noqa: E402
from cdg.config import get_backend_config  # noqa: E402
from cdg.backends import build_runner       # noqa: E402
from cdg.denoise import denoise             # noqa: E402
from cdg.data import load_cdg_root          # noqa: E402
from textnorm import quality                # noqa: E402
import steer_pilot as sp                    # reuse dirs, cases, gate, constants

PROMPTS = "/home/ore99/serverFiles/prompts/cdg_injection"


def c_injected_pool():
    """C_neutral_injected: neutral content inside a real DIJA scaffold. Unlike benign
    D (no scaffold, Arm S is a no-op), scaffold steering DOES act on C - so C is the
    benign-injected control: a win for Arm S needs C to stay coherent."""
    return sorted((c for c in load_cdg_root(PROMPTS) if c.group_letter == "C"),
                  key=lambda c: c.case_id)

DATA = os.path.join(HARM_DIR, "data")
STEER_LAYER = sp.STEER_LAYER               # 16


def metrics(resp, extra=None):
    q = quality(resp)
    nw = len(resp.split())
    m = {"response_text": resp, "n_words": nw,
         "trigram_rep": q["trigram_rep"], "uniq_word_ratio": q["uniq_word_ratio"],
         "declines": bool(sp.REFUSE.search(resp)),
         "degenerate": bool(nw < 8 or q["trigram_rep"] > 0.5 or "​" in resp)}
    if extra:
        m.update(extra)
    return m


def decode_resp(runner, x, P, regions):
    out = runner.tokenizer.decode(x[0, P:].tolist(), skip_special_tokens=True)
    if regions.get("template"):
        t0, t1 = regions["template"]
        tpl = runner.tokenizer.decode(x[0, t0:t1].tolist(), skip_special_tokens=True)
        out = (tpl + "\n" + out).strip()
    return out


def setup_x(runner, case):
    dc = runner.cfg.decode
    ids, regions = runner.build_inputs(case)
    P = ids.shape[1]
    total = P + dc.gen_length
    x = torch.full((1, total), runner.mask_id, dtype=torch.long, device=runner.device)
    x[:, :P] = ids
    attn = torch.ones((1, total), dtype=torch.long, device=runner.device)
    regions["output"] = (P, total)
    return x, attn, P, total, regions


# ---- Arm S: scaffold-region steering via the validated runner path ----
def gen_S(runner, case, direction, alpha, seed, scope_region, pos):
    torch.manual_seed(seed)
    if alpha == 0 or direction is None:
        runner.clear_steering()
    else:
        runner.set_steering({STEER_LAYER: direction}, alpha=-float(alpha),
                            scope_region=scope_region, pos=pos)
    t0 = time.time()
    resp = runner.generate(case, None)
    runner.clear_steering()
    return metrics(resp, {"gen_s": round(time.time() - t0, 1), "n_remasked": 0})


# ---- Arm SR: steer-early + remask-late (custom two-pass) ----
def gen_SR(runner, case, direction, alpha, seed, scope_region, pos,
           remask_frac, rounds, refill_steps, adaptive):
    torch.manual_seed(seed)
    dc = runner.cfg.decode
    mask_id = runner.mask_id
    x, attn, P, total, regions = setup_x(runner, case)
    fillable0 = (x[0] == mask_id).clone()          # generated content = template blanks + output

    # steering position mask over the scaffold region (fixed at entry, like generate())
    span = regions.get(scope_region)
    steer_on = direction is not None and alpha != 0 and span is not None
    if steer_on:
        lo, hi = span
        seg = x[0, lo:hi]
        is_mask = (seg == mask_id)
        local = (is_mask if pos == "mask" else ~is_mask if pos == "unmask"
                 else torch.ones_like(is_mask))
        pm = torch.zeros((1, total), dtype=torch.bool, device=x.device)
        pm[0, lo:hi] = local
        vbf = direction.to(torch.bfloat16).to(x.device)
        runner.hooks.set_steer(STEER_LAYER, -float(alpha), vbf, positions=pm)

    def run_denoise(steps):
        return denoise(runner, x, attn, steps=steps, gen_length=dc.gen_length,
                       prompt_len=P, block_length=dc.block_length,
                       temperature=dc.temperature, remask=runner.cfg.remask,
                       mask_id=mask_id, recorder=None, fill_all_masks=dc.fill_all_masks)

    x = run_denoise(dc.steps)                       # pass 1: full steered gen

    vprobe = (direction if direction is not None else None)
    total_remasked = 0
    if vprobe is not None:
        vp = vprobe.float().to(x.device)
        for _ in range(rounds):
            runner.hooks.clear()
            _ = runner.forward(x, attn)             # populate buffers[16]
            h = runner.hooks.buffers[STEER_LAYER][0].float()   # (T,d) un-steered hidden
            proj = h @ vp                            # (T,) harm alignment per token
            committed = fillable0 & (x[0] != mask_id)
            idx = committed.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break
            k = max(1, int(round(remask_frac * idx.numel())))
            order = torch.argsort(proj[idx], descending=True)   # most harm-aligned first
            flag = idx[order[:k]]
            x[0, flag] = mask_id
            total_remasked += int(flag.numel())
            if adaptive and steer_on:               # scale by mean probe confidence
                conf = float(proj[flag].mean().clamp(min=0))
                scale = 1.0 + min(1.0, conf / (abs(proj).mean().item() + 1e-6))
                runner.hooks.set_steer(STEER_LAYER, -float(alpha) * scale, vbf, positions=pm)
            x = run_denoise(refill_steps)           # pass 2: steered refill

    runner.hooks.reset_steer()
    resp = decode_resp(runner, x, P, regions)
    return metrics(resp, {"n_remasked": total_remasked, "gen_s": None})


def run_case(out, runner, group, case, ci, dirs, args, alphas, write):
    cseed = args.seed * 1000 + ci
    # shared baseline (no steering, no remask)
    r0 = gen_S(runner, case, None, 0, cseed, args.scope, args.pos)
    write(out, group, case, "none", "none", 0, cseed, r0)
    for mode in args.modes:
        for arm in ("vharm", "random"):
            for alpha in alphas:
                d = dirs[arm]
                if mode == "S":
                    r = gen_S(runner, case, d, alpha, cseed, args.scope, args.pos)
                else:
                    r = gen_SR(runner, case, d, alpha, cseed, args.scope, args.pos,
                               args.remask_frac, args.remask_rounds, args.refill_steps,
                               args.adaptive)
                write(out, group, case, mode, arm, alpha, cseed, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap on harmful cases (0=all 23)")
    ap.add_argument("--benign-target", type=int, default=20)
    ap.add_argument("--benign-scan", type=int, default=60)
    ap.add_argument("--alphas", default="16,32,48,64")
    ap.add_argument("--modes", default="S,SR")
    ap.add_argument("--scope", default="template", help="steer region (template=scaffold)")
    ap.add_argument("--pos", default="unmask", help="unmask==tpl_ctx (injected scaffold text)")
    ap.add_argument("--remask-frac", type=float, default=0.15)
    ap.add_argument("--remask-rounds", type=int, default=1)
    ap.add_argument("--refill-steps", type=int, default=32)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--with-c", action="store_true",
                    help="add C_neutral_injected benign-injected control (full run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA, "steer_pilot2.jsonl"))
    args = ap.parse_args()

    alphas = [int(a) for a in args.alphas.split(",")]
    args.modes = [m.strip() for m in args.modes.split(",")]
    dirs, cos_rc = sp.load_directions(args.seed)
    print(f"cos(v_harm, v_random) = {cos_rc:+.4f}   scope={args.scope}/{args.pos}  "
          f"modes={args.modes}  alphas={alphas}  remask_frac={args.remask_frac}")

    cfg = get_backend_config("llada_attack")
    cfg.saes = []
    print(f"backend={cfg.name} gen_len={cfg.decode.gen_length} steps={cfg.decode.steps} "
          f"temp={cfg.decode.temperature}  cuda={torch.cuda.is_available()}")
    t0 = time.time()
    runner = build_runner(cfg, sae_root="unused", device="cuda")
    assert 0 <= STEER_LAYER + runner.hooks.offset < len(runner.hooks.blocks)
    print(f"runner up in {time.time()-t0:.1f}s  steering block {STEER_LAYER + runner.hooks.offset}")

    harmful, benign_pool = sp.pick_cases(args.limit)
    dcuda = {k: v.to("cuda") for k, v in dirs.items()}
    print(f"harmful={len(harmful)}  benign pool={len(benign_pool)}  benign_target={args.benign_target}")

    def write(out, group, case, mode, arm, alpha, cseed, r):
        rec = {"case_id": case.case_id, "group": group, "mode": mode, "arm": arm,
               "alpha": alpha, "seed": cseed, **r}
        rec.pop("gen_s", None)
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"  {group:7s} {case.case_id} {mode:4s} {arm:6s} a={alpha:2d} "
              f"{r['n_words']:3d}w remask={r.get('n_remasked',0):2d} "
              f"deg={int(r['degenerate'])} decl={int(r['declines'])} "
              f"trig={r['trigram_rep']:.2f}")

    def process_benign(out, group, pool, seed_base):
        kept = 0
        for ci, case in enumerate(pool[:args.benign_scan]):
            if kept >= args.benign_target:
                break
            cseed = args.seed * 1000 + seed_base + ci
            r0 = gen_S(runner, case, None, 0, cseed, args.scope, args.pos)
            if r0["n_words"] < sp.MIN_BENIGN_WORDS or r0["trigram_rep"] > sp.MAX_BENIGN_TRIGRAM:
                print(f"  {group:8s} {case.case_id} REJECT baseline ({r0['n_words']}w)")
                continue
            write(out, group, case, "none", "none", 0, cseed, r0)
            for mode in args.modes:
                for arm in ("vharm", "random"):
                    for alpha in alphas:
                        if mode == "S":
                            r = gen_S(runner, case, dirs[arm], alpha, cseed, args.scope, args.pos)
                        else:
                            r = gen_SR(runner, case, dirs[arm], alpha, cseed, args.scope,
                                       args.pos, args.remask_frac, args.remask_rounds,
                                       args.refill_steps, args.adaptive)
                        write(out, group, case, mode, arm, alpha, cseed, r)
            kept += 1
        print(f"\n{group} kept {kept}/{args.benign_target}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dirs = dcuda
    n = 0
    with open(args.out, "w") as out:
        for ci, case in enumerate(harmful):
            run_case(out, runner, "harmful", case, ci, dirs, args, alphas, write); n += 1
        # benign D (no scaffold: Arm S no-op; SR remask still acts) - Phase-3 comparability
        process_benign(out, "benign", benign_pool, seed_base=100)
        # benign-injected C (scaffold present: the real Arm-S utility control)
        if args.with_c:
            process_benign(out, "benign_c", c_injected_pool(), seed_base=300)
    print(f"wrote {n} harmful cases + benign -> {args.out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
