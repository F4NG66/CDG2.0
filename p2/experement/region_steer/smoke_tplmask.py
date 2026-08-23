#!/usr/bin/env python3
"""region_steer/smoke_tplmask.py — SMOKE for the new TM (tpl_mask) arm.

Proves, at layer-16 block OUTPUT, that arm TM steers ONLY the masked-blank positions
inside the injected template span (tpl_mask), leaving tpl_ctx and out_mask untouched.

Method (per case, deterministic step-0 forward):
  h_before = block16 output with NO steer (arm E, alpha 0)
  h_after  = block16 output with arm TM, alpha 84
  delta    = h_after - h_before   -> nonzero ONLY where the hook wrote
The hook adds (-alpha)*v (unit-norm v), so every steered row must move by L2 ~= alpha,
and every non-steered row by ~0. We check the changed-row set == tpl_mask exactly, and
that it is disjoint from tpl_ctx and from the output region.

Tiny GPU touch: 2 cases, 2 forwards each + one full TM generation (diag check). Offline."""
import os, sys, json
import torch

EXP = "/home/ore99/experement"
sys.path[:0] = [EXP, os.path.join(EXP, "region_steer"), os.path.join(EXP, "dija_attack")]
from ladaAndH import discover_blocks
from runner import RegionSteerRunner, load_injection_direction, GEN_LENGTH
from run_dija import MASK_ID
from run40 import CASES, LAYER, POOL, MODEL_ID, CASE_SEED0

ALPHA = 84.0
LIMIT = 2


def main():
    dev = "cuda"
    Bm = {r["id"]: r for r in json.load(open(os.path.join(EXP, "dija_attack/refined_100.json")))}
    print(f"[smoke] loading {MODEL_ID} bf16 ...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.all_tied_weights_keys = {}
    model = model.to(dev).eval()
    _, blocks = discover_blocks(model)
    v_L, vmeta = load_injection_direction(LAYER, POOL, device=dev)
    print(f"[smoke] dir {vmeta['kind']} L{LAYER} norm={vmeta['norm']:.4f}", flush=True)
    runner = RegionSteerRunner(model, tok, dev, blocks, v_L, LAYER)

    all_ok = True
    for cid in CASES[:LIMIT]:
        seed = CASE_SEED0 + int(cid[1:])
        scaffold = Bm[cid]["Refined_behavior"]
        ids, (t0, t1) = runner.build_inputs(scaffold)
        P = ids.shape[1]; total = P + GEN_LENGTH
        counts = runner._setup_regions(ids[0], t0, t1, P, total, seed)
        x0 = torch.full((1, total), MASK_ID, dtype=torch.long, device=dev)
        x0[:, :P] = ids
        attn = torch.ones((1, total), dtype=torch.long, device=dev)

        tpl_mask = runner._tpl_mask
        tpl_ctx = runner._tpl_ctx
        out_reg = runner._out_region
        tpl_span = torch.zeros(total, dtype=torch.bool, device=dev)
        if t0 is not None:
            tpl_span[t0:t1] = True

        # ---- set-algebra proof on the arm's active set (step 0) ----
        runner._arm = "TM"; runner._alpha = ALPHA
        active = runner._active_positions(x0)
        eq_tplmask = bool(torch.equal(active, tpl_mask))
        int_ctx = int((active & tpl_ctx).sum())
        int_out = int((active & out_reg).sum())
        sub_span = bool((active & (~tpl_span)).sum() == 0)
        sub_prompt = int(active[P:].sum())  # must be 0: no output positions

        # ---- before/after residual diff at block-16 output (deterministic) ----
        cap = []
        def _cap(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            cap.append(h[0].detach().float().clone())
        chandle = blocks[LAYER].register_forward_hook(_cap)  # fires AFTER runner's steer hook
        with torch.no_grad():
            runner._arm = "E"; runner._alpha = 0.0; runner._diag = None
            runner.forward(x0, attn)          # cap[0] = h_before (no steer)
            runner._arm = "TM"; runner._alpha = ALPHA
            runner.forward(x0, attn)          # cap[1] = h_after  (TM steer)
        chandle.remove()
        delta = cap[1] - cap[0]               # [total, d]
        rn = delta.norm(dim=-1)               # per-position L2
        changed = rn > 1e-2
        changed_eq_tplmask = bool(torch.equal(changed, tpl_mask))
        moved_in_ctx = int((changed & tpl_ctx).sum())
        moved_in_out = int((changed & out_reg).sum())
        norms_tm = rn[tpl_mask]
        norm_ctx_max = float(rn[tpl_ctx].max()) if int(tpl_ctx.sum()) else 0.0
        norm_out_max = float(rn[out_reg].max())

        # ---- end-to-end: one full TM generation, check hook fired every step at |tpl_mask| ----
        torch.manual_seed(seed)
        r = runner.generate_arm(scaffold, "TM", ALPHA, seed=seed)
        d = r["diag"]

        ok = (eq_tplmask and int_ctx == 0 and int_out == 0 and sub_span and sub_prompt == 0
              and changed_eq_tplmask and moved_in_ctx == 0 and moved_in_out == 0
              and int(tpl_mask.sum()) > 0
              and d["steer_rows_last"] == int(tpl_mask.sum()) and d["steer_steps"] > 0)
        all_ok = all_ok and ok

        print(f"\n===== {cid}  (P={P} total={total} tpl_span=[{t0},{t1}) n_inject={r['n_inject']}) =====")
        print(f"  region counts        : tpl_mask={counts['tpl_mask']}  tpl_ctx={counts['tpl_ctx']}  "
              f"out_region={counts['out_region']}  R_out={counts['R_out']}  R_wrap={counts['R_wrap']}")
        print(f"  ACTIVE(TM) == tpl_mask                : {eq_tplmask}")
        print(f"  ACTIVE(TM) ∩ tpl_ctx  (must be 0)     : {int_ctx}")
        print(f"  ACTIVE(TM) ∩ out_region (must be 0)   : {int_out}")
        print(f"  ACTIVE(TM) ⊆ tpl_span (must be True)  : {sub_span}")
        print(f"  ACTIVE(TM) in output region (must 0)  : {sub_prompt}")
        print(f"  --- before/after L{LAYER} residual diff (alpha={ALPHA:g}, ||v||=1) ---")
        print(f"  #rows changed == tpl_mask             : {changed_eq_tplmask}  "
              f"(changed={int(changed.sum())}, tpl_mask={int(tpl_mask.sum())})")
        print(f"  changed rows inside tpl_ctx (must 0)  : {moved_in_ctx}")
        print(f"  changed rows inside out_region (must0): {moved_in_out}")
        print(f"  steered-row Δ‖h‖: mean={float(norms_tm.mean()):.2f} "
              f"min={float(norms_tm.min()):.2f} max={float(norms_tm.max()):.2f}  (expect ~{ALPHA:g})")
        print(f"  tpl_ctx max Δ‖h‖ (~0): {norm_ctx_max:.4f}   out_region max Δ‖h‖ (~0): {norm_out_max:.4f}")
        print(f"  full-gen diag: steer_steps={d['steer_steps']} steer_rows_last={d['steer_rows_last']} "
              f"(== tpl_mask? {d['steer_rows_last']==int(tpl_mask.sum())})")
        print(f"  tpl_mask mean_proj onto v (diag)      : {d['tpl_mask']['mean_proj']}")
        print(f"  VERDICT {cid}: {'PASS' if ok else 'FAIL'}")

    runner.close()
    print(f"\n[smoke] OVERALL: {'PASS — TM writes at tpl_mask ONLY' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
