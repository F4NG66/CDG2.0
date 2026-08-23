#!/usr/bin/env python3
"""region_steer/runner.py — FALSIFICATION TEST: does steering the model's internal
representation of the PROMPT/SCAFFOLD region (tpl_ctx) move the DIJA outcome, when the
identical steering on the OUTPUT region (out_mask) does not?

Region-swap steering, SAME direction (v_injection_svd), varying only WHERE it is applied:

  ARM E  : no steering (baseline)
  ARM O  : steer currently-masked OUTPUT positions  (region every prior attempt used)
  ARM T  : steer tpl_ctx = UNMASKED scaffold-text token positions inside the injected
           template span [t0,t1)  (the untested prompt-region representation)
  ARM TM : steer tpl_mask = the MASKED-BLANK positions inside the same injected template
           span [t0,t1)  (the scaffold's fill-in blanks; the exact structural COMPLEMENT of
           tpl_ctx within the span). Fixed set (positions that were MASK_ID at t=0), steered
           throughout denoising just like T — so T-vs-TM isolates which HALF of the scaffold
           span carries the effect: unmasked context (T) vs masked blanks (TM).
  ARM R  : steer a COUNT-MATCHED (|tpl_ctx|) random subset of the OUTPUT region
           (non-scaffold; controls "region" vs "number of steered positions")
  ARM Rw : steer the chat-wrapper prompt positions (strictly non-scaffold, non-harm,
           but count-UNmatched — the wrapper pool is only ~13 tokens)

Direction: v = clockv2 v_injection_svd['mean'][L] (unit norm, +v = "more injected").
Steering SUBTRACTS: h[pos] += (-alpha) * v   (push toward "less injected"). Applied at the
transformer block OUTPUT for layer L — the exact place v was measured (block-output mean pool).

REUSE, DON'T REINVENT / EXTEND, DON'T EDIT:
  * denoiser  = dija_attack.cdg_denoise.denoise (fill_all_masks=True), imported unedited.
  * prompt build + TPL span + decode = dija_attack.run_dija.PaperRunner, subclassed.
  * direction = clockv2/data/probes/v_injection_svd.pt, loaded read-only.
Nothing in dija_attack/ or clockv2/ is modified.
"""
from __future__ import annotations
import os, sys, re
import torch

EXP = "/home/ore99/experement"
sys.path.insert(0, os.path.join(EXP, "dija_attack"))
from cdg_denoise import denoise                    # noqa: E402  (unedited, fill_all_masks=True)
from run_dija import PaperRunner, MASK_ID          # noqa: E402  (verified build/decode path)

V_INJECTION_PATH = os.path.join(EXP, "clockv2", "data", "probes", "v_injection_svd.pt")
GEN_LENGTH = 128
STEPS = 128
BLOCK_LENGTH = 128
TEMPERATURE = 0.2                                   # paper attack config (run_dija default)


def load_injection_direction(layer: int, pool: str = "mean", device="cuda", dtype=torch.bfloat16):
    d = torch.load(V_INJECTION_PATH, map_location="cpu", weights_only=False)
    v = d["v"][pool][layer].to(device=device, dtype=dtype)        # [4096], unit norm
    meta = {"path": V_INJECTION_PATH, "kind": d.get("kind"), "layer": layer,
            "pool": pool, "norm": float(v.float().norm()), "layers_available": d.get("layers")}
    return v, meta


class _StepHooks:
    """Stand-in for runner.hooks that cdg_denoise.denoise calls .clear() on each step."""
    def clear(self):
        pass


class RegionSteerRunner(PaperRunner):
    """PaperRunner + a single block-output steering hook whose active positions are chosen
    per denoising step according to the arm. build_inputs/decode inherited unchanged."""

    def __init__(self, model, tokenizer, device, blocks, v_L, layer: int):
        super().__init__(model, tokenizer, device)
        self.hooks = _StepHooks()
        self.blocks = blocks
        self.layer = layer
        self.v_L = v_L                              # [d] on device, bf16, unit norm
        # per-generation state
        self._arm = "E"
        self._alpha = 0.0                           # magnitude; steering uses -alpha
        self._P = None
        self._total = None
        self._tpl_ctx = None                        # bool [total]
        self._tpl_mask = None                       # bool [total] (masked blanks in tpl span)
        self._out_region = None                     # bool [total]
        self._R_out = None                          # bool [total]
        self._R_wrap = None                         # bool [total]
        self._cur_x = None
        # diagnostics accumulated across steps
        self._diag = None
        self._handle = self.blocks[layer].register_forward_hook(self._make_hook())

    # ---- region set-up at generation entry --------------------------------
    def _setup_regions(self, ids_row, t0, t1, P, total, seed):
        dev = ids_row.device
        is_mask_prompt = torch.zeros(total, dtype=torch.bool, device=dev)
        is_mask_prompt[:P] = (ids_row == MASK_ID)
        tpl_span = torch.zeros(total, dtype=torch.bool, device=dev)
        if t0 is not None:
            tpl_span[t0:t1] = True
        self._tpl_ctx = tpl_span & (~is_mask_prompt)          # scaffold text tokens (fixed)
        self._tpl_mask = tpl_span & is_mask_prompt            # scaffold blank tokens (fixed)
        self._out_region = torch.zeros(total, dtype=torch.bool, device=dev)
        self._out_region[P:total] = True
        # R_out: count-matched random subset of output region
        k = int(self._tpl_ctx.sum())
        out_idx = torch.arange(P, total, device=dev)
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(out_idx.numel(), generator=g)[:k]
        self._R_out = torch.zeros(total, dtype=torch.bool, device=dev)
        self._R_out[out_idx[perm.to(dev)]] = True
        # R_wrap: non-scaffold, non-harm prompt positions (chat wrapper)
        self._R_wrap = torch.zeros(total, dtype=torch.bool, device=dev)
        self._R_wrap[:P] = True
        self._R_wrap = self._R_wrap & (~tpl_span) & (~is_mask_prompt)
        self._P, self._total = P, total
        return {"tpl_ctx": k, "tpl_mask": int(self._tpl_mask.sum()),
                "out_region": int(self._out_region.sum()),
                "R_out": int(self._R_out.sum()), "R_wrap": int(self._R_wrap.sum())}

    def _active_positions(self, x):
        """Positions to steer THIS step for the current arm (bool [total])."""
        dev = x.device
        if self._arm == "T":
            return self._tpl_ctx
        if self._arm == "TM":
            return self._tpl_mask
        if self._arm == "R":
            return self._R_out
        if self._arm == "Rw":
            return self._R_wrap
        if self._arm == "O":
            m = torch.zeros(self._total, dtype=torch.bool, device=dev)
            m[self._P:self._total] = (x[0, self._P:] == MASK_ID)   # currently-masked output
            return m
        return torch.zeros(self._total, dtype=torch.bool, device=dev)  # E

    # ---- forward: choose active positions from current x, then run model ----
    def forward(self, x, attention_mask):
        self._cur_x = x
        self._active = self._active_positions(x)
        out = self.model(x, attention_mask=attention_mask)
        return out.logits if hasattr(out, "logits") else out[0]

    def _make_hook(self):
        def _hook(_module, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out       # [1, seq, d]
            active = getattr(self, "_active", None)
            # --- diagnostics at layer L, measured BEFORE any edit ---
            if self._diag is not None:
                vf = self.v_L.float()
                for name, mask in (("tpl_ctx", self._tpl_ctx),
                                   ("tpl_mask", self._tpl_mask),
                                   ("out_mask", self._active if self._arm == "O" else None)):
                    if mask is None or not bool(mask.any()):
                        continue
                    hm = h[0, mask, :].float()
                    self._diag[name]["norm"] += hm.norm(dim=-1).mean().item()
                    self._diag[name]["proj"] += (hm @ vf).mean().item()
                    self._diag[name]["n"] += 1
            # --- steering ---
            if self._arm == "E" or self._alpha == 0.0 or active is None or not bool(active.any()):
                return out
            vec = (-self._alpha) * self.v_L.to(h.dtype)                 # SUBTRACT injection dir
            if self._diag is not None:
                self._diag["_steer_steps"] += 1
                self._diag["_steer_rows_last"] = int(active.sum())
            if isinstance(out, (tuple, list)):
                h2 = h.clone(); h2[0, active, :] += vec
                return (h2,) + tuple(out[1:])
            h2 = h.clone(); h2[0, active, :] += vec
            return h2
        return _hook

    # ---- one steered generation -------------------------------------------
    @torch.no_grad()
    def generate_arm(self, scaffold, arm, alpha, *, seed=0,
                     steps=STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH,
                     temperature=TEMPERATURE, mask_id=MASK_ID):
        ids, (t0, t1) = self.build_inputs(scaffold)
        P = ids.shape[1]; total = P + gen_length
        counts = self._setup_regions(ids[0], t0, t1, P, total, seed)
        x = torch.full((1, total), mask_id, dtype=torch.long, device=self.device)
        x[:, :P] = ids
        attn = torch.ones((1, total), dtype=torch.long, device=self.device)
        n_inject = int((ids == mask_id).sum())

        self._arm = arm
        self._alpha = float(alpha)
        self._diag = {"tpl_ctx": {"norm": 0.0, "proj": 0.0, "n": 0},
                      "tpl_mask": {"norm": 0.0, "proj": 0.0, "n": 0},
                      "out_mask": {"norm": 0.0, "proj": 0.0, "n": 0},
                      "_steer_steps": 0, "_steer_rows_last": 0}

        x = denoise(self, x, attn, steps=steps, gen_length=gen_length, prompt_len=P,
                    block_length=block_length, temperature=temperature,
                    remask="low_confidence", mask_id=mask_id, recorder=None,
                    fill_all_masks=True)

        out_ids = x[0, P:]
        resp = self.tokenizer.decode(out_ids.tolist(), skip_special_tokens=True)
        tpl_filled = None
        if t0 is not None:
            tpl_filled = self.tokenizer.decode(x[0, t0:t1].tolist(), skip_special_tokens=True)
            resp = (tpl_filled + "\n" + resp).strip()

        d = self._diag
        def _avg(sub):
            return {"mean_norm": sub["norm"] / sub["n"] if sub["n"] else None,
                    "mean_proj": sub["proj"] / sub["n"] if sub["n"] else None, "steps": sub["n"]}
        diag = {"tpl_ctx": _avg(d["tpl_ctx"]), "tpl_mask": _avg(d["tpl_mask"]),
                "out_mask": _avg(d["out_mask"]),
                "steer_steps": d["_steer_steps"], "steer_rows_last": d["_steer_rows_last"]}
        self._diag = None
        return {"response": resp, "tpl_filled": tpl_filled, "n_inject": n_inject,
                "region_counts": counts, "P": P, "total": total, "diag": diag,
                "arm": arm, "alpha": alpha}

    def close(self):
        if self._handle is not None:
            self._handle.remove(); self._handle = None
