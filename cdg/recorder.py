from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import RecordConfig, Scope
from .metrics import topk_entropy, full_entropy, max_prob


def fractions_to_steps(fractions, total_steps: int) -> dict[int, float]:
    out = {}
    for fr in fractions:
        s = max(1, min(total_steps, round(fr * total_steps)))
        out[s] = fr            # if two fracs map to same step, later wins (fine)
    return out


@dataclass
class GenerationRecord:
    meta: dict
    layout: dict = field(default_factory=dict)        # prompt_len/gen_length/total
    regions: dict = field(default_factory=dict)        # region -> (lo,hi)|None
    response_text: str = ""
    # scope -> frac -> {layer -> vec}
    sae: dict = field(default_factory=dict)
    hidden: dict = field(default_factory=dict)
    entropy: dict = field(default_factory=dict)        # scope -> frac -> {...}
    counts: dict = field(default_factory=dict)         # scope -> frac -> n_pos
    decoded: dict = field(default_factory=dict)        # region -> frac -> text
    judge: Optional[dict] = None
    # Token-level sparse SAE activations (only when RecordConfig.record_token_level=True).
    # scope -> layer -> {"token_ids": int32[T], "idx": int32[T,k], "val": f16[T,k]}
    # Recorded only at main_fraction for unmask-position scopes (pos == "unmask").
    sae_tokens: dict = field(default_factory=dict)


class Recorder:
    """Region/scope aware recorder.

    A *region* is a token span (lo, hi).  A *scope* says: take `region`, keep the
    `pos` positions inside it (mask / unmask / all, recomputed each step from x),
    encode with the SAE of kind `sae_kind`, pool, and store under `scope.name`.

    The PRIMARY scope for template-injection is `tpl_mask`: the injected blanks
    sitting inside the prompt.  The old design only ever recorded the appended
    output region; that was the bug.
    """

    def __init__(self, bundles, rcfg: RecordConfig, tokenizer=None):
        self.bundles_by_kind = {b.kind: b for b in bundles}
        self.bundles = bundles
        self.cfg = rcfg
        self.tok = tokenizer
        self.step_to_frac: dict[int, float] = {}
        self.rec: Optional[GenerationRecord] = None
        self.mask_id: Optional[int] = None

    # -- lifecycle ----------------------------------------------------------
    def begin(self, meta: dict, total_steps: int):
        self.step_to_frac = fractions_to_steps(self.cfg.record_fractions, total_steps)
        self.rec = GenerationRecord(meta={**meta, "main_fraction": self.cfg.main_fraction})

    def set_layout(self, *, prompt_len: int, gen_length: int, total: int,
                   regions: dict, mask_id: int):
        """Called by the runner once per case, before denoising."""
        self.mask_id = mask_id
        self.rec.layout = {"prompt_len": prompt_len, "gen_length": gen_length,
                           "total": total}
        # normalize regions: keep only valid spans
        clean = {}
        for k, span in regions.items():
            if span is None:
                clean[k] = None
            else:
                lo, hi = int(span[0]), int(span[1])
                clean[k] = (lo, hi) if hi > lo else None
        self.rec.regions = clean

    def should_record(self, global_step: int) -> bool:
        return global_step in self.step_to_frac

    # -- helpers ------------------------------------------------------------
    def _sel_positions(self, x_row: torch.Tensor, lo: int, hi: int, pos: str):
        """Return absolute indices in [lo,hi) matching the pos filter."""
        seg = x_row[lo:hi]
        is_mask = (seg == self.mask_id)
        if pos == "mask":
            local = is_mask.nonzero(as_tuple=True)[0]
        elif pos == "unmask":
            local = (~is_mask).nonzero(as_tuple=True)[0]
        else:
            local = torch.arange(seg.numel(), device=seg.device)
        return local + lo

    # -- main record --------------------------------------------------------
    @torch.no_grad()
    def record(self, global_step: int, x: torch.Tensor, logits: torch.Tensor,
               buffers: dict):
        frac = self.step_to_frac[global_step]
        x_row = x[0]

        for sc in self.cfg.scopes:
            span = self.rec.regions.get(sc.region)
            if span is None:
                continue
            lo, hi = span
            idx = self._sel_positions(x_row, lo, hi, sc.pos)
            n_pos = int(idx.numel())
            self.rec.counts.setdefault(sc.name, {})[frac] = n_pos

            bundle = self.bundles_by_kind.get(sc.sae_kind)
            sae_store = self.rec.sae.setdefault(sc.name, {}).setdefault(frac, {})
            hid_store = self.rec.hidden.setdefault(sc.name, {}).setdefault(frac, {})

            # entropy over the selected positions (region-local)
            if n_pos > 0:
                seg_logits = logits[0, idx]
                self.rec.entropy.setdefault(sc.name, {})[frac] = {
                    "topk_entropy": topk_entropy(seg_logits, self.cfg.entropy_top_k)
                                    .mean().half().cpu(),
                    "full_entropy": full_entropy(seg_logits).mean().half().cpu(),
                    "max_prob": max_prob(seg_logits).mean().half().cpu(),
                }

            for layer, sae in (bundle.saes.items() if bundle else []):
                h = buffers[layer][0]                       # (T, d)
                if n_pos == 0:
                    sae_store[layer] = torch.full((sae.n_features,),
                                                  float("nan")).half()
                    hid_store[layer] = torch.full((h.shape[-1],),
                                                  float("nan")).half()
                    continue
                h_sel = h[idx].float()                      # (m, d)
                z = sae.encode(h_sel)                       # (m, n)
                if self.cfg.pool == "mean":
                    sae_store[layer] = z.mean(0).half().cpu()
                    hid_store[layer] = h_sel.mean(0).half().cpu()
                else:
                    sae_store[layer] = z.half().cpu()
                    hid_store[layer] = h_sel.half().cpu()

                # Token-level sparse activations: only at token_record_fraction
                # (default 1.0 = fully decoded output, real content tokens),
                # only for unmask-position scopes, only when enabled.
                # Do NOT use main_fraction here: at frac=0.10 the diffusion model
                # has already filled EOS padding with high confidence, so unmask
                # positions are dominated by <|endoftext|> tokens, not content.
                if (self.cfg.record_token_level
                        and sc.pos == "unmask"
                        and abs(frac - self.cfg.token_record_fraction) < 1e-6):
                    k_top = min(sae.k, z.shape[-1])
                    topv, topi = torch.topk(z, k_top, dim=-1)  # (m, k)
                    tok_store = self.rec.sae_tokens.setdefault(sc.name, {})
                    tok_store[layer] = {
                        "token_ids": x_row[idx].int().cpu(),   # (m,)
                        "idx": topi.int().cpu(),               # (m, k)
                        "val": topv.half().cpu(),              # (m, k)
                    }

        # decode each region's current content (for inspection / judge)
        if self.cfg.decode_regions and self.tok is not None:
            for rname, span in self.rec.regions.items():
                if span is None:
                    continue
                lo, hi = span
                seg = x_row[lo:hi]
                seg = seg[seg != self.mask_id]
                self.rec.decoded.setdefault(rname, {})[frac] = self.tok.decode(
                    seg.tolist(), skip_special_tokens=True)

    # -- finalize -----------------------------------------------------------
    def finish(self, response_text: str):
        self.rec.response_text = response_text

    def set_judge(self, verdict: dict):
        self.rec.judge = verdict

    def save(self, out_dir: str) -> str:
        m = self.rec.meta
        sub = os.path.join(out_dir, m["model_name"], m["variant"])
        os.makedirs(sub, exist_ok=True)
        # Include steer_alpha in filename so repeated runs with different alpha
        # values don't overwrite each other.
        alpha_tag = ""
        if m.get("steer_alpha") is not None:
            alpha_tag = f"__a{m['steer_alpha']:.1f}"
        path = os.path.join(sub, f"{m['case_id']}__seed{m['seed']}{alpha_tag}.pt")
        torch.save(self.rec.__dict__, path)
        row = {k: m.get(k) for k in
               ("case_id", "variant", "content_type", "has_template",
                "attack_method", "is_neutral", "model_name", "seed")}
        # Also persist steering metadata when present
        for steer_key in ("steer_alpha", "steer_layer", "steer_direction",
                          "steer_mode", "steer_scope_region", "steer_pos"):
            if steer_key in m:
                row[steer_key] = m[steer_key]
        row.update({"path": path,
                    "response_text": self.rec.response_text,
                    "regions": self.rec.regions,
                    "judge": self.rec.judge})
        with open(os.path.join(out_dir, "manifest.jsonl"), "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
