from __future__ import annotations
import re
from typing import Optional

import torch

from ..config import BackendConfig
from ..hooks import HookManager
from ..sae import load_sae, sae_ckpt_path, SAEBundle
from ..denoise import denoise


# --------------------------------------------------------------------------
# tolerant token-subsequence search (handles BPE boundary wobble)
# --------------------------------------------------------------------------
def _find_subseq(hay: list[int], needle: list[int]) -> tuple[int, int]:
    if not needle:
        return (-1, -1)
    n = len(needle)
    for s in range(0, len(hay) - n + 1):
        if hay[s:s + n] == needle:
            return (s, s + n)
    return (-1, -1)


def _tolerant_find(hay: list[int], needle: list[int]) -> tuple[int, int]:
    for cand in (needle, needle[1:], needle[:-1], needle[1:-1]):
        if not cand:
            continue
        s, e = _find_subseq(hay, cand)
        if s >= 0:
            return (s, e)
    return (-1, -1)


class DLMRunner:
    def __init__(self, cfg: BackendConfig, sae_root: str, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self._load_model()
        self._resolve_mask_id()
        self._load_bundles(sae_root)
        self.hooks = HookManager(self.model, cfg.record_layers,
                                 offset=cfg.layer_to_block_offset,
                                 blocks_attr=cfg.blocks_attr)
        self._steer = None   # set via set_steering(...)

    def set_steering(self, vectors: dict, alpha: float, scope_region: str = "template",
                     pos: str = "mask"):
        """vectors: {layer:int -> 1D tensor[d]}. Subtract with alpha<0 to remove
        the injection direction. Applied at `scope_region` positions (pos filter,
        fixed at step 0)."""
        self._steer = dict(vectors=vectors, alpha=alpha,
                           scope_region=scope_region, pos=pos)

    def clear_steering(self):
        self._steer = None
        self.hooks.reset_steer()

    # -- setup --------------------------------------------------------------
    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.cfg.model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16).to(self.device).eval()

    def _resolve_mask_id(self):
        mid = self.cfg.mask_id
        if mid is None:
            mid = getattr(self.model.config, "mask_token_id", None) \
                or getattr(self.tokenizer, "mask_token_id", None)
        if mid is None:
            raise ValueError("cannot resolve mask id; set BackendConfig.mask_id.")
        self.mask_id = int(mid)

    def _load_bundles(self, sae_root: str):
        import os
        self.bundles = []
        for spec in self.cfg.saes:
            repo_root = os.path.join(sae_root, spec.name)
            saes = {}
            for layer in self.cfg.record_layers:
                ckpt, cfgp = sae_ckpt_path(repo_root, layer, spec.trainer)
                saes[layer] = load_sae(ckpt, config_path=cfgp).to(self.device)
            self.bundles.append(SAEBundle(name=spec.name, kind=spec.kind, saes=saes))

    # -- forward ------------------------------------------------------------
    def forward(self, x, attention_mask):
        out = self.model(x, attention_mask=attention_mask)
        return out.logits if hasattr(out, "logits") else out[0]

    def _tok(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # -- prompt building + region resolution --------------------------------
    def build_inputs(self, case) -> tuple[torch.Tensor, dict]:
        """Return (ids[1,P], regions) for one PromptCase.

        regions = {"template": (t0,t1)|None, "harm": (h0,h1)|None,
                   "output": (P, total)}  (output added later by generate())
        """
        tcfg = self.cfg.template
        content = case.user_content.replace(tcfg.behavior_placeholder, case.behavior)

        # expand <mask:N> -> mask_token * N
        content = re.sub(tcfg.mask_marker_re,
                         lambda m: self.cfg.mask_token * int(m.group(1)), content)

        # keep sentinels in the text so we can locate them in token space,
        # then strip the sentinel tokens out.
        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False)

        ids = self._tok(prompt_str)
        t0 = t1 = None

        if tcfg.open_tag in content and tcfg.close_tag in content:
            o_ids = self._tok(tcfg.open_tag)
            c_ids = self._tok(tcfg.close_tag)
            os_, oe = _tolerant_find(ids, o_ids)
            cs, ce = _tolerant_find(ids, c_ids)
            if os_ >= 0 and cs >= 0 and oe <= cs:
                inner = ids[oe:cs]                       # template content tokens
                new_ids = ids[:os_] + inner + ids[ce:]   # remove both tags
                t0, t1 = os_, os_ + len(inner)
                ids = new_ids

        # locate behavior (harm) span by tolerant subsequence search
        h0 = h1 = None
        if case.behavior:
            beh_ids = self._tok(case.behavior)
            hs, he = _tolerant_find(ids, beh_ids)
            if hs >= 0:
                h0, h1 = hs, he

        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        regions = {
            "template": (t0, t1) if t0 is not None else None,
            "harm": (h0, h1) if h0 is not None else None,
        }
        return ids_t, regions

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(self, case, recorder) -> str:
        dc = self.cfg.decode
        ids, regions = self.build_inputs(case)
        P = ids.shape[1]
        total = P + dc.gen_length
        x = torch.full((1, total), self.mask_id, dtype=torch.long, device=self.device)
        x[:, :P] = ids
        attn = torch.ones((1, total), dtype=torch.long, device=self.device)

        regions["output"] = (P, total)
        if recorder is not None:
            recorder.set_layout(prompt_len=P, gen_length=dc.gen_length, total=total,
                                regions=regions, mask_id=self.mask_id)

        # optional steering: build a fixed position mask over the chosen region.
        if self._steer is not None:
            self.hooks.reset_steer()
            st = self._steer
            span = regions.get(st["scope_region"])
            if span is not None:
                lo, hi = span
                seg = x[0, lo:hi]
                is_mask = (seg == self.mask_id)
                local = (is_mask if st["pos"] == "mask"
                         else ~is_mask if st["pos"] == "unmask"
                         else torch.ones_like(is_mask))
                posmask = torch.zeros((1, total), dtype=torch.bool, device=self.device)
                posmask[0, lo:hi] = local
                
                for layer, vec in st["vectors"].items():
                    vec_tensor = torch.as_tensor(vec, device=self.device)
                    hidden_size = getattr(self.model.config, "hidden_size", 4096)
                    
                    if vec_tensor.shape[0] > hidden_size and self.bundles:
                        sae = self.bundles[0].saes.get(layer)
                        if sae is not None:
                            if hasattr(sae, "W_dec"):
            
                                vec_tensor = (vec_tensor.to(sae.W_dec.dtype) @ sae.W_dec.T).view(-1)
                            elif hasattr(sae, "decode"):
                                with torch.no_grad():
                                    vec_tensor = sae.decode(vec_tensor.view(1, -1)).view(-1)
                                    
                    self.hooks.set_steer(layer, st["alpha"],
                                         vec_tensor.to(torch.bfloat16),
                                         positions=posmask)

        x = denoise(self, x, attn, steps=dc.steps, gen_length=dc.gen_length,
                    prompt_len=P, block_length=dc.block_length,
                    temperature=dc.temperature, remask=self.cfg.remask,
                    mask_id=self.mask_id, recorder=recorder,
                    fill_all_masks=dc.fill_all_masks)

        # response = everything generated after the original prompt's NON-mask
        # prefix.  For injection cases the filled blanks live inside [0,P), so we
        # decode both the (now-filled) template span and the output region.
        out_ids = x[0, P:]
        resp = self.tokenizer.decode(out_ids.tolist(), skip_special_tokens=True)
        if regions.get("template"):
            t0, t1 = regions["template"]
            tpl_filled = self.tokenizer.decode(x[0, t0:t1].tolist(),
                                               skip_special_tokens=True)
            resp = (tpl_filled + "\n" + resp).strip()
        if recorder is not None:
            recorder.finish(resp)
        return resp