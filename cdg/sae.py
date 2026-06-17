from __future__ import annotations
import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = int(k)
        self.W_enc = nn.Parameter(torch.zeros(n_features, d_model))  # (n,d)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.zeros(d_model, n_features))  # (d,n)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.W_enc.dtype)
        pre = F.relu((x - self.b_dec) @ self.W_enc.t() + self.b_enc)
        if self.k and self.k < self.n_features:
            topv, topi = torch.topk(pre, self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, topi, topv)
            return z
        return pre

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    def feature_direction(self, f: int) -> torch.Tensor:
        v = self.W_dec[:, f]
        return v / (v.norm() + 1e-8)


_ALIASES = {
    "W_enc": ["encoder.weight", "W_enc"],
    "b_enc": ["encoder.bias", "b_enc"],
    "W_dec": ["decoder.weight", "W_dec"],
    "b_dec": ["b_dec", "b_pre", "decoder.bias"],
}


def _pick(state, names):
    for n in names:
        if n in state:
            return state[n]
    return None


def _load_state(path: str):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("state_dict", obj) if isinstance(obj, dict) else obj


def load_sae(ckpt_path: str, config_path: Optional[str] = None,
             k: Optional[int] = None) -> TopKSAE:
    state = _load_state(ckpt_path)

    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    tcfg = cfg.get("trainer", cfg)

    W_enc = _pick(state, _ALIASES["W_enc"])
    W_dec = _pick(state, _ALIASES["W_dec"])
    b_enc = _pick(state, _ALIASES["b_enc"])
    b_dec = _pick(state, _ALIASES["b_dec"])
    if W_enc is None or W_dec is None:
        raise KeyError("missing encoder/decoder weight, actual keys:\n  "
                       + "\n  ".join(map(str, state.keys())))

    n, d = W_enc.shape
    if tuple(W_dec.shape) == (n, d):    
        W_dec = W_dec.t().contiguous()

    kk = k or tcfg.get("k")
    if kk is None and "k" in state:      
        kk = int(state["k"].item() if hasattr(state["k"], "item") else state["k"])
    if kk is None:
        raise ValueError(" cannot get top k (config with no trainer.k, state with no k buffer")

    sae = TopKSAE(d_model=d, n_features=n, k=int(kk))
    with torch.no_grad():
        sae.W_enc.copy_(W_enc.float())
        sae.W_dec.copy_(W_dec.float())
        if b_enc is not None:
            sae.b_enc.copy_(b_enc.float())
        if b_dec is not None:
            sae.b_dec.copy_(b_dec.float())
    return sae.eval()


def sae_ckpt_path(repo_root: str, layer: int, trainer: int) -> tuple[str, str]:
    import glob
    leaf = os.path.join(f"resid_post_layer_{layer}", f"trainer_{trainer}")
    candidates = [os.path.join(repo_root, leaf)]                       # flat (LLaDA)
    candidates += sorted(glob.glob(os.path.join(repo_root, "*", leaf)))  # nested (Dream)
    for d in candidates:
        if os.path.exists(os.path.join(d, "ae.pt")):
            return os.path.join(d, "ae.pt"), os.path.join(d, "config.json")
    d = candidates[0]   
    return os.path.join(d, "ae.pt"), os.path.join(d, "config.json")


from dataclasses import dataclass


@dataclass
class SAEBundle:
    name: str
    kind: str                       # "mask" | "unmask"
    saes: dict                      # {layer:int -> TopKSAE}
