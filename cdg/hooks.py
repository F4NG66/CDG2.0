from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


_CANDIDATE_PATHS = [
    "model.transformer.blocks",  
    "transformer.blocks",
    "model.model.layers",         
    "model.layers",
    "model.transformer.h",
    "transformer.h",
]


def _getattr_path(obj, dotted: str):
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def get_blocks(model, blocks_attr: Optional[str] = None) -> nn.ModuleList:
    if blocks_attr:
        blocks = _getattr_path(model, blocks_attr)
        if blocks is not None:
            return blocks
        raise AttributeError(f" blocks_attr='{blocks_attr}' cannot find block list")
    for path in _CANDIDATE_PATHS:
        blocks = _getattr_path(model, path)
        if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
            return blocks
    raise AttributeError(
        "block list fail, print model "
        "BackendConfig.blocks_attr 'model.transformer.blocks'）。"
    )


def _to_tensor(out):
    if isinstance(out, tuple):
        return out[0]
    return out


class HookManager:

    def __init__(self, model, layers: tuple[int, ...], offset: int = -1,
                 blocks_attr: Optional[str] = None):
        self.blocks = get_blocks(model, blocks_attr)
        self.layers = list(layers)
        self.offset = offset
        self.buffers: dict[int, torch.Tensor] = {}    
        self.steer: dict[int, tuple[float, torch.Tensor]] = {} 
        self.steer_positions: Optional[torch.Tensor] = None 
        self._handles = []
        self._register()

    def _register(self):
        for layer in self.layers:
            bi = layer + self.offset
            if not (0 <= bi < len(self.blocks)):
                raise IndexError(
                    f"layer {layer} -> block {bi} overbounded, in total {len(self.blocks)} layers)."
                    " adjust BackendConfig.layer_to_block_offset."
                )
            self._handles.append(
                self.blocks[bi].register_forward_hook(self._make_hook(layer))
            )

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            h = _to_tensor(output)
            self.buffers[layer] = h.detach()
            # ---- steering later----
            if layer in self.steer:
                alpha, vec = self.steer[layer]
                vec = vec.to(h.dtype).to(h.device)
                add = alpha * vec  # (d,)
                if self.steer_positions is None:
                    h = h + add
                else:
                    mask = self.steer_positions.to(h.device).unsqueeze(-1)  # (B,L,1)
                    h = h + add * mask
                if isinstance(output, tuple):
                    return (h,) + tuple(output[1:])
                return h
            return None
        return hook

    def clear(self):
        self.buffers.clear()

    def set_steer(self, layer: int, alpha: float, vec: torch.Tensor,
                  positions: Optional[torch.Tensor] = None):
        self.steer[layer] = (alpha, vec)
        self.steer_positions = positions

    def reset_steer(self):
        self.steer.clear()
        self.steer_positions = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
