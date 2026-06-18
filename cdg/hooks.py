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
        raise AttributeError(f"blocks_attr='{blocks_attr}' cannot find block list")
    for path in _CANDIDATE_PATHS:
        blocks = _getattr_path(model, path)
        if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
            return blocks
    raise AttributeError(
        "block list fail; set BackendConfig.blocks_attr (e.g. 'model.transformer.blocks')."
    )


def _to_tensor(out):
    if isinstance(out, tuple):
        return out[0]
    return out


class HookManager:
    """Forward-hook manager supporting two intervention modes:

    1. Direction steering  (set_steer):
       h += alpha * vec   at target positions.

    2. Feature zeroing     (set_feature_zero):
       Encodes h through a SAE, zeros the listed feature activations, then
       SUBTRACTS those features' decoder contributions from the ORIGINAL h.
       This is exact: h_new = h - sum_f( z_f * W_dec[:,f] )
       It preserves the reconstruction error (no encoder approximation noise).
    """

    def __init__(self, model, layers: tuple[int, ...], offset: int = -1,
                 blocks_attr: Optional[str] = None):
        self.blocks = get_blocks(model, blocks_attr)
        self.layers = list(layers)
        self.offset = offset
        self.buffers: dict[int, torch.Tensor] = {}
        # direction steering: layer -> (alpha, vec)
        self.steer: dict[int, tuple[float, torch.Tensor]] = {}
        self.steer_positions: Optional[torch.Tensor] = None
        # feature zeroing: layer -> (sae, feat_ids_tensor)
        self._feat_zero: dict[int, tuple] = {}
        self._feat_zero_positions: Optional[torch.Tensor] = None
        self._handles = []
        self._register()

    def _register(self):
        for layer in self.layers:
            bi = layer + self.offset
            if not (0 <= bi < len(self.blocks)):
                raise IndexError(
                    f"layer {layer} -> block {bi} out of range "
                    f"({len(self.blocks)} blocks). "
                    "Adjust BackendConfig.layer_to_block_offset."
                )
            self._handles.append(
                self.blocks[bi].register_forward_hook(self._make_hook(layer))
            )

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            h = _to_tensor(output)
            self.buffers[layer] = h.detach()

            modified = False
            h_out = h

            # -- mode 1: direction steering ----------------------------------
            if layer in self.steer:
                alpha, vec = self.steer[layer]
                vec = vec.to(h_out.dtype).to(h_out.device)
                add = alpha * vec  # (d,)
                if self.steer_positions is None:
                    h_out = h_out + add
                else:
                    pos = self.steer_positions.to(h_out.device).unsqueeze(-1)
                    h_out = h_out + add * pos
                modified = True

            # -- mode 2: feature zeroing -------------------------------------
            if layer in self._feat_zero:
                sae, feat_ids = self._feat_zero[layer]
                feat_ids = feat_ids.to(h_out.device)
                # encode → pick activated values for target features
                h_f = h_out.float()
                z = sae.encode(h_f)                    # (B, T, n_features)
                z_sel = z[..., feat_ids]               # (B, T, k)
                # decoder columns for those features: W_dec shape (d, n_features)
                W = sae.W_dec.float().to(h_out.device) # (d, n)
                W_sel = W[:, feat_ids]                 # (d, k)
                # contribution to remove: (B, T, k) @ (k, d) → (B, T, d)
                delta = z_sel @ W_sel.t()
                if self._feat_zero_positions is not None:
                    pos = self._feat_zero_positions.to(h_out.device).unsqueeze(-1)
                    delta = delta * pos
                h_out = (h_out.float() - delta).to(h.dtype)
                modified = True

            if not modified:
                return None
            if isinstance(output, tuple):
                return (h_out,) + tuple(output[1:])
            return h_out
        return hook

    # -- lifecycle -----------------------------------------------------------
    def clear(self):
        self.buffers.clear()

    # -- direction steering --------------------------------------------------
    def set_steer(self, layer: int, alpha: float, vec: torch.Tensor,
                  positions: Optional[torch.Tensor] = None):
        self.steer[layer] = (alpha, vec)
        self.steer_positions = positions

    def reset_steer(self):
        self.steer.clear()
        self.steer_positions = None

    # -- feature zeroing -----------------------------------------------------
    def set_feature_zero(self, layer: int, sae, feature_ids,
                         positions: Optional[torch.Tensor] = None):
        """Zero out the contribution of `feature_ids` at `layer`.

        Args:
            sae: a TopKSAE for this layer (must be on same device as model during fwd).
            feature_ids: list[int] or 1-D LongTensor of SAE feature indices to suppress.
            positions: optional bool tensor (B, T); if given, only suppress at True positions.
        """
        ids = (feature_ids if isinstance(feature_ids, torch.Tensor)
               else torch.tensor(feature_ids, dtype=torch.long))
        self._feat_zero[layer] = (sae, ids)
        self._feat_zero_positions = positions

    def reset_feature_zero(self):
        self._feat_zero.clear()
        self._feat_zero_positions = None

    def reset_all(self):
        self.reset_steer()
        self.reset_feature_zero()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
