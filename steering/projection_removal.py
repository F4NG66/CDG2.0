from __future__ import annotations

import torch


def remove_projection(hidden: torch.Tensor, direction: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
    """Remove a state-dependent injection component.

    For unit v_inj: ``h' = h - rho * <h, v_inj> * v_inj``.
    ``direction`` is normalized here to prevent vector norm from changing rho.
    """
    vector = direction.to(device=hidden.device, dtype=hidden.dtype)
    vector = vector / vector.norm().clamp_min(torch.finfo(hidden.dtype).eps)
    coefficient = torch.einsum("...d,d->...", hidden, vector).unsqueeze(-1)
    return hidden - float(rho) * coefficient * vector
