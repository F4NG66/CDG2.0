from __future__ import annotations

import torch


def additive_steer(hidden: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    """Apply the fixed displacement ``h' = h + alpha * v_inj``.

    Unlike projection removal, every selected state receives the same vector.
    The caller owns the sign convention; with v_inj pointing toward injection,
    a negative alpha moves away from it.
    """
    vector = direction.to(device=hidden.device, dtype=hidden.dtype)
    return hidden + float(alpha) * vector
