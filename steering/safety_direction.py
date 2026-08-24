from __future__ import annotations

import torch


def safety_steer(hidden: torch.Tensor, safety_direction: torch.Tensor, beta: float) -> torch.Tensor:
    """Apply ``h' = h + beta * v_safety`` using a separately learned behavior vector."""
    vector = safety_direction.to(device=hidden.device, dtype=hidden.dtype)
    return hidden + float(beta) * vector
