from __future__ import annotations

import torch

from .projection_removal import remove_projection
from .safety_direction import safety_steer


def combined_steer(
    hidden: torch.Tensor,
    injection_direction: torch.Tensor,
    safety_direction: torch.Tensor,
    *,
    rho: float,
    beta: float,
) -> torch.Tensor:
    """Independently configure injection removal and safety addition.

    ``h_temp = h - rho <h,v_inj>v_inj`` then
    ``h' = h_temp + beta v_safety``.
    """
    return safety_steer(remove_projection(hidden, injection_direction, rho), safety_direction, beta)
