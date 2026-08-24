from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .additive import additive_steer
from .combined import combined_steer
from .projection_removal import remove_projection
from .safety_direction import safety_steer
from .schedules import SteeringSchedule

Family = Literal["additive", "projection_removal", "safety_direction", "combined"]


@dataclass
class SteeringIntervention:
    """The four independent intervention dimensions: direction, dose, location, time."""

    family: Family
    layer: int
    token_scope: str
    schedule: SteeringSchedule
    injection_direction: torch.Tensor | None = None
    safety_direction: torch.Tensor | None = None
    alpha: float = 0.0
    beta: float = 0.0
    rho: float = 0.0

    def apply(self, hidden: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.family == "additive":
            if self.injection_direction is None:
                raise ValueError("additive steering requires v_inj")
            changed = additive_steer(hidden, self.injection_direction, self.alpha)
        elif self.family == "projection_removal":
            if self.injection_direction is None:
                raise ValueError("projection removal requires v_inj")
            changed = remove_projection(hidden, self.injection_direction, self.rho)
        elif self.family == "safety_direction":
            if self.safety_direction is None:
                raise ValueError("safety steering requires v_safety (not -v_inj)")
            changed = safety_steer(hidden, self.safety_direction, self.beta)
        elif self.family == "combined":
            if self.injection_direction is None or self.safety_direction is None:
                raise ValueError("combined steering requires both v_inj and v_safety")
            changed = combined_steer(
                hidden, self.injection_direction, self.safety_direction, rho=self.rho, beta=self.beta
            )
        else:  # pragma: no cover - guarded by the type/config parser
            raise ValueError(f"unknown steering family: {self.family}")
        if positions is None:
            return changed
        mask = positions.to(device=hidden.device, dtype=torch.bool).unsqueeze(-1)
        return torch.where(mask, changed, hidden)


class ScheduledHookController:
    """Adapter usable by model forward hooks without duplicating model loading."""

    def __init__(self, intervention: SteeringIntervention) -> None:
        self.intervention = intervention
        self.step = 0
        self.total_steps = 1
        self.positions: torch.Tensor | None = None

    def prepare_step(self, step: int, total_steps: int, positions: torch.Tensor | None) -> None:
        self.step, self.total_steps, self.positions = step, total_steps, positions

    def __call__(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        if layer != self.intervention.layer:
            return hidden
        if not self.intervention.schedule.active(self.step, self.total_steps):
            return hidden
        return self.intervention.apply(hidden, self.positions)
