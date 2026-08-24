from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SteeringSchedule:
    """Temporal intervention window in normalized denoising time [0, 1]."""

    start_fraction: float = 0.0
    end_fraction: float = 1.0
    persistent: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_fraction <= self.end_fraction <= 1.0:
            raise ValueError("schedule requires 0 <= start <= end <= 1")

    def active(self, step: int, total_steps: int) -> bool:
        if total_steps <= 0 or step <= 0:
            return False
        target = max(1, round(self.start_fraction * total_steps))
        if not self.persistent:
            return step == target
        fraction = step / total_steps
        return self.start_fraction <= fraction <= self.end_fraction
