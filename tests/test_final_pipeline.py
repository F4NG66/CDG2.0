from __future__ import annotations

import math

import torch

from rrae.utils import shared_injection_direction
from steering.additive import additive_steer
from steering.combined import combined_steer
from steering.projection_removal import remove_projection
from steering.run_frozen_replication import paired_metrics
from steering.schedules import SteeringSchedule


def test_shared_direction_uses_controlled_contrasts_and_canonical_sign() -> None:
    rows = [{"group": group, "pair_id": "x"} for group in "ABCD"]
    vectors = torch.tensor([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]])
    direction, geometry = shared_injection_direction(vectors, rows)
    assert torch.dot(direction, torch.tensor([1.0, 0.0])) > 0.999
    assert math.isclose(geometry["cos_ba_cd"], 1.0, abs_tol=1e-6)


def test_additive_and_projection_are_fundamentally_different() -> None:
    hidden = torch.tensor([[2.0, 3.0], [0.0, 3.0]])
    direction = torch.tensor([1.0, 0.0])
    additive = additive_steer(hidden, direction, alpha=-1.0)
    projected = remove_projection(hidden, direction, rho=1.0)
    assert torch.equal(additive[:, 0], torch.tensor([1.0, -1.0]))
    assert torch.equal(projected[:, 0], torch.tensor([0.0, 0.0]))


def test_combined_has_independent_doses() -> None:
    hidden = torch.tensor([[2.0, 0.0]])
    result = combined_steer(
        hidden, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), rho=0.5, beta=3.0
    )
    assert torch.allclose(result, torch.tensor([[1.0, 3.0]]))


def test_single_and_persistent_schedule() -> None:
    single = SteeringSchedule(0.25, 0.25, persistent=False)
    assert [step for step in range(1, 9) if single.active(step, 8)] == [2]
    persistent = SteeringSchedule(0.25, 0.5, persistent=True)
    assert [step for step in range(1, 9) if persistent.active(step, 8)] == [2, 3, 4]


def test_pairwise_metrics_require_both_members() -> None:
    rows = [
        {"family": "safety", "pair_id": "1", "b_safe": True, "c_helpful": True},
        {"family": "safety", "pair_id": "2", "b_safe": True, "c_helpful": False},
    ]
    metrics = paired_metrics(rows, expected_pairs=2)["safety"]
    assert metrics["b_safe_conversion"] == 1.0
    assert metrics["c_helpful_preservation"] == 0.5
    assert metrics["strict_paired_success"] == 0.5
