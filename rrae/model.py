from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class RRAE(nn.Module):
    """Low-rank residual representation autoencoder.

    Given a hidden state ``h``, the model computes ``h_hat = D(E(h))`` through
    a rank-k bottleneck.  The paper-path representation is the residual
    ``r = h - h_hat``; downstream injection directions are constructed in that
    residual space rather than from a content-confounded B-vs-C contrast.
    """

    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        if not 0 < rank <= input_dim:
            raise ValueError("rank must be in [1, input_dim]")
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.encoder = nn.Linear(input_dim, rank, bias=True)
        self.decoder = nn.Linear(rank, input_dim, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(hidden))

    def residual(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden - self(hidden)


@dataclass(frozen=True)
class RRAECheckpoint:
    input_dim: int
    rank: int
    mean: list[float]
    scale: list[float]
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "RRAECheckpoint":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**payload)

    def build_model(self) -> RRAE:
        model = RRAE(self.input_dim, self.rank)
        model.load_state_dict(self.state_dict)
        return model.eval()

    def standardize(self, hidden: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=hidden.dtype, device=hidden.device)
        scale = torch.tensor(self.scale, dtype=hidden.dtype, device=hidden.device)
        return (hidden - mean) / scale.clamp_min(1e-8)

    def residual(self, hidden: torch.Tensor) -> torch.Tensor:
        standardized = self.standardize(hidden.float())
        return self.build_model().to(hidden.device).residual(standardized)
