from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def topk_entropy(logits: torch.Tensor, k: int) -> torch.Tensor:

    p = F.softmax(logits.float(), dim=-1)
    topp = torch.topk(p, k, dim=-1).values          # (..., k)
    topp = topp / topp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(topp * torch.log(topp + 1e-12)).sum(dim=-1)


@torch.no_grad()
def full_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits.float(), dim=-1)
    return -(p * torch.log(p + 1e-12)).sum(dim=-1)


@torch.no_grad()
def max_prob(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.float(), dim=-1).max(dim=-1).values
