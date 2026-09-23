"""
M2 norm-preserving additive steering hook (frozen candidate).

Equation (contract item 1):

    h_tilde = h + beta * v
    h_new   = ||h||_2 * h_tilde / (||h_tilde||_2 + eps)

Norm axis: per individual current_mask token position, over the hidden
dimension (4096). NOT pooled across tokens; NOT a whole-sequence norm.
Precision: all steering arithmetic in float32; cast back to the model's
bf16 activation dtype only at the end. eps is applied in float32.

Non-target positions are returned bit-identical to the input (the
bf16 -> fp32 -> bf16 round trip is exact), so the M1 guard
non_target_max_abs_delta == 0.0 continues to hold.

Structurally mirrors M1's SafetyLayerLocationHookBank
(scripts/official_rrae/run_canonical_train_layer_matched_safety_dose_response_canary_v1.py)
so that layer / token scope / schedule / audit semantics are unchanged and
the ONLY difference between M1 and M2 is the update rule.
"""

from typing import Any

import torch

M2_EPS = 1e-8
M2_STEERING_DTYPE = torch.float32
M2_SCOPE = "current_mask"
M2_SCHEDULE = "persistent"
M2_EQUATION = "h_new = ||h||_2 * (h + beta*v) / (||h + beta*v||_2 + eps)"
M2_NORM_AXIS = "per_token_position_over_hidden_dim_4096"


def norm_preserving_update(
    hidden: torch.Tensor,
    target: torch.Tensor,
    vector: torch.Tensor,
    beta: float,
    eps: float = M2_EPS,
) -> tuple[torch.Tensor, dict[str, float]]:
    """hidden: [B, T, H] (model dtype).  target: [B, T] bool.  vector: [H] unit-norm."""
    h32 = hidden.to(M2_STEERING_DTYPE)
    v32 = vector.to(hidden.device, M2_STEERING_DTYPE).view(1, 1, -1)

    # Per-token, per-position L2 over the hidden dimension only.
    h_norm = torch.linalg.vector_norm(h32, dim=-1, keepdim=True)

    h_tilde = h32 + beta * v32
    t_norm = torch.linalg.vector_norm(h_tilde, dim=-1, keepdim=True)

    h_renorm = h_norm * h_tilde / (t_norm + eps)

    mask = target.unsqueeze(-1)
    hidden_new32 = torch.where(mask, h_renorm, h32)

    # ---- audits (computed in fp32, on target positions only) ----
    sel_h = h32[target]
    sel_new = hidden_new32[target]
    delta = sel_new - sel_h
    v_flat = v32.view(1, -1).expand_as(sel_h)

    src_norm = torch.linalg.vector_norm(sel_h, dim=-1)
    out_norm = torch.linalg.vector_norm(sel_new, dim=-1)
    pre_norm = torch.linalg.vector_norm(h_tilde[target], dim=-1)
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    disp = (delta * v_flat).sum(-1)

    non_target = (hidden_new32 - h32)[~target]

    audit = {
        "target_position_count": int(target.sum().item()),
        "target_mean_delta_l2": float(delta_norm.mean().item()),
        "target_min_delta_l2": float(delta_norm.min().item()),
        "target_max_delta_l2": float(delta_norm.max().item()),
        "target_mean_cosine_to_v_safety": float(
            torch.nn.functional.cosine_similarity(delta, v_flat, dim=-1).mean().item()
        ),
        # --- M2-specific norm-preservation audits ---
        "target_mean_source_norm": float(src_norm.mean().item()),
        "target_mean_pre_renorm_norm": float(pre_norm.mean().item()),
        "target_mean_post_norm": float(out_norm.mean().item()),
        "target_max_abs_norm_drift": float((out_norm - src_norm).abs().max().item()),
        "target_max_rel_norm_drift": float(
            ((out_norm - src_norm).abs() / src_norm.clamp_min(eps)).max().item()
        ),
        "target_mean_directional_displacement": float(disp.mean().item()),
        "target_median_directional_displacement": float(disp.median().item()),
        "non_target_max_abs_delta": (
            float(non_target.abs().max().item()) if non_target.numel() else 0.0
        ),
    }

    return hidden_new32.to(hidden.dtype), audit


class M2NormPreservingHookBank:
    """Drop-in analogue of M1's SafetyLayerLocationHookBank, norm-preserving update."""

    def __init__(self, blocks: Any, vector: torch.Tensor, layers: tuple[int, ...],
                 output_to_hidden, output_with_hidden, eps: float = M2_EPS):
        self.vector = vector.detach().to(M2_STEERING_DTYPE).view(-1)
        self.layers = tuple(int(l) for l in layers)
        self.eps = float(eps)
        self._to_hidden = output_to_hidden
        self._with_hidden = output_with_hidden
        self.active_layer: int | None = None
        self.beta = 0.0
        self.active_steps: set[int] = set()
        self.global_step: int | None = None
        self.current_mask: torch.Tensor | None = None
        self.input_ids_sha256: str | None = None
        self.application_count = 0
        self.step_audits: list[dict[str, Any]] = []
        self._handles = {
            l: blocks[l].register_forward_hook(self._make_hook(l)) for l in self.layers
        }

    def reset_run(self, *, layer: int | None, beta: float, active_steps: list[int]) -> None:
        if not (layer is None or layer in self.layers):
            raise RuntimeError(f"Unknown intervention layer: {layer}")
        self.active_layer = layer
        self.beta = float(beta)
        self.active_steps = {int(s) for s in active_steps}
        self.global_step = None
        self.current_mask = None
        self.input_ids_sha256 = None
        self.application_count = 0
        self.step_audits = []

    def set_step_context(self, *, global_step: int, current_mask: torch.Tensor,
                         input_ids_sha256: str) -> None:
        self.global_step = int(global_step)
        self.current_mask = current_mask
        self.input_ids_sha256 = input_ids_sha256

    def _make_hook(self, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any):
            step = self.global_step
            if (self.active_layer != layer or step not in self.active_steps
                    or self.beta == 0.0):
                return None

            hidden = self._to_hidden(output)
            if hidden.ndim != 3:
                raise RuntimeError(f"Unexpected hidden shape at L{layer}: {hidden.shape}")
            if hidden.shape[-1] != self.vector.numel():
                raise RuntimeError(f"Vector/hidden mismatch at L{layer}")
            if self.current_mask is None:
                raise RuntimeError("Missing current mask")

            target = self.current_mask.to(hidden.device, torch.bool)
            if list(target.shape) != list(hidden.shape[:2]):
                raise RuntimeError(f"Target shape mismatch at L{layer}")
            if int(target.sum().item()) == 0:
                raise RuntimeError(f"Empty target mask at L{layer}")

            hidden_new, audit = norm_preserving_update(
                hidden, target, self.vector, self.beta, self.eps
            )

            self.application_count += 1
            audit.update({
                "step": int(step),
                "layer_zero_based": int(layer),
                "scope": M2_SCOPE,
                "equation": M2_EQUATION,
                "norm_axis": M2_NORM_AXIS,
                "eps": self.eps,
                "steering_dtype": str(M2_STEERING_DTYPE),
                "input_ids_sha256": self.input_ids_sha256,
            })
            self.step_audits.append(audit)
            return self._with_hidden(output, hidden_new)

        return hook

    def close(self) -> None:
        for h in self._handles.values():
            h.remove()
