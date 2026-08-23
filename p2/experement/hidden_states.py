"""hidden_states.py - Per-denoising-step hidden-state capture (Person 2).

FEATURE side of the Shield contrastive dataset Z^(t) = {(H_i^(t), y_i)}: the
FSM/records layer supplies the labels y (survive/collapse); this module supplies
H. It registers forward hooks on chosen transformer blocks of a diffusion LM
(LLaDA 8B / Dream-7B), collects hidden states at every denoising step of the
*target reply*, and persists them. `make_refs_hook` returns a
`refs_hook(turn) -> {"hidden_states": path}` to plug into run_attack.

torch is imported LAZILY so the module loads on machines without it; you only
need torch when you actually capture activations.

This module is a SKELETON: two MODEL-SPECIFIC pieces must be adapted -
  (1) which blocks to hook (see StepActivationCollector ADAPT note), and
  (2) calling .mark_step() once per denoising step inside your generate loop.
Everything else - schema, layout, dtype, refs - is ready.

On-disk layout (one file per turn, all steps/layers stacked):
    {out_dir}/{traj_id}/turn_{turn:02d}.pt
    payload = {"layers": [...], "hidden": Tensor[n_steps, n_layers, seq_len, d_model]}
"""

from __future__ import annotations
import os
from typing import Callable, Sequence


class StepActivationCollector:
    """Registers forward hooks on selected blocks and accumulates their outputs
    across denoising steps. Call .new_turn() before each target generation, run
    generation (calling .mark_step() after each denoising step), then the
    refs_hook calls .stack() -> [n_steps, n_layers, seq_len, d_model].

    ADAPT: `blocks` must be the actual transformer block modules whose *output*
    hidden state you want (e.g. model.model.layers[i]). The hook assumes the
    block returns a tensor or a tuple whose first element is the hidden state
    [batch, seq, d_model]; adjust `_make_hook` if your block's output differs.
    """

    def __init__(self, blocks: "Sequence", layer_ids: Sequence[int]):
        self.layer_ids = list(layer_ids)
        self._handles = []
        self._step_buffers: list = []      # per step: stacked [n_layers, seq, d_model]
        self._cur: dict = {}
        for pos, block in enumerate(blocks):
            self._handles.append(block.register_forward_hook(self._make_hook(pos)))

    def _make_hook(self, pos: int):
        import torch

        def _hook(_module, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            # snapshot this block's activation for the current step (CPU, fp16)
            self._cur[pos] = h.detach()[0].to(torch.float16).cpu()
        return _hook

    def new_turn(self) -> None:
        self._step_buffers = []
        self._cur = {}

    def mark_step(self) -> None:
        """Call once per denoising step, AFTER that step's forward pass, to
        snapshot all hooked layers."""
        import torch
        if not self._cur:
            return
        ordered = [self._cur[p] for p in range(len(self.layer_ids))]
        self._step_buffers.append(torch.stack(ordered))   # [n_layers, seq, d_model]
        self._cur = {}

    def stack(self):
        """[n_steps, n_layers, seq_len, d_model] for the turn just generated."""
        import torch
        if not self._step_buffers:
            return torch.empty(0)
        return torch.stack(self._step_buffers)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def make_refs_hook(collector: StepActivationCollector, *,
                   out_dir: str, traj_id: str) -> Callable[[int], dict]:
    """Build a refs_hook(turn) for run_attack. Persists whatever the collector
    accumulated for the turn just generated and returns its path.

    Usage per turn (inside your target_model_call):
        collector.new_turn()
        for step in denoising_loop: ...; collector.mark_step()
        reply = decode(...)
    run_attack then calls refs_hook(turn), which saves collector.stack()."""
    def refs_hook(turn: int) -> dict:
        import torch
        h = collector.stack()
        if h.numel() == 0:
            return {}
        path = os.path.join(out_dir, traj_id, f"turn_{turn:02d}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"layers": collector.layer_ids, "hidden": h}, path)
        return {"hidden_states": path}
    return refs_hook