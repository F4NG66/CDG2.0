"""clock_attack/steerer.py — Step-1 latent-clock steering hook (Subliminal Clocks Eq. 4/5).

A forward_hook on the transformer block(s) that edits the block OUTPUT (out[0]) at
the CURRENTLY-MASKED positions only, pushing the model's internal denoising "clock"
from its current step-bin t toward a target bin t̂ (Eq. 4):

    out[0][mask_positions] += alpha * (mu[t̂, l] − mu[t, l])

`mu` is the content-pure mean-vector bank built by build_mu_bank.py on a NEUTRAL
held-out seed set, measured at the same place we steer (block output, layer l).

A RANDOM control (Eq. 5) is included: a norm-matched Gaussian shaped by the per-bin
covariance (diagonal std) and rescaled to the τ-delta's norm. This reproduces the old
random-direction run (execute_latent_clock_attack.py) and must stay INERT.

The generate loop is expected to call `steerer.update(global_step, mask_index)` right
BEFORE every model() call (so the hook knows the current bin t and where the masks are),
and `steerer.set(mode, target_bin, alpha)` once per condition.
"""
from __future__ import annotations
import torch


def load_mu_bank(path: str, device, dtype):
    """Load a per-layer μ bank saved by build_mu_bank.py and move tensors to device/dtype."""
    bank = torch.load(path, map_location="cpu", weights_only=False)
    bank["mu_bin"] = bank["mu_bin"].to(device=device, dtype=dtype)      # [n_bins, d]
    bank["sigma_bin"] = bank["sigma_bin"].to(device=device, dtype=dtype)  # [n_bins, d]
    return bank


class StepSteerer:
    """Holds the per-step steering state and owns the forward hooks.

    Parameters
    ----------
    blocks      : the model's transformer ModuleList (from discover_blocks)
    banks       : {layer_index: mu_bank_dict}  (each dict has mu_bin[n_bins,d], sigma_bin, n_bins)
    total_steps : T (denoising steps), used to map global_step -> bin in [0, n_bins-1]
    device,dtype: model device / compute dtype (bfloat16)
    seed        : RNG seed for the random control (reproducibility)
    """

    def __init__(self, blocks, banks: dict, total_steps: int, device, dtype, seed: int = 0):
        self.banks = banks
        self.layers = sorted(banks.keys())
        self.n_bins = int(banks[self.layers[0]]["n_bins"])
        self.total_steps = int(total_steps)
        self.device = device
        self.dtype = dtype
        self._gen = torch.Generator(device="cpu").manual_seed(seed)

        # per-condition knobs
        self.mode = "off"        # "off" | "tau" | "random"
        self.target_bin = self.n_bins - 1
        self.alpha = 0.0
        # per-step state (updated by the generate loop before each model() call)
        self.cur_bin = 0
        self.mask_index = None   # bool tensor [1, seq] on device

        self._handles = []
        for L in self.layers:
            self._handles.append(blocks[L].register_forward_hook(self._make_hook(L)))

    # -- configuration -----------------------------------------------------
    def set(self, mode: str, target_bin: int, alpha: float):
        assert mode in ("off", "tau", "random")
        self.mode = mode
        # target given as a bin in [0, n_bins-1]; a percent t̂=100 maps to the last bin
        self.target_bin = max(0, min(self.n_bins - 1, int(target_bin)))
        self.alpha = float(alpha)

    def update(self, global_step: int, mask_index: torch.Tensor):
        """Called right BEFORE each model() forward: fix current bin t and mask positions."""
        if self.total_steps <= 1:
            self.cur_bin = 0
        else:
            self.cur_bin = int(round(global_step / (self.total_steps - 1) * (self.n_bins - 1)))
        self.mask_index = mask_index

    # -- the hook ----------------------------------------------------------
    def _make_hook(self, layer: int):
        bank = self.banks[layer]

        def _hook(_module, _inp, out):
            if self.mode == "off" or self.alpha == 0.0 or self.mask_index is None:
                return out
            mrow = self.mask_index[0]            # [seq] bool
            if not bool(mrow.any()):
                return out

            mu = bank["mu_bin"]                  # [n_bins, d] on device/dtype
            delta = mu[self.target_bin] - mu[self.cur_bin]     # [d]  (Eq. 4 direction)

            if self.mode == "random":
                # Eq. 5: norm-matched Gaussian from the per-bin covariance (diag std),
                # rescaled to the τ-delta's norm so it is a *fair* inert control.
                sigma = bank["sigma_bin"][self.cur_bin]        # [d]
                g = torch.randn(mu.shape[-1], generator=self._gen).to(mu) * sigma
                gn = g.norm().clamp_min(1e-6)
                delta = g * (delta.norm() / gn)

            vec = (self.alpha * delta).to(out[0].dtype if isinstance(out, tuple) else out.dtype)

            if isinstance(out, (tuple, list)):
                h = out[0].clone()
                h[0, mrow, :] += vec
                return (h,) + tuple(out[1:])
            else:
                h = out.clone()
                h[0, mrow, :] += vec
                return h
        return _hook

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []
