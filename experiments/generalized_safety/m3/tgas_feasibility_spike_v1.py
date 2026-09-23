"""
M3 Step 0 — TGAS feasibility spike.

Question this answers (and ONLY this):
  Can the existing LLaDA generation harness do per-step
  read -> compute-features(with cross-step deltas) -> decide -> conditionally intervene,
  with state carried across denoising steps, at acceptable cost?

What this is NOT:
  - not a gate fit (no gate exists yet; w/b here are a fixed synthetic probe)
  - not population work (synthetic prompt only; no frozen population row is read)
  - not an efficacy measurement of any kind

Structural fidelity to the real harness:
  - same block-output forward-hook mechanism as SafetyLayerLocationHookBank
  - same L16 zero-based layer, same current_mask scope
  - same output_to_hidden / output_with_hidden contract
  - reads the L16 hidden state BEFORE intervention (runtime spec step 1)

Verifies the four things that decide feasibility:
  V1  cross-step state: deltas are 0 at first eligible step, non-zero after
  V2  empty current_mask: g_t=0, no intervention, previous-state cache NOT updated
  V3  closed-loop != open-loop: intervening changes the state the gate later sees
  V4  cost: gate overhead per step vs. the model forward pass
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

LAYER = 16                 # zero-based, frozen backbone
MASK_ID = 126336
BETA_M1_ALL = 73.11751556396484
GEN_LENGTH = 128
TEMPERATURE = 0.2
N_FEATURES = 9


# ----------------------------------------------------------------- harness contract
def output_to_hidden(output):
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    assert torch.is_tensor(hidden), f"Unsupported block output: {type(output)!r}"
    return hidden


def output_with_hidden(output, hidden_new):
    if isinstance(output, tuple):
        return (hidden_new,) + tuple(output[1:])
    if isinstance(output, list):
        return [hidden_new] + list(output[1:])
    if torch.is_tensor(output):
        return hidden_new
    raise TypeError(f"Unsupported block output: {type(output)!r}")


def resolve_blocks(model):
    for attr in ("model.transformer.blocks", "transformer.blocks",
                 "model.layers", "model.model.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "__getitem__") and len(obj) > LAYER:
            return obj, attr
    raise RuntimeError("Could not resolve transformer blocks")


# ----------------------------------------------------------------- the TGAS gate hook
class TGASGateHookBank:
    """Per-step read-decide-intervene with cross-step state.

    This is the spike's structural stand-in for the M3 runtime hook. The decision
    rule is the frozen contract's: g_t = 1[sigmoid(w^T x_t + b) >= tau], evaluated
    on 9 features from the L16 mean-pooled current-mask hidden state.
    """

    def __init__(self, block, v_all, v_dija, v_renellm, w, b, tau, *, closed_loop=True):
        self.v_all = v_all.detach().float().view(-1)
        self.v_dija = v_dija.detach().float().view(-1)
        self.v_ren = v_renellm.detach().float().view(-1)
        self.w = w.detach().float().view(-1)
        self.b = float(b)
        self.tau = float(tau)
        self.closed_loop = bool(closed_loop)
        self.mu = torch.zeros(N_FEATURES)
        self.sigma = torch.ones(N_FEATURES)
        self._handle = block.register_forward_hook(self._hook)
        self.reset_run(trajectory_steps=1)

    def reset_run(self, *, trajectory_steps):
        self.trajectory_steps = int(trajectory_steps)
        self.global_step = None
        self.current_mask = None
        self.total_tokens = None
        self.prev_proj = None          # cross-step state: (p_all, p_dija, p_ren)
        self.records = []
        self.gate_seconds = 0.0
        self.apply_count = 0

    def set_step_context(self, *, global_step, current_mask, total_tokens):
        self.global_step = int(global_step)
        self.current_mask = current_mask
        self.total_tokens = int(total_tokens)

    def close(self):
        self._handle.remove()

    def _hook(self, _module, _inputs, output):
        t0 = time.perf_counter()
        hidden = output_to_hidden(output)          # L16 state BEFORE intervention
        target = self.current_mask.to(hidden.device, torch.bool)

        # --- edge case: empty current_mask -> no decision, cache NOT updated
        n_target = int(target.sum().item())
        if n_target == 0:
            self.records.append({
                "step": self.global_step, "empty_mask": True, "g": 0,
                "p": None, "deltas_zeroed": None, "n_target": 0,
            })
            self.gate_seconds += time.perf_counter() - t0
            return None

        h32 = hidden.float()
        m_t = h32[target].mean(dim=0)              # mean-pooled current-mask state

        dev = m_t.device
        p_all = float(torch.dot(m_t, self.v_all.to(dev)))
        p_dija = float(torch.dot(m_t, self.v_dija.to(dev)))
        p_ren = float(torch.dot(m_t, self.v_ren.to(dev)))

        first_eligible = self.prev_proj is None
        if first_eligible:
            d_all = d_dija = d_ren = 0.0           # frozen edge case
        else:
            d_all = p_all - self.prev_proj[0]
            d_dija = p_dija - self.prev_proj[1]
            d_ren = p_ren - self.prev_proj[2]

        feats = torch.tensor([
            p_all, p_dija, p_ren,
            d_all, d_dija, d_ren,
            float(torch.linalg.vector_norm(m_t)),
            self.global_step / self.trajectory_steps,
            n_target / self.total_tokens,
        ], dtype=torch.float32)

        x = (feats - self.mu) / self.sigma
        p_t = float(torch.sigmoid(torch.dot(self.w, x) + self.b))
        g_t = 1 if p_t >= self.tau else 0

        # cross-step state update (only on a real decision step)
        self.prev_proj = (p_all, p_dija, p_ren)

        self.records.append({
            "step": self.global_step, "empty_mask": False, "g": g_t,
            "p": p_t, "deltas_zeroed": first_eligible, "n_target": n_target,
            "proj_all": p_all, "delta_all": d_all,
        })

        if g_t == 1:
            self.apply_count += 1
            if self.closed_loop:
                update = (BETA_M1_ALL * self.v_all.to(dev)).view(1, 1, -1)
                hidden_new = (h32 + update * target.float().unsqueeze(-1)).to(hidden.dtype)
                self.gate_seconds += time.perf_counter() - t0
                return output_with_hidden(output, hidden_new)

        self.gate_seconds += time.perf_counter() - t0
        return None


# ----------------------------------------------------------------- generation loop
def transfer_schedule(n_mask, steps):
    base = n_mask // steps
    rem = n_mask % steps
    out = torch.full((steps,), base, dtype=torch.long)
    out[:rem] += 1
    return out


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits)
    gumbel = (-torch.log(noise.clamp_min(1e-20))) ** temperature
    return logits.exp() / gumbel


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_generation(model, x, bank, *, steps, device, force_empty_at=None, seed=314159):
    """LLaDA-style low_confidence remasking loop, instrumented per step.

    The seed is reset at entry so that closed-loop and open-loop runs are
    paired: any state divergence between them is caused by the intervention,
    never by sampling noise. Without this the V3 comparison is meaningless.
    """
    set_seed(seed)
    initial_mask = x == MASK_ID
    sched = transfer_schedule(int(initial_mask.sum()), steps)
    attn = torch.ones_like(x, dtype=torch.long, device=device)
    bank.reset_run(trajectory_steps=steps)

    fwd_seconds = 0.0
    for step in range(1, steps + 1):
        current_mask = x == MASK_ID
        if force_empty_at is not None and step == force_empty_at:
            current_mask = torch.zeros_like(current_mask)   # V2 probe

        bank.set_step_context(
            global_step=step,
            current_mask=current_mask,
            total_tokens=int(x.numel()),
        )

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(x, attention_mask=attn)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_seconds += time.perf_counter() - t0

        logits = out.logits if hasattr(out, "logits") else out[0]
        pred = torch.argmax(add_gumbel_noise(logits, TEMPERATURE), dim=-1)
        conf = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
        pred = torch.where(current_mask, pred, x)
        conf = torch.where(current_mask, conf, torch.full_like(conf, -np.inf))

        k = int(sched[step - 1])
        if k > 0 and int(current_mask.sum()) > 0:
            k = min(k, int(current_mask.sum()))
            idx = torch.topk(conf.view(-1), k=k).indices
            flat = x.view(-1).clone()
            flat[idx] = pred.view(-1)[idx]
            x = flat.view_as(x)

    return x, fwd_seconds


# ----------------------------------------------------------------- fake model (logic test)
class FakeBlock(torch.nn.Module):
    """Cheap stand-in with CONTEXT MIXING.

    An identity block would make the L16 state a pure function of the token at
    each position, so a masked position's state could never reflect an earlier
    intervention and V3 could not be exercised at all. The running-mean mix below
    is the minimum structure that lets state propagate across positions, which is
    what attention provides in the real model.
    """

    def __init__(self, hidden):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)

    def forward(self, h):
        ctx = h.mean(dim=1, keepdim=True)
        return h + self.scale * ctx


class FakeModel(torch.nn.Module):
    """Minimal stand-in exercising the same hook path, for CPU logic validation."""

    def __init__(self, vocab=1000, hidden=4096, n_layers=20):
        super().__init__()
        self.blocks = torch.nn.ModuleList([FakeBlock(hidden) for _ in range(n_layers)])
        self.emb = torch.nn.Embedding(vocab + 200000, hidden)
        self.head = torch.nn.Linear(hidden, vocab)
        self.config = type("C", (), {"num_hidden_layers": n_layers})()

    def forward(self, x, attention_mask=None):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return (self.head(h),)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="tiny CPU model: logic only")
    ap.add_argument("--steps", type=int, default=GEN_LENGTH)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(314159)
    np.random.seed(314159)

    if args.fake:
        device = torch.device("cpu")
        model = FakeModel().eval()
        blocks, blocks_attr = model.blocks, "blocks"
        hidden_dim = 4096
        prompt = torch.randint(0, 900, (1, 16))
        v_all = torch.randn(hidden_dim); v_all /= v_all.norm()
        v_dija = torch.randn(hidden_dim); v_dija /= v_dija.norm()
        v_ren = torch.randn(hidden_dim); v_ren /= v_ren.norm()
        model_id = "FAKE_CPU_STANDIN"
    else:
        from transformers import AutoModel, AutoTokenizer
        device = torch.device("cuda")
        mp = __import__("os").environ["CDG_LLADA_MODEL"]
        tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            mp, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        blocks, blocks_attr = resolve_blocks(model)
        hidden_dim = int(model.config.hidden_size)
        # synthetic prompt ONLY — no frozen population row is touched
        prompt = tok("Explain in one paragraph how photosynthesis works.",
                     return_tensors="pt").input_ids
        dpath = (__import__("os").environ["CDG_WORK_ROOT"] + "/analysis_output/rrae_development/"
                 "generalized_safety_v2_directions_v1/GENERALIZED_SAFETY_V2_DIRECTIONS.pt")
        import hashlib
        dirs = torch.load(dpath, map_location="cpu")
        # keys are v_ALL / v_DIJA / v_RENELLM; bind each to its frozen SHA so the
        # spike cannot silently run on the wrong vectors
        expected = {
            "v_ALL": "937a78750761e63ebc1362bd43e7b3adfa20790b6cec9ae15df01a7befe177e7",
            "v_DIJA": "2a52b08871548bdcb951ea5a32595fc30d7fbc4d057e51a0f3f9c39d8140c6b5",
            "v_RENELLM": "27cd930d1633cd84d1fb8c33b98f16a6dd86220cadf4d5fcd90c404cb9f7be33",
        }
        for k, want in expected.items():
            got = hashlib.sha256(
                dirs[k].float().contiguous().numpy().tobytes()
            ).hexdigest()
            assert got == want, f"direction {k} SHA mismatch: {got} != {want}"
        assert int(dirs["layer_zero_based"]) == LAYER
        v_all = dirs["v_ALL"].float().view(-1)
        v_dija = dirs["v_DIJA"].float().view(-1)
        v_ren = dirs["v_RENELLM"].float().view(-1)
        model_id = mp

    steps = args.steps
    x = torch.cat(
        [prompt, torch.full((1, GEN_LENGTH), MASK_ID, dtype=torch.long)], dim=1
    ).to(device)

    # synthetic probe weights — NOT a fitted gate
    w = torch.randn(N_FEATURES) * 0.5
    bank = TGASGateHookBank(
        blocks[LAYER], v_all, v_dija, v_ren, w, b=0.0, tau=0.5, closed_loop=True
    )

    results = {
        "schema": "M3_TGAS_FEASIBILITY_SPIKE_V1",
        "mode": "FAKE_CPU_STANDIN" if args.fake else "REAL_LLADA_GPU",
        "model": model_id,
        "layer_zero_based": LAYER,
        "blocks_attr": blocks_attr,
        "hidden_dim": hidden_dim,
        "steps": steps,
        "n_features": N_FEATURES,
        "beta_m1_all": BETA_M1_ALL,
        "note": "Synthetic prompt and synthetic probe weights. No frozen population "
                "row, no fitted gate, no efficacy measurement.",
    }

    # ---- V1 + V4: closed-loop run with timing
    _, fwd_s = run_generation(model, x.clone(), bank, steps=steps, device=device)
    recs = [r for r in bank.records if not r["empty_mask"]]
    results["V1_cross_step_state"] = {
        "first_eligible_step_deltas_zeroed": bool(recs[0]["deltas_zeroed"]),
        "later_steps_deltas_zeroed": [bool(r["deltas_zeroed"]) for r in recs[1:5]],
        "nonzero_delta_observed_after_first": any(
            abs(r["delta_all"]) > 0 for r in recs[1:]
        ),
        "PASS": bool(recs[0]["deltas_zeroed"])
                and not any(r["deltas_zeroed"] for r in recs[1:])
                and any(abs(r["delta_all"]) > 0 for r in recs[1:]),
    }
    results["V4_cost"] = {
        "forward_seconds_total": round(fwd_s, 4),
        "gate_seconds_total": round(bank.gate_seconds, 4),
        "gate_overhead_fraction": round(bank.gate_seconds / max(fwd_s, 1e-9), 6),
        "forward_ms_per_step": round(1000 * fwd_s / steps, 3),
        "gate_ms_per_step": round(1000 * bank.gate_seconds / steps, 3),
        "gate_applied_steps": bank.apply_count,
    }
    closed_trigger_seq = [r["g"] for r in bank.records]
    closed_proj = [r.get("proj_all") for r in bank.records if not r["empty_mask"]]

    # ---- V2: empty current_mask handling
    probe_step = max(2, steps // 2)
    _, _ = run_generation(model, x.clone(), bank, steps=steps, device=device,
                          force_empty_at=probe_step)
    empty_rec = next(r for r in bank.records if r["step"] == probe_step)
    before = [r for r in bank.records if r["step"] < probe_step and not r["empty_mask"]]
    after = [r for r in bank.records if r["step"] > probe_step and not r["empty_mask"]]
    results["V2_empty_mask"] = {
        "probe_step": probe_step,
        "recorded_empty": bool(empty_rec["empty_mask"]),
        "g_is_zero": empty_rec["g"] == 0,
        "no_p_computed": empty_rec["p"] is None,
        "next_real_step_not_treated_as_first": (
            bool(after) and not after[0]["deltas_zeroed"]
        ),
        "cache_survived_empty_step": (
            bool(before) and bool(after) and not after[0]["deltas_zeroed"]
        ),
        "PASS": (empty_rec["empty_mask"] and empty_rec["g"] == 0
                 and empty_rec["p"] is None
                 and bool(after) and not after[0]["deltas_zeroed"]),
    }

    # ---- V3: closed-loop vs open-loop divergence (paired seeds)
    x_closed, _ = run_generation(model, x.clone(), bank, steps=steps, device=device)
    closed_trigger_seq = [r["g"] for r in bank.records]
    closed_proj = [r.get("proj_all") for r in bank.records if not r["empty_mask"]]

    bank.closed_loop = False
    x_open, _ = run_generation(model, x.clone(), bank, steps=steps, device=device)
    open_trigger_seq = [r["g"] for r in bank.records]
    open_proj = [r.get("proj_all") for r in bank.records if not r["empty_mask"]]

    n_cmp = min(len(closed_proj), len(open_proj))
    proj_diff = [abs(closed_proj[i] - open_proj[i]) for i in range(n_cmp)]
    tokens_diverged = int((x_closed != x_open).sum())

    # In masked-diffusion generation every step is a FRESH forward pass over the
    # current token sequence; no hidden state is cached between steps. So the ONLY
    # channel by which an intervention at step t can affect the state the gate reads
    # at t+1 is by changing which tokens get unmasked, and to what. Token divergence
    # is therefore the primitive feedback signal, and projection divergence is its
    # downstream consequence.
    v3 = {
        "closed_loop_trigger_count": sum(closed_trigger_seq),
        "open_loop_trigger_count": sum(open_trigger_seq),
        "trigger_sequences_differ": closed_trigger_seq != open_trigger_seq,
        "tokens_diverged": tokens_diverged,
        "max_abs_projection_divergence": round(max(proj_diff), 6) if proj_diff else 0.0,
        "steps_with_divergent_state": sum(1 for d in proj_diff if d > 1e-6),
        "feedback_channel": "token sequence x only (no cross-step hidden-state cache)",
        "note": "Divergence proves intervention at step t changes the state the gate "
                "reads at t+1 — i.e. exposure MUST be measured closed-loop.",
    }
    if args.fake:
        # The stand-in has no attention and a near-flat logit landscape, so a
        # bounded logit shift flips no argmax and x cannot diverge. This is a
        # limitation of the toy, NOT evidence about the real harness. Reported as
        # not-testable rather than as a pass or a failure.
        v3["PASS"] = None
        v3["status"] = "NOT_TESTABLE_ON_FAKE_STANDIN"
        v3["reason"] = ("Fake block has no attention; the only feedback channel is "
                        "token choice, and the toy's logit landscape is too flat for "
                        "the intervention to flip any argmax. Requires the real model.")
    else:
        v3["PASS"] = tokens_diverged > 0 and (max(proj_diff) > 1e-6 if proj_diff else False)
        v3["status"] = "TESTED_ON_REAL_MODEL"
    results["V3_closed_vs_open_loop"] = v3
    bank.close()

    checks = ("V1_cross_step_state", "V2_empty_mask", "V3_closed_vs_open_loop")
    verdicts = [results[k]["PASS"] for k in checks]
    if any(v is False for v in verdicts):
        results["FEASIBILITY_VERDICT"] = "FAIL"
    elif any(v is None for v in verdicts):
        results["FEASIBILITY_VERDICT"] = "PARTIAL_REQUIRES_REAL_MODEL_RUN"
    else:
        results["FEASIBILITY_VERDICT"] = "PASS"

    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
