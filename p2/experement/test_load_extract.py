"""test_hidden_states.py - Standalone test for LLaDA hidden-state + logits extraction.

Validates that StepActivationCollector correctly captures:
  - hidden states from selected transformer blocks at every denoising step
    (ANSWER TOKENS ONLY -- prompt positions are sliced off before saving)
  - top-K logits at every denoising step (answer tokens only;
    full vocab ~126k is too large; we keep top-2048 values+indices)

LLaDA model structure (from modeling_llada.py):
  model                        <- LLaDAModelLM (HF wrapper)
  model.model                  <- LLaDAModel
  model.model.transformer.blocks[i]   <- LLaDALlamaBlock (32 total for 8B)
  model.model.transformer.wte  <- embedding / lm_head (weight-tied)
  model.model.transformer.ln_f <- final layer norm (applied BEFORE logits)

Because logits are computed as F.linear(ln_f(x), wte.weight) inside
LLaDAModel.forward(), there is NO separate lm_head module to hook.
We capture logits by passing lm_head=None to StepActivationCollector and
instead storing them directly from the generate loop's `logits` variable.

Usage:
    python test_hidden_states.py

Output layout:
    hidden_states_test/
      test_traj/
        turn_01.pt   <- {layers, hidden [S,L,A,D], topk_values [S,A,K],
                          topk_indices [S,A,K], prompt_len int}
        summary.txt
"""

from __future__ import annotations
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Top-K logits to save (per token position, per step).
# Full vocab is ~126k tokens -- way too large to store per step.
# 2048 covers any reasonable downstream use (probe training, beam analysis).
# ---------------------------------------------------------------------------
TOPK_LOGITS = 2048


# ---------------------------------------------------------------------------
# StepActivationCollector
# ---------------------------------------------------------------------------

class StepActivationCollector:
    """Forward-hook collector. blocks = LLaDALlamaBlock instances.
    lm_head is NOT used for LLaDA (weight-tied); pass None always.

    Only answer token positions are kept:
      call new_turn(prompt_len=N) to set the slice offset.
    """

    def __init__(self, blocks, layer_ids, lm_head=None):
        self.layer_ids = list(layer_ids)
        self._handles = []
        self._step_buffers = []       # per step: [n_layers, A, D]
        self._cur_hidden = {}
        self._topk_val_buffers = []   # per step: [A, K]
        self._topk_idx_buffers = []   # per step: [A, K]
        self._prompt_len = 0          # set via new_turn()
        self._has_lm_head = False

        for pos, block in enumerate(blocks):
            self._handles.append(
                block.register_forward_hook(self._make_hidden_hook(pos))
            )

    def _make_hidden_hook(self, pos: int):
        def _hook(_module, _inp, out):
            # LLaDALlamaBlock.forward() returns (x, cache) tuple; x is [B, T, D]
            h = out[0] if isinstance(out, (tuple, list)) else out
            # Slice answer tokens only: [:, prompt_len:, :]
            h_ans = h.detach()[0, self._prompt_len:, :].to(torch.float16).cpu()
            self._cur_hidden[pos] = h_ans   # [A, D]
        return _hook

    def new_turn(self, prompt_len: int = 0):
        self._step_buffers = []
        self._cur_hidden = {}
        self._topk_val_buffers = []
        self._topk_idx_buffers = []
        self._prompt_len = prompt_len

    def mark_step(self, logits_this_step: torch.Tensor | None = None):
        """Call once per denoising step AFTER the forward pass.

        Args:
            logits_this_step: raw logits [B, T, V] from this step.
        """
        if not self._cur_hidden:
            return
        ordered = [self._cur_hidden[p] for p in range(len(self.layer_ids))]
        self._step_buffers.append(torch.stack(ordered))  # [L, A, D]
        self._cur_hidden = {}

        if logits_this_step is not None:
            # Slice answer tokens: [B, T, V] -> [A, V]
            ans_logits = logits_this_step.detach()[0, self._prompt_len:, :].float().cpu()
            k = min(TOPK_LOGITS, ans_logits.shape[-1])
            vals, idxs = torch.topk(ans_logits, k=k, dim=-1)  # [A, K]
            self._topk_val_buffers.append(vals.to(torch.float16))
            self._topk_idx_buffers.append(idxs.to(torch.int32))

    def stack_hidden(self):
        """Returns [S, L, A, D] or empty tensor."""
        if not self._step_buffers:
            return torch.empty(0)
        return torch.stack(self._step_buffers)   # [S, L, A, D]

    def stack_topk(self):
        """Returns (values [S,A,K], indices [S,A,K]) or (None, None)."""
        if not self._topk_val_buffers:
            return None, None
        return (
            torch.stack(self._topk_val_buffers),   # [S, A, K]
            torch.stack(self._topk_idx_buffers),   # [S, A, K]
        )

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---------------------------------------------------------------------------
# Modified generate() with collector integration.
# ---------------------------------------------------------------------------

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps,
                    device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_with_capture(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    collector: StepActivationCollector | None = None,
):
    """Identical to the reference generate(), with optional collector hooks."""
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id, dtype=torch.long
    ).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((prompt.shape[0], gen_length),
                        dtype=attention_mask.dtype, device=model.device)],
            dim=-1,
        )

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length:
                  prompt.shape[1] + (num_block + 1) * block_length] == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat(
                        [attention_mask, attention_mask], dim=0
                    )
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            # --- collector: capture logits for this step ---
            if collector is not None:
                collector.mark_step(logits_this_step=logits)
            # -----------------------------------------------

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = -torch.inf
                logits_with_noise[:, :, 126348] = -torch.inf

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

EOS_ID = 126081  # LLaDA EOS token id


def get_actual_len(out: torch.Tensor, prompt_len: int) -> int:
    """Return number of real answer tokens (up to but not including first EOS).

    Args:
        out:        full output tensor [1, prompt_len + gen_length]
        prompt_len: number of prompt tokens
    Returns:
        actual_len: int, 1 <= actual_len <= gen_length
    """
    ans = out[0, prompt_len:]          # [gen_length]
    eos_positions = (ans == EOS_ID).nonzero(as_tuple=True)[0]
    if len(eos_positions) == 0:
        return ans.shape[0]            # no EOS found, keep everything
    return max(1, eos_positions[0].item())  # at least 1 token


def save_turn(
    collector: StepActivationCollector,
    *,
    out_dir: str,
    traj_id: str,
    turn: int,
    prompt_len: int,
    actual_len: int,                   # real answer length after EOS truncation
) -> str:
    """Save hidden states + top-K logits for one turn, truncated to actual_len.

    hidden    : [S, L, actual_len, D]
    topk_*    : [S, actual_len, K]
    """
    h = collector.stack_hidden()
    if h.numel() == 0:
        return ""
    topk_vals, topk_idxs = collector.stack_topk()

    # Truncate answer dimension to actual_len
    h = h[:, :, :actual_len, :]                          # [S, L, A', D]
    if topk_vals is not None:
        topk_vals = topk_vals[:, :actual_len, :]         # [S, A', K]
        topk_idxs = topk_idxs[:, :actual_len, :]         # [S, A', K]

    path = os.path.join(out_dir, traj_id, f"turn_{turn:02d}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "layers":       collector.layer_ids,
            "hidden":       h,           # [S, L, actual_len, D]
            "topk_values":  topk_vals,   # [S, actual_len, K] fp16, or None
            "topk_indices": topk_idxs,   # [S, actual_len, K] int32, or None
            "prompt_len":   prompt_len,
            "actual_len":   actual_len,
            "topk_k":       TOPK_LOGITS,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Verification helper  (BUG FIXED: was calling float() on a list repr string)
# ---------------------------------------------------------------------------

def verify_payload(path: str) -> dict:
    """Load a saved .pt file and return a shape/sanity report dict."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    h = payload["hidden"]          # [S, L, A, D]
    tv = payload["topk_values"]    # [S, A, K] or None
    ti = payload["topk_indices"]   # [S, A, K] or None

    report = {
        "path":            path,
        "layers":          payload["layers"],
        "prompt_len":      payload["prompt_len"],
        "actual_len":      payload.get("actual_len"),
        "topk_k":          payload.get("topk_k"),
        "hidden_shape":    tuple(h.shape),
        "hidden_dtype":    str(h.dtype),
        "hidden_has_nan":  bool(torch.isnan(h).any()),
        "hidden_has_inf":  bool(torch.isinf(h).any()),
        "hidden_norm_mean": float(h.float().norm(dim=-1).mean().item()),
        "topk_shape":      tuple(tv.shape) if tv is not None else None,
        "topk_has_nan":    bool(torch.isnan(tv).any()) if tv is not None else None,
    }

    if tv is not None:
        # tv is top-K log-softmax-able: compute entropy approx over top-K
        # (lower bound; missing mass from tail not modelled)
        probs_topk = F.softmax(tv.float(), dim=-1)   # [S, A, K]
        entropy_topk = -(probs_topk * probs_topk.clamp(min=1e-12).log()).sum(-1)  # [S, A]
        per_step = entropy_topk.mean(dim=-1)   # [S]  -- mean over answer tokens
        report["entropy_per_step_mean"] = per_step.mean().item()   # scalar  ← BUG FIX
        report["entropy_per_step_list"] = [round(v, 4) for v in per_step.tolist()]
        report["entropy_shape"] = tuple(entropy_topk.shape)

    return report


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    device = "cuda"
    out_dir = "hidden_states_test"
    traj_id = "test_traj"

    # ---- layer selection ----
    # 9 layers evenly spread across the 32-layer 8B model.
    # Early layers catch surface/syntactic features; late layers catch semantics.
    SELECTED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    print("Loading model...")
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    # ---- wire up collector ----
    blocks = [model.model.transformer.blocks[i] for i in SELECTED_LAYERS]
    collector = StepActivationCollector(
        blocks=blocks,
        layer_ids=SELECTED_LAYERS,
        lm_head=None,   # LLaDA: weight-tied, no separate lm_head module
    )

    # ---- test prompts ----
    prompts_raw = [
        "What is the capital of France?",
        "What is 2 + 2?",
    ]

    gen_length   = 64
    steps        = 64
    block_length = 32

    saved_paths = []
    prompt = None

    for turn_idx, user_input in enumerate(prompts_raw, start=1):
        print(f"\n=== Turn {turn_idx}: {user_input!r} ===")

        m = [{"role": "user", "content": user_input}]
        templated = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(templated)["input_ids"]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if turn_idx == 1:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)

        prompt_len = prompt.shape[1]   # number of prompt tokens this turn

        # Reset collector buffers; tell it where answer tokens start
        collector.new_turn(prompt_len=prompt_len)

        out = generate_with_capture(
            model, prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            collector=collector,
        )

        answer = tokenizer.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )[0]
        print(f"  Reply: {answer!r}")

        actual_len = get_actual_len(out, prompt_len)
        print(f"  Actual answer tokens (EOS-truncated): {actual_len} / {gen_length}")

        path = save_turn(
            collector,
            out_dir=out_dir,
            traj_id=traj_id,
            turn=turn_idx,
            prompt_len=prompt_len,
            actual_len=actual_len,
        )
        saved_paths.append(path)
        print(f"  Saved: {path}")

        # Update prompt (strip EOS, matching chat.py pattern)
        prompt = out[out != 126081].unsqueeze(0)

    # ---- verification ----
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)

    all_ok = True
    summary_lines = []

    # steps_per_block = steps / num_blocks
    expected_steps = steps // (gen_length // block_length)

    for path in saved_paths:
        if not path:
            print("  [WARN] empty path, skipping")
            continue
        r = verify_payload(path)
        S, L, A, D = r["hidden_shape"]
        ok = (
            not r["hidden_has_nan"]
            and not r["hidden_has_inf"]
            and S == steps
            and L == len(SELECTED_LAYERS)
            and A == r["actual_len"]      # answer tokens == EOS-truncated length
        )
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False

        entropy_str = (
            f"{r['entropy_per_step_mean']:.4f} (mean over steps)"
            if "entropy_per_step_mean" in r else "N/A"
        )

        line = (
            f"  [{status}] {os.path.basename(path)}\n"
            f"         hidden : {r['hidden_shape']} {r['hidden_dtype']}"
            f"  nan={r['hidden_has_nan']}  inf={r['hidden_has_inf']}\n"
            f"         topk   : {r['topk_shape']} (K={r['topk_k']})"
            f"  nan={r['topk_has_nan']}\n"
            f"         layers : {r['layers']}\n"
            f"         prompt_len: {r['prompt_len']}  actual_len: {r['actual_len']}\n"
            f"         hidden_norm_mean : {r['hidden_norm_mean']:.4f}\n"
            f"         entropy (topK approx): {entropy_str}"
        )
        print(line)
        summary_lines.append(line)

    print(f"\n  Expected n_steps per file : {expected_steps}"
          f"  (steps={steps}, num_blocks={gen_length // block_length})")
    print(f"  TOPK_LOGITS stored        : {TOPK_LOGITS}")

    # Write summary
    summary_path = os.path.join(out_dir, traj_id, "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
        f.write(f"\n\nExpected steps per file : {expected_steps}\n")
        f.write(f"Selected layers         : {SELECTED_LAYERS}\n")
        f.write(f"TOPK_LOGITS             : {TOPK_LOGITS}\n")
        f.write(f"Overall: {'PASS' if all_ok else 'FAIL'}\n")

    print(f"\nSummary written to {summary_path}")
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")

    collector.close()
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)