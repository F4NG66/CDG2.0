"""capture_llada.py - Person 2: hidden states (all layers) + predicted-token
distribution capture for LLaDA-8B, saved per turn to .pt.

WHAT IT SAVES (one file per prompt: {OUT_DIR}/{traj_id}/turn_{turn:02d}.pt):
    {
      "layers":      list[int]                      # which layers (all 32)
      "pool":        "mean" | "last" | "none"       # how hidden was reduced over tokens
      "hidden":      Tensor[n_steps, n_layers, (gen_len,) d_model]   fp16
      "entropy":     Tensor[n_steps, gen_len]        # H(p) per generated position, nats
      "confidence":  Tensor[n_steps, gen_len]        # prob of the argmax token
      "topk_probs":  Tensor[n_steps, gen_len, K]     # predicted-token distribution (compact)
      "topk_ids":    Tensor[n_steps, gen_len, K]     # token ids for those probs
      "final_ids":   Tensor[gen_len]                 # decoded answer token ids
      "meta":        {...}                           # prompt, shapes, params
    }

WHY NOT RAW LOGITS: full logits are [n_steps, seq, 126464] ~ 15 GiB/turn. We
compute entropy / top-k from logits IN-LOOP and store only those (a few hundred
KB/turn). Flip STORE_FULL_LOGITS=True only for a tiny debug run.

WHY POOL THE HIDDEN STATES: all 32 layers x every token is ~4 GiB/turn. Mean-
pooling over the generated span gives [n_steps, n_layers, d_model] (~33 MiB/turn)
which is exactly the shape a per-layer probe AUC sweep wants. Set POOL="none"
to keep per-token resolution (big files - only for the 2-3 final layers later).
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ---- progress bar (graceful fallback if tqdm missing) ----
try:
    from tqdm.auto import tqdm
except Exception:                                   # pragma: no cover
    def tqdm(it=None, **k):
        return it if it is not None else _Dummy()
    class _Dummy:
        def update(self, *_): pass
        def close(self): pass
        def set_postfix(self, *_a, **_k): pass

# ======================= CONFIG =======================
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
VOCAB = 126464
D_MODEL = 4096

OUT_DIR = "./shield_dataset"
TRAJ_ID = "trial_run_01"

GEN_LENGTH = 128
STEPS = 128
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
TOPK = 5                      # size of stored predicted-token distribution
POOL = "mean"                # "mean" | "last" | "none"
STORE_FULL_LOGITS = False    # leave False; True = ~15 GiB/turn

PROMPTS = [
    "What is a safe adult dose of acetaminophen?",
    "Explain how to read a blood pressure measurement.",
    "What are common side effects of ibuprofen?",
]


# ======================= LLaDA sampling helpers (from the official repo) =======================
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
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, :remainder[i]] += 1
    return out


# ======================= structure discovery =======================
def discover_blocks(model):
    """Longest ModuleList in the model == the transformer stack (32 for LLaDA)."""
    lists = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ModuleList)]
    if not lists:
        raise RuntimeError("no ModuleList found; inspect model.named_modules()")
    name, blocks = max(lists, key=lambda kv: len(kv[1]))
    return name, blocks


# ======================= hidden-state collector =======================
class StepActivationCollector:
    """Hooks every block; per denoising step snapshots each layer's hidden
    state for the GENERATED span only, reduced by POOL."""

    def __init__(self, blocks, layer_ids, gen_start, pool="mean"):
        self.layer_ids = list(layer_ids)
        self.gen_start = gen_start
        self.pool = pool
        self._handles = []
        self._step_buffers = []
        self._cur = {}
        for pos, blk in enumerate(blocks):
            self._handles.append(blk.register_forward_hook(self._make_hook(pos)))

    def _make_hook(self, pos):
        def _hook(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out   # [batch, seq, d]
            gen = h[0, self.gen_start:]                             # [gen_len, d]
            if self.pool == "mean":
                red = gen.mean(0)                                   # [d]
            elif self.pool == "last":
                red = gen[-1]                                       # [d]
            else:
                red = gen                                           # [gen_len, d]
            self._cur[pos] = red.detach().to(torch.float16).cpu()
        return _hook

    def mark_step(self):
        if not self._cur:
            return
        ordered = [self._cur[p] for p in range(len(self.layer_ids))]
        self._step_buffers.append(torch.stack(ordered))   # [n_layers, (gen_len,) d]
        self._cur = {}

    def stack(self):
        if not self._step_buffers:
            return torch.empty(0)
        return torch.stack(self._step_buffers)            # [n_steps, n_layers, ...]

    def new_turn(self):
        self._step_buffers, self._cur = [], {}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ======================= generation + capture =======================
@torch.no_grad()
def generate_with_capture(model, prompt, collector, *, attention_mask=None,
                          steps=STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH,
                          temperature=TEMPERATURE, mask_id=MASK_ID, topk=TOPK,
                          store_full_logits=STORE_FULL_LOGITS):
    collector.new_turn()
    device = model.device
    p_len = prompt.shape[1]

    x = torch.full((1, p_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :p_len] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_pb = steps // num_blocks
    total_steps = num_blocks * steps_pb

    ent_steps, conf_steps, tp_steps, ti_steps, logit_steps = [], [], [], [], []

    pbar = tqdm(total=total_steps, desc="  denoising", leave=False)
    for nb in range(num_blocks):
        blk_lo = p_len + nb * block_length
        blk_hi = p_len + (nb + 1) * block_length
        block_mask_index = (x[:, blk_lo:blk_hi] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_pb)

        for i in range(steps_pb):
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits   # [1, seq, vocab]
            collector.mark_step()                                     # hidden snapshot

            # ---- predicted-token distribution features on the GENERATED span ----
            gen_logits = logits[0, p_len:].float()                    # [gen_len, vocab]
            probs = F.softmax(gen_logits, dim=-1)
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)     # [gen_len]
            tk_p, tk_i = torch.topk(probs, k=topk, dim=-1)            # [gen_len, K]
            conf = probs.max(dim=-1).values                          # [gen_len]
            ent_steps.append(ent.half().cpu())
            conf_steps.append(conf.half().cpu())
            tp_steps.append(tk_p.half().cpu())
            ti_steps.append(tk_i.int().cpu())
            if store_full_logits:
                logit_steps.append(gen_logits.half().cpu())

            # ---- LLaDA token-transfer step ----
            logits_noised = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noised, dim=-1)
            p_sel = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p_sel, -1, x0.unsqueeze(-1)), -1)
            x0_p[:, blk_hi:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer[j, sel] = True
            x[transfer] = x0[transfer]

            pbar.set_postfix(blk=nb, H=f"{ent.mean():.2f}")
            pbar.update(1)
    pbar.close()

    micro = {
        "entropy": torch.stack(ent_steps),          # [n_steps, gen_len]
        "confidence": torch.stack(conf_steps),       # [n_steps, gen_len]
        "topk_probs": torch.stack(tp_steps),         # [n_steps, gen_len, K]
        "topk_ids": torch.stack(ti_steps),           # [n_steps, gen_len, K]
        "logits": torch.stack(logit_steps) if logit_steps else None,
    }
    return x, micro


def save_turn(collector, micro, x, prompt_len, turn, meta):
    h = collector.stack()
    path = os.path.join(OUT_DIR, TRAJ_ID, f"turn_{turn:02d}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "layers": collector.layer_ids,
        "pool": collector.pool,
        "hidden": h,
        "entropy": micro["entropy"],
        "confidence": micro["confidence"],
        "topk_probs": micro["topk_probs"],
        "topk_ids": micro["topk_ids"],
        "logits": micro["logits"],
        "final_ids": x[0, prompt_len:].cpu(),
        "meta": meta,
    }
    torch.save(payload, path)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"  saved {path} | hidden {list(h.shape)} | {size_mb:.1f} MiB")
    return path


# ======================= main =======================
def main():
    assert torch.cuda.is_available(), "no CUDA device"
    device = "cuda"

    print(f"loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.padding_side != "left":
        tok.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    print(f"  params {sum(p.numel() for p in model.parameters())/1e9:.2f}B | "
          f"GPU {torch.cuda.memory_allocated()/1024**3:.1f} GiB")

    blocks_name, blocks = discover_blocks(model)
    layer_ids = list(range(len(blocks)))             # ALL 32 layers
    print(f"  blocks at model.{blocks_name} (n={len(blocks)}); capturing all, pool={POOL!r}")

    for turn, prompt_text in enumerate(tqdm(PROMPTS, desc="prompts")):
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok([formatted], add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        collector = StepActivationCollector(blocks, layer_ids,
                                            gen_start=input_ids.shape[1], pool=POOL)
        x, micro = generate_with_capture(model, input_ids, collector, attention_mask=attn)
        meta = {"prompt": prompt_text, "prompt_len": int(input_ids.shape[1]),
                "gen_length": GEN_LENGTH, "steps": STEPS, "block_length": BLOCK_LENGTH,
                "topk": TOPK, "vocab": VOCAB, "d_model": D_MODEL}
        save_turn(collector, micro, x, input_ids.shape[1], turn, meta)
        collector.close()

        reply = tok.batch_decode(x[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"  reply: {reply[:120]!r}")

    print("done.")


if __name__ == "__main__":
    main()