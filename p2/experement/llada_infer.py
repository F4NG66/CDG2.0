"""llada_infer.py - Minimal single-turn inference with LLaDA-8B-Instruct.

Usage:
    python llada_infer.py
    python llada_infer.py --prompt "What is the first-line treatment for type 2 diabetes?"
    python llada_infer.py --prompt "..." --steps 128 --gen_length 128
"""

import argparse
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID    = 126336
EOS_ID     = 126081

# ── Default generation params (consistent with main.py) ─────────────────────
DEFAULT_GEN_LENGTH = 64
DEFAULT_STEPS = 64
DEFAULT_BLOCK_LENGTH = 32



# ── Generate core — lifted verbatim from hidden_states.py ───────────────────
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise  = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)



def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base      = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.repeat(1, steps)
    for i in range(remainder.shape[0]):
        num_transfer_tokens[i, : remainder[i].item()] += 1
    return num_transfer_tokens



@torch.no_grad()
def generate(model, prompt_ids,
             steps=DEFAULT_STEPS,
             gen_length=DEFAULT_GEN_LENGTH,
             block_length=DEFAULT_BLOCK_LENGTH,
             temperature=0.0,
             remasking="low_confidence"):
    import torch.nn.functional as F
    import numpy as np

    device = model.device
    x = torch.full(
        (1, prompt_ids.shape[1] + gen_length),
        MASK_ID, dtype=torch.long, device=device,
    )
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

    prompt_len  = prompt_ids.shape[1]
    num_blocks  = gen_length // block_length
    steps_per_block = steps // num_blocks

    for b in range(num_blocks):
        block_mask = (
            x[:, prompt_len + b * block_length : prompt_len + (b + 1) * block_length]
            == MASK_ID
        )
        n_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == MASK_ID)
            logits     = model(x).logits                    # [1, T, V]

            logits_noisy = add_gumbel_noise(logits, temperature)
            x0           = torch.argmax(logits_noisy, dim=-1)

            if remasking == "low_confidence":
                p    = F.softmax(logits, dim=-1)
                x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                x0_p = torch.rand(x0.shape, device=device)

            x0_p[:, prompt_len + (b + 1) * block_length:] = -np.inf
            x0         = torch.where(mask_index, x0, x)
            confidence  = torch.where(mask_index, x0_p,
                                      torch.tensor(-np.inf, device=device))

            transfer = torch.zeros_like(x0, dtype=torch.bool)
            _, sel   = torch.topk(confidence[0], k=n_transfer[0, i].item())
            transfer[0, sel] = True
            x[transfer] = x0[transfer]

    return x



# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",       default="What is the first-line treatment for type 2 diabetes?")
    parser.add_argument("--steps",        type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length",   type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--block_length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--temperature",  type=float, default=0.0)
    args = parser.parse_args()

    print(f"[load] {MODEL_NAME} ...")
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    messages   = [{"role": "user", "content": args.prompt}]
    templated  = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids  = torch.tensor(
        tokenizer(templated)["input_ids"], dtype=torch.long
    ).unsqueeze(0).to(model.device)

    print(f"[prompt] {args.prompt}")
    print(f"[params] steps={args.steps} gen_length={args.gen_length} "
          f"block_length={args.block_length} temperature={args.temperature}")
    print("[generating ...]")

    out = generate(
        model, input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
    )

    reply = tokenizer.batch_decode(
        out[:, input_ids.shape[1]:], skip_special_tokens=True
    )[0]

    print("\n── Reply ──────────────────────────────────────────")
    print(reply)
    print("───────────────────────────────────────────────────")



if __name__ == "__main__":
    main()