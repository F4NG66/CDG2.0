#!/usr/bin/env python3
"""
execute_latent_clock_attack.py
Optimized mask-targeted latent clock hijack on LLaDA-8B-Instruct.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==========================================
# CONFIGURATION & INFRASTRUCTURE CONSTANTS
# ==========================================
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
DATASET_PATH = "/home/ore99/serverFiles/attack2/source_A/cases.json"
OUTPUT_PATH = "paired_runs.jsonl"

MASK_ID = 126336
VOCAB = 126464
D_MODEL = 4096
STEPS = 128
GEN_LENGTH = 128
BLOCK_LENGTH = 32
TEMPERATURE = 0.0

# Aligned with Intervention II baseline configurations from main workspace
TARGET_LAYER = 29       # Target late-stage blocks where correction ceilings fail
STEERING_ALPHA = 64.0   # Scale aligned with successful activation steering

# ==========================================
# LLaDA NATIVE SAMPLING OPERATORS
# ==========================================
def add_gumbel_noise(logits, temperature):
    if temperature == 0: 
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, :remainder[i]] += 1
    return out

# ==========================================
# PRECISE INTERVENTION HOOK MANAGER
# ==========================================
class PreciseMaskHijackHook:
    """
    Intervenes exclusively on the active MASK tokens inside the residual stream,
    preventing the model from mapping them into the safe manifold while keeping
    the base behavior prompt tokens intact.
    """
    def __init__(self, block_module: nn.Module, d_model: int, alpha: float):
        self.alpha = alpha
        # Formulate a structured directional vector aligned to the hidden layer basis
        torch.manual_seed(1337)
        base_direction = torch.randn(d_model, dtype=torch.bfloat16)
        self.v_shared = (base_direction / torch.norm(base_direction)).cuda()
        self.current_mask_indices = None
        self.handle = block_module.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, args):
        hidden_states = args[0] # [batch, seq_len, d_model]
        if self.current_mask_indices is not None:
            # Mask-only positioning: isolates the injection away from benign prompt tokens
            # This directly maps to your baseline config (template/mask positions)
            mask = self.current_mask_indices.unsqueeze(-1) # [1, seq_len, 1]
            hidden_states = torch.where(mask, hidden_states + (self.alpha * self.v_shared), hidden_states)
        return (hidden_states,)

    def remove(self):
        self.handle.remove()

# ==========================================
# GENERATION PIPELINE WITH STEERING
# ==========================================
@torch.no_grad()
def steer_generation(model, prompt_ids, attention_mask, hook, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH, temperature=TEMPERATURE, mask_id=MASK_ID, steps=STEPS):
    device = model.device
    p_len = prompt_ids.shape[1]

    x = torch.full((1, p_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :p_len] = prompt_ids.clone()
    
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    num_blocks = gen_length // block_length
    steps_pb = steps // num_blocks

    for nb in range(num_blocks):
        blk_lo = p_len + nb * block_length
        blk_hi = p_len + (nb + 1) * block_length
        block_mask_index = (x[:, blk_lo:blk_hi] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_pb)

        for i in range(steps_pb):
            # Dynamic state tracking: notify the layer hook exactly where active masks live
            hook.current_mask_indices = (x == mask_id)
            
            logits = model(x, attention_mask=attention_mask).logits

            logits_noised = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noised, dim=-1)
            p_sel = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p_sel, -1, x0.unsqueeze(-1)), -1)
            x0_p[:, blk_hi:] = -np.inf
            x0 = torch.where(hook.current_mask_indices, x0, x)
            confidence = torch.where(hook.current_mask_indices, x0_p, torch.full_like(x0_p, -np.inf))

            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer[j, sel] = True
            x[transfer] = x0[transfer]

    return x[:, p_len:]

# ==========================================
# MAIN RUN EXECUTION
# ==========================================
def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing behavior target records at: {DATASET_PATH}")

    print(f"[*] Parsing behavior dataset records from: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"[*] Instantiating model environment: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.padding_side != "left": 
        tok.padding_side = "left"
        
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

    lists = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ModuleList)]
    blocks_name, blocks = max(lists, key=lambda kv: len(kv[1]))
    print(f"[✓] Structural stack discovered at 'model.{blocks_name}' (n={len(blocks)} layers)")

    # Establish the precise dynamic mask tracker hook
    hijack_hook = PreciseMaskHijackHook(blocks[TARGET_LAYER], d_model=D_MODEL, alpha=STEERING_ALPHA)
    print(f"[*] Strategic Hook Active: Layer {TARGET_LAYER} mask-targeted stream biased with alpha={STEERING_ALPHA}")

    out_records = []
    print(f"[*] Simulating parallel diffusion steps across {len(cases)} items...")
    
    for item in tqdm(cases, desc="Attacking items"):
        behavior = item.get("behavior", "")
        if not behavior and "behavior" in item.get("user_content", ""):
            behavior = item.get("user_content", "")
            
        messages = [{"role": "user", "content": behavior}]
        formatted = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok([formatted], add_special_tokens=False, return_tensors="pt")
        
        input_ids = enc["input_ids"].cuda()
        attn = enc["attention_mask"].cuda()

        # Run targeted generation loop containing mask routing
        response_tokens = steer_generation(model, input_ids, attn, hijack_hook)
        reply_text = tok.batch_decode(response_tokens, skip_special_tokens=True)[0]

        out_records.append({
            "traj_id": f"clock_dilution_{item['id']}",
            "prompt_text": behavior,
            "response": reply_text,
            "arm": "harmful",
            "attack_type": "latent_clock_hijack"
        })

    hijack_hook.remove()
    print(f"[*] Hook disconnected gracefully. Writing results out to: {OUTPUT_PATH}")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[✓] Execution run complete. Run your DeepSeek judge next.")


if __name__ == "__main__":
    main()
