"""run_llada_prefill.py - Run a prefill set through LLaDA-8B-Instruct.

Reads a prefill JSON produced by make_prefill_sure.py / make_prefill_positive.py
(a list of {id, topic, question, prefix, method}) and, for each item:

  1. Builds a chat-templated prompt from `question`.
  2. Prefills the answer region with `prefix`:
         x = [ prompt | prefix | MASK * (gen_length - len(prefix)) ]
     LLaDA never revises non-[MASK] positions, so the prefix is frozen and only
     the remaining masked tokens are generated (the prefilling attack).
  3. Decodes the model's continuation and writes per-item answers + a summary.

The generation loop is a self-contained copy of the sampler used in
prefill_capture.py / paired_dija_runner.py (no hidden-state capture), so this
file runs standalone.

Output: answers/<method>_answers.json  (default name derived from the input)

Usage:
    python run_llada_prefill.py --prefills prefill_data/sure_prefills.json
    python run_llada_prefill.py --prefills prefill_data/positive_prefills.json
    python run_llada_prefill.py --prefills prefill_data/sure_prefills.json --limit 2
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336


# ===================================================================
# Sampler helpers - identical to generate.py / paired_dija_runner.py
# ===================================================================
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
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps,
                                      device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_prefill(model, prompt, prefix_ids, *, attention_mask=None, steps=128,
                     gen_length=128, block_length=32, temperature=0.0,
                     cfg_scale=0.0, mask_id=MASK_ID):
    """LLaDA denoising with an attacker prefix written into the answer region.

    Mirrors prefill_capture.generate_prefill: the prefix is written right after
    the prompt and prompt_index is computed AFTER the write, so the prefix is
    frozen and never remasked.
    """
    device = model.device
    p_len = prompt.shape[1]
    pre_len = prefix_ids.shape[1]
    if pre_len >= gen_length:
        raise SystemExit(
            f"[fatal] prefix length ({pre_len}) >= gen_length ({gen_length}); "
            f"increase --gen-length or shorten the prefix.")

    # x = [ prompt | prefix | MASK * (gen_length - pre_len) ]
    x = torch.full((1, p_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :p_len] = prompt.clone()
    x[:, p_len:p_len + pre_len] = prefix_ids.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1)

    prompt_index = (x != mask_id)  # prompt AND prefix are frozen

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_pb = steps // num_blocks

    for nb in range(num_blocks):
        blk_lo = p_len + nb * block_length
        blk_hi = p_len + (nb + 1) * block_length
        block_mask_index = (x[:, blk_lo:blk_hi] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_pb)

        for i in range(steps_pb):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                am_ = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
                logits = model(x_, attention_mask=am_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

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
    return x


def load_prefills(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"[fatal] {path} is not a non-empty JSON list")
    return data


def default_out_name(prefill_path: Path, records: list[dict]) -> str:
    method = records[0].get("method") or prefill_path.stem
    return f"{method}_answers.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--prefills", required=True, help="prefill JSON from a make_prefill_* script")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: answers/<method>_answers.json)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=None, help="first N items (smoke test)")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cfg-scale", type=float, default=0.0)
    ap.add_argument("--mask-id", type=int, default=MASK_ID)
    args = ap.parse_args()

    prefill_path = Path(args.prefills)
    records = load_prefills(prefill_path)
    if args.limit:
        records = records[:args.limit]

    out_path = Path(args.out) if args.out else here / "answers" / default_out_name(prefill_path, records)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "no CUDA device visible"
    print(f"[load] {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.padding_side != "left":
        tok.padding_side = "left"
    assert tok.pad_token_id != args.mask_id, "pad_token_id collides with mask_id"
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.all_tied_weights_keys = {}  # transformers-compat patch (see generate.py)
    model = model.to("cuda").eval()

    gen_params = {"steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "temperature": args.temperature,
                  "cfg_scale": args.cfg_scale}

    answers = []
    n = len(records)
    for i, item in enumerate(records, 1):
        question = item["question"]
        prefix_text = item["prefix"]

        messages = [{"role": "user", "content": question}]
        formatted = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tok([formatted], add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        attn = enc["attention_mask"].to("cuda")
        prefix_ids = tok(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")

        t0 = time.time()
        x = generate_prefill(model, input_ids, prefix_ids, attention_mask=attn,
                             steps=args.steps, gen_length=args.gen_length,
                             block_length=args.block_length, temperature=args.temperature,
                             cfg_scale=args.cfg_scale, mask_id=args.mask_id)
        dt = time.time() - t0

        # full_answer = prefix + model continuation; body_answer = model's own part.
        full_answer = tok.batch_decode(x[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        body_answer = tok.decode(
            x[0, input_ids.shape[1] + prefix_ids.shape[1]:], skip_special_tokens=True).strip()

        answers.append({
            "id": item.get("id"),
            "topic": item.get("topic"),
            "method": item.get("method"),
            "question": question,
            "prefix": prefix_text,
            "full_answer": full_answer,
            "body_answer": body_answer,
            "gen_params": gen_params,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"[{i}/{n}] id={item.get('id')} | {dt:.1f}s | {full_answer[:80]!r}")

    out_path.write_text(json.dumps({"answers": answers, "gen_params": gen_params},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {len(answers)} answers -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
