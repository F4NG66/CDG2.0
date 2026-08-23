"""paired_dija_runner.py - Paired benign/harmful LLaDA runner (no hidden states).

For each seed it runs TWO arms through identical LLaDA generation:
    - "benign"  : the seed's `prompt`
    - "harmful" : the seed's `harmful_variant`
Same model, same gen params, same record schema -> each `id` yields a matched
pair differing only in `arm` / text, so per-item comparison is a clean subtraction.

DIJA-style mask handling (process_prompt_instruct):
    - if the text contains <mask:N> markers, each is expanded to N mask tokens
      (the structured scaffold path, used once you add the Qwen refine step);
    - otherwise, with --mask-counts > 0, a blank canvas of that many mask tokens
      is appended in the assistant slot (DIJA's fallback prefill).
Set --mask-counts 0 for a pure no-prefill control.

Hidden-state capture is intentionally omitted here. The sampler below mirrors
the project's generate.py exactly; swap in generate_with_capture from
ladaAndH.py when you want activations.

Self-harm-adjacent items are excluded from the run regardless of input.

Smoke test:
    python paired_dija_runner.py --seeds curated/seeds_curated.jsonl --out runs --limit 2
Full set:
    python paired_dija_runner.py --seeds curated/seeds_curated.jsonl --out runs
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Constants
DEFAULT_GEN_LENGTH = 128
DEFAULT_STEPS = 128
DEFAULT_MASK_ID = 126336
DEFAULT_MASK_COUNTS = 36
MASK_TOKEN = "<|mdm_mask|>"
START_TOKEN = "<startoftext>"
END_TOKEN = "<endoftext>"
SPECIAL_TOKEN_PATTERN = r'<mask:(\d+)>'
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

# Self-harm-adjacent items kept out of the run (different handling track).
DEFAULT_EXCLUDE = {"case_014", "case_036", "case_067"}
ARMS = (("benign", "prompt"), ("harmful", "harmful_variant"))


# ===================================================================
# Sampler — mirrors generate.py (no capture). Kept inline so this file
# runs standalone; behaviour is identical to the project's generate().
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
def generate(model, prompt, attention_mask=None, steps=DEFAULT_STEPS,
             gen_length=DEFAULT_GEN_LENGTH, block_length=32, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=DEFAULT_MASK_ID):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((prompt.shape[0], gen_length),
             dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length:
                              prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                am = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
                logits = model(x_, attention_mask=am).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x


# ===================================================================
# Prompt handling (DIJA process_prompt_instruct) + pure helpers
# ===================================================================
def expand_masks(text: str, mask_token: str = MASK_TOKEN) -> str:
    """Replace numeric placeholders like <mask:N> with N actual victim mask tokens."""
    return re.sub(SPECIAL_TOKEN_PATTERN, lambda m: mask_token * int(m.group(1)), text)


def expected_mask_count(text: str, mask_counts: int) -> int:
    """How many mask tokens SHOULD be in the prompt after processing, derived from
    the <mask:N> markers (or the fallback canvas if there are none)."""
    marker_total = sum(int(n) for n in re.findall(SPECIAL_TOKEN_PATTERN, text))
    if marker_total:
        return marker_total
    return mask_counts  # blank-canvas fallback


def process_prompt_instruct(prompt: str, mask_counts: int) -> str:
    """Expand <mask:N> into real mask tokens; if none present and mask_counts>0,
    append a blank mask canvas (DIJA fallback)."""
    prompt = expand_masks(prompt, MASK_TOKEN)
    if MASK_TOKEN not in prompt and mask_counts:
        prompt += START_TOKEN + MASK_TOKEN * mask_counts + END_TOKEN
    return prompt


def load_seeds(path: str, exclude_ids: set[str]) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["id"] not in exclude_ids]


def build_row(seed: dict, arm: str, text: str, response: str, gen: dict) -> dict:
    return {
        "id": seed["id"],
        "topic": seed.get("topic"),
        "bucket": seed.get("bucket"),     # present only if stratify was run
        "arm": arm,
        "traj_id": f"{seed['id']}__{arm}",
        "prompt_text": text,
        "response": response,
        "gen": gen,
    }


# ===================================================================
# Model execution
# ===================================================================
def generate_response(model, tokenizer, text, device, args) -> str:
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    processed = process_prompt_instruct(formatted, args.mask_counts)

    proc_ids = tokenizer(processed, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    attn = torch.ones_like(proc_ids)
    vanilla_ids = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    # --- verify the expand actually survived into token ids ---------------
    expected = expected_mask_count(formatted, args.mask_counts)
    actual = int((proc_ids == args.mask_id).sum())
    if expected > 0 and actual == 0:
        raise RuntimeError(
            f"expand failed: {expected} mask(s) expected but 0 token(s) == mask_id "
            f"({args.mask_id}). The tokenizer is not mapping {MASK_TOKEN!r} to the mask "
            f"id - check that it is a registered special token. This is the bug that "
            f"produces nonsense fills.")
    if expected != actual:
        print(f"  [warn] mask count mismatch: expected {expected}, got {actual}")
    elif args.debug_masks:
        print(f"  [masks] expected={expected} actual={actual} (expand OK)")

    # Shared-prefix length, so we decode only the model-filled region (DIJA).
    matching_count = min(
        int(sum(a == b for a, b in zip(vanilla_ids[0], proc_ids[0]))),
        len(vanilla_ids[0]))

    out = generate(model, proc_ids, attention_mask=attn,
                   steps=args.steps, gen_length=args.gen_length,
                   block_length=args.block_length, temperature=args.temperature,
                   cfg_scale=args.cfg_scale, remasking=args.remasking, mask_id=args.mask_id)

    response = tokenizer.batch_decode(out[:, matching_count:], skip_special_tokens=True)[0]
    return response.split("assistant\n")[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired benign/harmful LLaDA runner (no hidden states).")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="first N seeds (smoke test)")
    ap.add_argument("--mask-counts", type=int, default=DEFAULT_MASK_COUNTS,
                    help="blank-canvas masks appended when no <mask:N> present (0 = pure control)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--gen-length", type=int, default=DEFAULT_GEN_LENGTH)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)  # DIJA used 0.2; 0.0 = deterministic
    ap.add_argument("--cfg-scale", type=float, default=0.0)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--mask-id", type=int, default=DEFAULT_MASK_ID)
    ap.add_argument("--debug-masks", action="store_true",
                    help="print the expected/actual mask-token count per generation")
    args = ap.parse_args()

    gen = {"steps": args.steps, "gen_length": args.gen_length, "block_length": args.block_length,
           "temperature": args.temperature, "cfg_scale": args.cfg_scale,
           "remasking": args.remasking, "mask_counts": args.mask_counts}

    seeds = load_seeds(args.seeds, DEFAULT_EXCLUDE)
    if args.limit:
        seeds = seeds[:args.limit]
    print(f"loaded {len(seeds)} seeds (excluded {sorted(DEFAULT_EXCLUDE)}); mask_counts={args.mask_counts}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.pad_token_id != args.mask_id, "pad id collides with mask id"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "paired_runs.jsonl"

    n = 0
    with rows_path.open("w", encoding="utf-8") as fout:
        for s in seeds:
            for arm, field in ARMS:
                text = s[field]
                print(f"  [{s['id']}__{arm}] {text[:60]!r}")
                resp = generate_response(model, tokenizer, text, device, args)
                print(f"  {COLOR_BLUE}{resp[:120]!r}{COLOR_RESET}")
                fout.write(json.dumps(build_row(s, arm, text, resp, gen), ensure_ascii=False) + "\n")
                n += 1

    print(f"\nwrote {n} rows ({len(seeds)} seeds x 2 arms) -> {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())