#!/usr/bin/env python3
"""PHASE 1 step 4 - teacher-forced hidden-state capture (GPU job).

For every pair, push [prompt + response] through LLaDA in ONE forward pass with
no masking and mean-pool block outputs over the RESPONSE tokens only. Done for
the harm response and the safe response of every case.

Two prompt layouts are captured because the right one is not obvious a priori:

  bare : user turn = the bare behavior ("How can I ...")
         -> the literal reading of "same prompt, different response"; cleanest
  dija : user turn = the full DIJA scaffold with <mask:N> expanded to mask tokens
         -> what the model actually sees at steering time, so a direction built
            here should transfer better to Phase 3

Both are cheap (2 layouts x 2 sides x n cases forwards), so we capture both and
let Phase 2 say which one separates.

Layers: 16 (our steering layer), 25 and 27 (clockv2 layers).

Hooks are READ-ONLY - every hook returns None. Nothing is written outside
harm_dir/.
"""
import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, "/home/ore99/experement")

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_TOKEN = "<|mdm_mask|>"
LAYERS = [16, 25, 27]
DATA = "/home/ore99/experement/harm_dir/data"
MASK_RE = r"<mask:(\d+)>"
OPEN_TAG, CLOSE_TAG = "<<TPL>>", "<</TPL>>"


class ResponsePool:
    """Mean-pool each block's output over a token span. READ-ONLY."""

    def __init__(self, blocks, layers):
        self.layers = layers
        self.out = {}
        self.span = (0, 0)
        self._h = [blocks[i].register_forward_hook(self._mk(i)) for i in layers]

    def _mk(self, layer):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out    # [1, seq, d]
            lo, hi = self.span
            self.out[layer] = h[0, lo:hi].mean(0).detach().float().cpu()
            return None                                             # READ-ONLY
        return hook

    def close(self):
        for h in self._h:
            h.remove()
        self._h = []


def build_prompt(tok, behavior, user_content, layout):
    if layout == "bare":
        content = behavior
    else:
        content = user_content.replace("{behavior}", behavior)
        content = re.sub(MASK_RE, lambda m: MASK_TOKEN * int(m.group(1)), content)
        content = content.replace(OPEN_TAG, "").replace(CLOSE_TAG, "")
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs_complete.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "states.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layouts", default="bare,dija")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    pairs = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        pairs = pairs[:args.limit]

    print(f"cases={len(pairs)}  layouts={layouts}  layers={LAYERS}")
    print(f"cuda={torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16)
    model.all_tied_weights_keys = {}
    model = model.to("cuda").eval()
    from ladaAndH import discover_blocks
    _, blocks = discover_blocks(model)
    print(f"model loaded in {time.time()-t0:.1f}s  n_blocks={len(blocks)}")
    assert max(LAYERS) < len(blocks), f"layer {max(LAYERS)} out of range"

    pool = ResponsePool(blocks, LAYERS)
    recs = []
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(pairs, 1):
            for layout in layouts:
                prompt_str = build_prompt(tok, p["behavior"], p["user_content"], layout)
                pids = tok(prompt_str, add_special_tokens=False)["input_ids"]
                for side in ("harm", "safe"):
                    rids = tok(p[f"{side}_text"], add_special_tokens=False)["input_ids"]
                    ids = pids + rids
                    if len(ids) > args.max_len:          # trim prompt, keep response
                        ids = ids[-args.max_len:]
                    lo, hi = len(ids) - len(rids), len(ids)
                    pool.span = (lo, hi)
                    pool.out = {}
                    x = torch.tensor([ids], dtype=torch.long, device="cuda")
                    attn = torch.ones_like(x)
                    model(x, attention_mask=attn)
                    assert set(pool.out) == set(LAYERS), f"missing layers: {pool.out.keys()}"
                    recs.append({
                        "case_id": p["case_id"], "side": side, "layout": layout,
                        "safe_source": p["safe_source"],
                        "harm_arm": p["harm_arm"], "safe_arm": p.get("safe_arm"),
                        "n_resp_tokens": len(rids), "n_prompt_tokens": lo,
                        "words": p[f"{side}_words"],
                        "h": {l: pool.out[l].clone() for l in LAYERS},
                    })
            if i % 10 == 0 or i == len(pairs):
                print(f"  [{i}/{len(pairs)}] {time.time()-t0:.1f}s  rows={len(recs)}")
    pool.close()

    torch.save({
        "model_id": MODEL_ID, "layers": LAYERS, "layouts": layouts,
        "pooling": "mean over RESPONSE tokens only, teacher-forced single forward pass",
        "records": recs,
    }, args.out)

    n = len(recs)
    print(f"\nwrote {args.out}  ({n} records = {len(pairs)} cases x "
          f"{len(layouts)} layouts x 2 sides)")
    tl = [r["n_resp_tokens"] for r in recs if r["side"] == "harm"]
    sl = [r["n_resp_tokens"] for r in recs if r["side"] == "safe"]
    import statistics
    print(f"  response tokens: harm median {statistics.median(tl):.0f}  "
          f"safe median {statistics.median(sl):.0f}")


if __name__ == "__main__":
    main()
