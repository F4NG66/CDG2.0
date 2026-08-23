#!/usr/bin/env python3
"""PHASE 1' step 3 - teacher-forced hidden-state capture for the DeepSeek-both pairs.

Same extraction as capture_states.py (imported wholesale so it cannot drift):
teacher-force [prompt + response] through LLaDA in one no-mask forward pass and
mean-pool block outputs over the RESPONSE tokens only, layers 16/25/27, layouts
bare + dija, READ-ONLY hooks.

The ONLY difference from Phase 1 is the source of the pairs: here BOTH the harm
and safe response are DeepSeek-authored (pairs_ds.jsonl), so authorship is matched
by construction. Records are tagged author="deepseek".

CRITICAL - Gate E depends on this: the layers, layouts, and pooling here are
byte-identical to the old states.pt (LLaDA-authored natural pairs), so a v_harm
built on these states can be scored directly on those LLaDA pairs to test whether
the direction transfers across author or is just a DeepSeek-content axis.

Output: data/states_ds.pt
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_states import (LAYERS, MODEL_ID, ResponsePool, build_prompt)

DATA = "/home/ore99/experement/harm_dir/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs_ds.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "states_ds.pt"))
    ap.add_argument("--author", default="deepseek",
                    help="author tag for the records (deepseek for Phase 2', llada for Gate E)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layouts", default="bare,dija")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    pairs = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        pairs = pairs[:args.limit]

    print(f"cases={len(pairs)}  layouts={layouts}  layers={LAYERS}  (author={args.author})")
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
                        "author": args.author,
                        "safe_source": f"{args.author}_both",
                        "jaccard": p.get("jaccard"),
                        "n_resp_tokens": len(rids), "n_prompt_tokens": lo,
                        "words": p[f"{side}_words"],
                        "h": {l: pool.out[l].clone() for l in LAYERS},
                    })
            if i % 10 == 0 or i == len(pairs):
                print(f"  [{i}/{len(pairs)}] {time.time()-t0:.1f}s  rows={len(recs)}")
    pool.close()

    torch.save({
        "model_id": MODEL_ID, "layers": LAYERS, "layouts": layouts,
        "author": args.author,
        "pooling": "mean over RESPONSE tokens only, teacher-forced single forward pass",
        "records": recs,
    }, args.out)

    n = len(recs)
    print(f"\nwrote {args.out}  ({n} records = {len(pairs)} cases x "
          f"{len(layouts)} layouts x 2 sides)")
    tl = [r["n_resp_tokens"] for r in recs if r["side"] == "harm"]
    sl = [r["n_resp_tokens"] for r in recs if r["side"] == "safe"]
    print(f"  response tokens: harm median {statistics.median(tl):.0f}  "
          f"safe median {statistics.median(sl):.0f}")


if __name__ == "__main__":
    main()
