"""test_seeds_extract.py - Run the LLaDA hidden-state + top-K capture over real
benchmark seeds from seed_fang.jsonl (single-turn, one file per question).

Reuses the exact collector / generate / save / verify functions from the
working test_load_extract.py - only the prompt SOURCE and the reporting change.
Each seed is an INDEPENDENT single-turn generation (no multi-turn concatenation),
so every question gets its own clean trajectory file.

Output:
    seed_states/seed_run/turn_01.pt ... turn_10.pt
    seed_states/seed_run/report.txt   (entropy + hidden_norm grouped by risk_level)

Usage:
    python test_seeds_extract.py            # first 10 valid seeds
    python test_seeds_extract.py 20         # first 20
"""

from __future__ import annotations
import os
import sys
import json
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

# reuse everything that already works
from test_load_extract import (
    StepActivationCollector, generate_with_capture, get_actual_len,
    save_turn, verify_payload, TOPK_LOGITS,
)

SEED_FILE   = "seed_fang.jsonl"
OUT_DIR     = "seed_states"
TRAJ_ID     = "seed_run"
N_SEEDS     = int(sys.argv[1]) if len(sys.argv) > 1 else 10

SELECTED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
GEN_LENGTH, STEPS, BLOCK_LENGTH = 64, 64, 32


def load_seeds(path, n):
    """Read JSONL, skip blank/malformed lines, return first n with a question."""
    seeds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue                       # tolerate truncated/garbage lines
            if obj.get("question"):
                seeds.append(obj)
            if len(seeds) >= n:
                break
    return seeds


def main():
    device = "cuda"
    seeds = load_seeds(SEED_FILE, N_SEEDS)
    print(f"Loaded {len(seeds)} seeds from {SEED_FILE}")

    print("Loading model...")
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

    blocks = [model.model.transformer.blocks[i] for i in SELECTED_LAYERS]
    collector = StepActivationCollector(blocks=blocks, layer_ids=SELECTED_LAYERS,
                                        lm_head=None)

    rows = []   # (id, risk, question, reply, actual_len, path)
    for i, seed in enumerate(seeds, start=1):
        q = seed["question"]
        risk = seed.get("risk_level", "?")
        print(f"\n=== Seed {i}/{len(seeds)} [{risk}] {seed.get('id','')} ===")
        print(f"  Q: {q[:90]!r}")

        msg = [{"role": "user", "content": q}]
        templated = tokenizer.apply_chat_template(msg, add_generation_prompt=True,
                                                  tokenize=False)
        input_ids = torch.tensor(tokenizer(templated)["input_ids"]).to(device).unsqueeze(0)
        prompt_len = input_ids.shape[1]

        collector.new_turn(prompt_len=prompt_len)
        out = generate_with_capture(
            model, input_ids, steps=STEPS, gen_length=GEN_LENGTH,
            block_length=BLOCK_LENGTH, temperature=0.0, cfg_scale=0.0,
            remasking="low_confidence", collector=collector,
        )

        reply = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0]
        actual_len = get_actual_len(out, prompt_len)
        print(f"  Reply: {reply[:100]!r}")
        print(f"  answer tokens: {actual_len}/{GEN_LENGTH}")

        path = save_turn(collector, out_dir=OUT_DIR, traj_id=TRAJ_ID, turn=i,
                         prompt_len=prompt_len, actual_len=actual_len)
        rows.append((seed.get("id", ""), risk, q, reply, actual_len, path))

    collector.close()

    # ---- report: entropy + hidden_norm, grouped by risk_level ----
    print("\n" + "=" * 70)
    print("ENTROPY / HIDDEN-NORM REPORT  (does risk track entropy?)")
    print("=" * 70)
    by_risk = defaultdict(list)
    lines = []
    for sid, risk, q, reply, alen, path in rows:
        if not path:
            continue
        r = verify_payload(path)
        ent = r.get("entropy_per_step_mean", float("nan"))
        hn = r["hidden_norm_mean"]
        by_risk[risk].append(ent)
        line = (f"  [{risk:<5}] ent={ent:7.4f}  hnorm={hn:7.2f}  len={alen:>2}  "
                f"{sid[:32]:<32}  reply={reply[:40]!r}")
        print(line)
        lines.append(line)

    print("-" * 70)
    summary = []
    for risk, vals in sorted(by_risk.items()):
        m = sum(vals) / len(vals)
        s = f"  risk={risk:<6} n={len(vals):<2}  mean_entropy={m:.4f}"
        print(s)
        summary.append(s)

    rep_path = os.path.join(OUT_DIR, TRAJ_ID, "report.txt")
    os.makedirs(os.path.dirname(rep_path), exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines + ["", "GROUPED:"] + summary) + "\n")
    print(f"\nReport written to {rep_path}")


if __name__ == "__main__":
    main()