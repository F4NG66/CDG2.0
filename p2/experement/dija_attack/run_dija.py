#!/usr/bin/env python3
"""dija_attack/run_dija.py - PAPER-FAITHFUL, resumable, single-arm DIJA LLaDA runner.

Matches the paper's attack reference harness exactly (cdg.config.llada_attack_config +
cdg.backends.dlm_runner.DLMRunner.generate + cdg.denoise.denoise):

  gen_length=128, steps=128, block_length=128 (single block), temperature=0.2,
  cfg_scale=0.0, remask="low_confidence", mask_id=126336, dtype=bfloat16,
  fill_all_masks=True.

Why this differs from the earlier paired_dija_runner smoke (which collapsed to "1.."):
  - fill_all_masks=True: ONE unified fill schedule over EVERY mask in the sequence
    (injected worksheet blanks in the prompt + appended output), not a block-restricted
    schedule sized to the output region only. This is the paper's template-injection
    denoiser. (denoise() copied byte-identical -> cdg_denoise.py.)
  - Scaffold wrapped in <<TPL>>...<</TPL>> sentinels so the FILLED blanks (which live
    inside the prompt span) are decoded into the response and reach the judge. The tags
    are stripped before the model sees the prompt; generation is identical with/without.
  - block_length=128 (single block) and temperature=0.2, per the paper config.

The denoiser (cdg_denoise.denoise) and the build/decode logic below are reused from the
reference harness unedited/verbatim; only orchestration (resume, incremental, per-row
diagnostics) is added here. Nothing outside dija_attack/ is modified.

  python dija_attack/run_dija.py --limit 3          # smoke
  python dija_attack/run_dija.py --limit 0          # full (resumable)
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from cdg_denoise import denoise  # noqa: E402  (byte-identical copy of cdg/denoise.py)

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
MASK_TOKEN = "<|mdm_mask|>"
OPEN_TAG = "<<TPL>>"
CLOSE_TAG = "<</TPL>>"
MASK_MARKER_RE = r"<mask:(\d+)>"


# --- tolerant token-subsequence search (verbatim from cdg/backends/dlm_runner.py) ---
def _find_subseq(hay, needle):
    if not needle:
        return (-1, -1)
    n = len(needle)
    for s in range(0, len(hay) - n + 1):
        if hay[s:s + n] == needle:
            return (s, s + n)
    return (-1, -1)


def _tolerant_find(hay, needle):
    for cand in (needle, needle[1:], needle[:-1], needle[1:-1]):
        if not cand:
            continue
        s, e = _find_subseq(hay, cand)
        if s >= 0:
            return (s, e)
    return (-1, -1)


class _NoHooks:
    def clear(self):
        pass


class PaperRunner:
    """Minimal stand-in for DLMRunner: just what denoise() touches (forward + hooks.clear)."""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = _NoHooks()

    def forward(self, x, attention_mask):
        out = self.model(x, attention_mask=attention_mask)
        return out.logits if hasattr(out, "logits") else out[0]

    def _tok(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # --- prompt build + TPL span resolution (mirrors DLMRunner.build_inputs) ---
    def build_inputs(self, scaffold: str):
        content = OPEN_TAG + scaffold + CLOSE_TAG            # bracket the worksheet
        content = re.sub(MASK_MARKER_RE,
                         lambda m: MASK_TOKEN * int(m.group(1)), content)
        prompt_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False)
        ids = self._tok(prompt_str)
        t0 = t1 = None
        if OPEN_TAG in content and CLOSE_TAG in content:
            o_ids = self._tok(OPEN_TAG)
            c_ids = self._tok(CLOSE_TAG)
            os_, oe = _tolerant_find(ids, o_ids)
            cs, ce = _tolerant_find(ids, c_ids)
            if os_ >= 0 and cs >= 0 and oe <= cs:
                inner = ids[oe:cs]
                ids = ids[:os_] + inner + ids[ce:]           # strip both tags
                t0, t1 = os_, os_ + len(inner)
        return torch.tensor([ids], dtype=torch.long, device=self.device), (t0, t1)

    # --- generation (mirrors DLMRunner.generate, no steering/recorder/SAE) ---
    @torch.no_grad()
    def generate(self, scaffold, *, steps, gen_length, block_length, temperature,
                 remask, mask_id):
        ids, (t0, t1) = self.build_inputs(scaffold)
        P = ids.shape[1]
        total = P + gen_length
        x = torch.full((1, total), mask_id, dtype=torch.long, device=self.device)
        x[:, :P] = ids
        attn = torch.ones((1, total), dtype=torch.long, device=self.device)

        n_inject = int((ids == mask_id).sum())               # blanks inside the prompt
        x = denoise(self, x, attn, steps=steps, gen_length=gen_length, prompt_len=P,
                    block_length=block_length, temperature=temperature, remask=remask,
                    mask_id=mask_id, recorder=None, fill_all_masks=True)

        out_ids = x[0, P:]
        resp = self.tokenizer.decode(out_ids.tolist(), skip_special_tokens=True)
        if t0 is not None:
            tpl_filled = self.tokenizer.decode(x[0, t0:t1].tolist(), skip_special_tokens=True)
            resp = (tpl_filled + "\n" + resp).strip()
        return resp, n_inject


def done_trajids(path):
    d = set()
    if os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if l:
                try:
                    d.add(json.loads(l)["traj_id"])
                except Exception:
                    pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", default=os.path.join(HERE, "refined_100.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "generations.jsonl"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=0)
    # paper attack config (llada_attack_config) as defaults:
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=128)   # single block
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--remask", default="low_confidence")
    ap.add_argument("--mask-id", type=int, default=MASK_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = json.load(open(args.refined))
    if args.limit:
        rows = rows[:args.limit]

    gen_cfg = {"steps": args.steps, "gen_length": args.gen_length,
               "block_length": args.block_length, "temperature": args.temperature,
               "cfg_scale": 0.0, "remask": args.remask, "mask_id": args.mask_id,
               "fill_all_masks": True, "dtype": "bfloat16", "config": "llada_attack_config"}

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[run] loading {args.model} on {device} (bf16) ...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    runner = PaperRunner(model, tokenizer, device)
    print(f"[run] PAPER config: steps={args.steps} gen={args.gen_length} "
          f"block={args.block_length} temp={args.temperature} remask={args.remask} "
          f"fill_all_masks=True", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set() if args.overwrite else done_trajids(args.out)
    if args.overwrite and os.path.exists(args.out):
        open(args.out, "w").close()
    print(f"[run] {len(rows)} scaffolds; {len(done)} already done", flush=True)

    smoke = []
    fout = open(args.out, "a", encoding="utf-8")
    n = 0
    for r in rows:
        tid = f"{r['id']}__harmful"
        if tid in done:
            continue
        scaffold = r["Refined_behavior"]
        n_expected = sum(int(x) for x in re.findall(MASK_MARKER_RE, scaffold))
        resp, n_inject = runner.generate(
            scaffold, steps=args.steps, gen_length=args.gen_length,
            block_length=args.block_length, temperature=args.temperature,
            remask=args.remask, mask_id=args.mask_id)
        row = {"traj_id": tid, "id": r["id"], "behavior": r["behavior"], "arm": "harmful",
               "prompt_text": scaffold, "response": resp,
               "mask_tokens_expected": n_expected, "mask_tokens_actual": n_inject,
               "n_masks": r.get("n_masks"), "gen": gen_cfg}
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        n += 1
        print(f"[run] {r['id']} inject-masks {n_inject}/{n_expected}; resp[:80]={resp[:80]!r}",
              flush=True)
        if args.limit and args.limit <= 3:
            smoke.append(row)
        if n % 10 == 0:
            print(f"[run] {n} generations done", flush=True)
    fout.close()
    print(f"[run] wrote {n} new rows -> {args.out}", flush=True)

    if smoke:
        print("\n" + "=" * 90 + "\nSMOKE DECODED OUTPUTS (full)\n" + "=" * 90, flush=True)
        for r in smoke:
            print(f"\n### {r['traj_id']}  (inject-masks {r['mask_tokens_actual']}/"
                  f"{r['mask_tokens_expected']})")
            print(f"  behavior: {r['behavior']}")
            print(f"  response: {r['response']}")


if __name__ == "__main__":
    main()
