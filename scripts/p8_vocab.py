#!/usr/bin/env python3
"""p8_vocab.py — Logit-Lens Vocabulary Labels (standalone tool).

Projects each SAE feature's decoder direction through the model's unembedding
matrix to find which vocabulary tokens the feature "writes towards" (promotes)
and "writes against" (suppresses).

  score(token t, feature f) = (W_unembed[t] · W_dec[:,f]) / ‖W_dec[:,f]‖

Efficiently computes for ALL active features in batch:
  scores_all = W_unembed @ W_dec   →  (vocab_size, n_features)
  then top-k / bottom-k per column

Requires the full model for W_unembed (~2-3 min to load on CPU, <1 min on GPU).
The model is only used to extract lm_head.weight; no forward pass is run.

Usage
-----
# All active features from p5 clustering (recommended):
python scripts/p8_vocab.py \
    --module-csv outputs/analysis/modules/feature_module.csv \
    --layer 16 --topk 12

# Specific feature IDs:
python scripts/p8_vocab.py --features 12130 5255 3338 4041 --layer 16

# Use a locally cached model path:
python scripts/p8_vocab.py \
    --module-csv outputs/analysis/modules/feature_module.csv \
    --model-path /path/to/LLaDA-8B-Instruct

Output: analysis_output/vocab_labels.json  (overwrites old file)
"""
import os, json, argparse, gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)

import torch
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--layer",       type=int,   default=16)
ap.add_argument("--sae-kind",    default="llada_mask")
ap.add_argument("--topk",        type=int,   default=12,
                help="Top-k AND bottom-k vocab tokens per feature")
ap.add_argument("--features",    type=int,   nargs="+", default=None,
                help="Explicit feature IDs (overrides --module-csv)")
ap.add_argument("--module-csv",  default=None,
                help="feature_module.csv from p5 — labels all active features")
ap.add_argument("--model-path",  default=None,
                help="Local path to LLaDA model dir (or HF repo ID). "
                     "Default: reads from cdg config (GSAI-ML/LLaDA-8B-Instruct)")
ap.add_argument("--batch-size",  type=int,   default=2048,
                help="Features per batch for memory efficiency (default 2048)")
ap.add_argument("--device",      default="cuda",
                help="Device for batch matmul (cuda or cpu)")
ap.add_argument("--out",         default="analysis_output/vocab_labels.json")
ap.add_argument("--skip-model",  action="store_true",
                help="Dry run: only print decoder norms, skip vocab labels")
args = ap.parse_args()

from cdg.sae    import load_sae, sae_ckpt_path
from cdg.config import get_backend_config

cfg = get_backend_config("llada_attack")

# ── Collect feature IDs ───────────────────────────────────────────────────────
if args.features:
    feature_ids = list(args.features)
    print(f"Using {len(feature_ids)} explicitly specified features")

elif args.module_csv and os.path.exists(args.module_csv):
    import pandas as pd
    fm = pd.read_csv(args.module_csv)
    feature_ids = sorted(fm["feature_id"].tolist())
    print(f"Using {len(feature_ids)} active features from {args.module_csv!r}")

else:
    # fall back to diff_features.json (old behaviour)
    for diff_path in ["outputs/analysis/delta/diff_features.json",
                      "analysis_output/diff_features.json"]:
        if os.path.exists(diff_path):
            diff = json.load(open(diff_path))
            focus = [r for r in diff.get("focus_rows", []) if r.get("pair") == "BvsC"]
            if focus:
                feature_ids = [f["feature"] for f in focus[0]["features"][:20]]
                print(f"Using {len(feature_ids)} top B-vs-C features from {diff_path!r}")
                break
    else:
        print("[error] No feature source found. Pass --module-csv or --features.")
        raise SystemExit(1)

# ── Load SAE ──────────────────────────────────────────────────────────────────
sae_root = os.path.join("saes", args.sae_kind)
ckpt, cfgp = sae_ckpt_path(sae_root, layer=args.layer, trainer=1)
sae = load_sae(ckpt, config_path=cfgp)
print(f"SAE: d_model={sae.d_model}  n_features={sae.n_features}  k={sae.k}")

W_dec = sae.W_dec.detach().float().cpu()   # (d_model, n_features)
# Normalise each column: logit-lens uses unit decoder directions
norms = W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
W_dec_norm = W_dec / norms                 # (d_model, n_features)

if args.skip_model:
    print("\nDecoder norms for requested features:")
    for fid in feature_ids[:20]:
        print(f"  feature {fid:6d}  ‖w_dec‖ = {float(norms[0, fid]):.4f}")
    print("(pass without --skip-model to compute vocab labels)")
    raise SystemExit(0)

# ── Load model unembedding ────────────────────────────────────────────────────
model_path = args.model_path or cfg.model_id
print(f"\nLoading tokenizer + unembedding from {model_path!r} ...")
print("(loading only, no forward pass — fast on GPU, ~2-3 min on CPU)\n")

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                     local_files_only=(args.model_path is not None))

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",          # load to CPU; we only need the weight matrix
)

# Find unembedding weight
W_unembed = None
for attr in ["lm_head.weight", "model.embed_tokens.weight",
             "embed_out.weight", "transformer.wte.weight"]:
    obj = model
    try:
        for part in attr.split("."):
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor) and obj.ndim == 2 and obj.shape[0] > 10000:
            W_unembed = obj.detach().float().cpu()
            print(f"  Unembedding: {attr}  shape={tuple(W_unembed.shape)}")
            break
    except AttributeError:
        continue

if W_unembed is None:
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.shape[0] > 50000:
            W_unembed = p.detach().float().cpu()
            print(f"  Unembedding (scan): {name}  shape={tuple(W_unembed.shape)}")
            break

if W_unembed is None:
    raise RuntimeError("Cannot find unembedding matrix. Inspect model.named_parameters().")

vocab_size = W_unembed.shape[0]
del model; gc.collect(); torch.cuda.empty_cache()
print(f"  Model unloaded from memory. vocab_size={vocab_size}\n")

# ── Batch logit-lens computation ──────────────────────────────────────────────
# scores[:, f] = W_unembed @ W_dec_norm[:, f]   shape: (vocab_size,)
# Compute in batches of features to control memory.
# Peak memory: batch_size × vocab_size × 4 bytes
#   default batch_size=2048: 2048 × 126340 × 4 ≈ 1 GB

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
W_u = W_unembed.to(device)       # (vocab, d_model)
W_d = W_dec_norm.to(device)      # (d_model, n_feats)

fid_array = torch.tensor(feature_ids, dtype=torch.long)
n_feats   = len(feature_ids)
B         = args.batch_size
topk      = args.topk

print(f"Computing logit-lens for {n_feats} features "
      f"(batch_size={B}, device={device})...")

all_labels = []
for start in range(0, n_feats, B):
    end   = min(start + B, n_feats)
    fids  = fid_array[start:end]           # (b,)
    W_b   = W_d[:, fids]                   # (d_model, b)
    scores = W_u @ W_b                     # (vocab, b)  — the key matmul

    top_vals,  top_idx  = torch.topk(scores,  topk, dim=0)   # (topk, b)
    bot_vals,  bot_idx  = torch.topk(-scores, topk, dim=0)   # (topk, b)

    top_vals  = top_vals.cpu().numpy()
    top_idx   = top_idx.cpu().numpy()
    bot_vals  = (-bot_vals).cpu().numpy()
    bot_idx   = bot_idx.cpu().numpy()

    for i, fid in enumerate(fids.tolist()):
        top_tokens = [
            {"token_id": int(top_idx[j, i]),
             "token_str": tok.decode([top_idx[j, i]], skip_special_tokens=True),
             "score": float(top_vals[j, i])}
            for j in range(topk)
        ]
        bot_tokens = [
            {"token_id": int(bot_idx[j, i]),
             "token_str": tok.decode([bot_idx[j, i]], skip_special_tokens=True),
             "score": float(bot_vals[j, i])}
            for j in range(topk)
        ]
        all_labels.append({
            "feature":      fid,
            "top_tokens":   top_tokens,
            "bottom_tokens": bot_tokens,
        })

    pct = 100 * end / n_feats
    print(f"  {end:5d}/{n_feats}  ({pct:.0f}%)", end="\r", flush=True)

print(f"\nDone. Computed labels for {len(all_labels)} features.")

# ── Print a sample for quick inspection ──────────────────────────────────────
print("\nSample (first 5 features):")
for entry in all_labels[:5]:
    top_str = " | ".join(repr(t["token_str"]) for t in entry["top_tokens"][:6])
    print(f"  feat {entry['feature']:6d}: top → {top_str}")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(all_labels, f, indent=2, ensure_ascii=False)
print(f"\n[saved] {args.out}  ({len(all_labels)} features)")
print(f"\nNext: re-run p6_annotate.py with updated vocab_labels.json")
