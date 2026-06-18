#!/usr/bin/env python3
"""
s4_vocab_labels.py
------------------
Project each SAE feature's decoder direction through the model's unembedding
matrix to find which vocabulary tokens the feature "writes towards" (promotes)
and "writes against" (suppresses).

Method: score(token t, feature f) = (W_unembed[t] · W_dec[:,f]) / ||W_dec[:,f]||
        This is the "logit-lens" projection — the same technique used in
        Anthropic's mechanistic interpretability work.

Reads feature IDs from analysis_output/diff_features.json (produced by s3)
or accepts --features 1234 5678 ... on the command line.

Requires the full LLaDA model to extract lm_head.weight (CPU-only is fine,
just slow ~2-3 min to load).  Pass --skip-model to skip this step.

Output: analysis_output/vocab_labels.json + printed report

Run: python scripts/s4_vocab_labels.py [--layer 16] [--features 1234 5678 ...]
     python scripts/s4_vocab_labels.py --skip-model    # dry run, no model needed
"""
import os, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--layer",      type=int, default=16)
ap.add_argument("--sae-kind",   default="llada_mask")
ap.add_argument("--topk",       type=int, default=12)
ap.add_argument("--features",   type=int, nargs="+", default=None,
                help="Feature IDs to label; defaults to top-10 from diff_features.json")
ap.add_argument("--skip-model", action="store_true",
                help="Skip loading the LLM (no vocab labels, just SAE norms)")
args = ap.parse_args()

from cdg.sae    import load_sae, sae_ckpt_path
from cdg.config import get_backend_config

cfg = get_backend_config("llada_attack")

# ── get feature IDs ──────────────────────────────────────────────────────────
if args.features:
    feature_ids = args.features
else:
    diff_path = "analysis_output/diff_features.json"
    if not os.path.exists(diff_path):
        print(f"[error] {diff_path} not found. Run s3_diff_features.py first, "
              "or pass --features manually.")
        raise SystemExit(1)
    diff = json.load(open(diff_path))
    focus = [r for r in diff["focus_rows"] if r["pair"] == "BvsC"]
    if not focus:
        print("[error] No BvsC row in diff_features.json.")
        raise SystemExit(1)
    feature_ids = [f["feature"] for f in focus[0]["features"][:15]]

print(f"Analyzing {len(feature_ids)} features: {feature_ids}\n")

# ── load SAE ─────────────────────────────────────────────────────────────────
sae_root = os.path.join("saes", args.sae_kind)
ckpt, cfgp = sae_ckpt_path(sae_root, layer=args.layer, trainer=1)
sae = load_sae(ckpt, config_path=cfgp)
print(f"SAE loaded: d_model={sae.d_model}  n_features={sae.n_features}  k={sae.k}")

# ── print SAE decoder norms (no model needed) ─────────────────────────────────
import torch
W_dec = sae.W_dec.detach().float()   # (d_model, n_features)
print(f"\n-- Decoder direction norms (feature importance proxy) --")
for fid in feature_ids:
    norm = float(W_dec[:, fid].norm())
    print(f"  feature {fid:6d}  ||w_dec|| = {norm:.4f}")

if args.skip_model:
    print("\n[skipped] Model not loaded (--skip-model).  "
          "Re-run without --skip-model to get vocab labels.")
    raise SystemExit(0)

# ── load model unembedding ────────────────────────────────────────────────────
print(f"\nLoading tokenizer + model unembedding from {cfg.model_id} ...")
print("(this takes ~2-3 min on CPU; use --skip-model to skip)\n")

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)

model = AutoModel.from_pretrained(
    cfg.model_id, trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
# LLaDA uses a different attribute name for the output projection
_unembed = None
for _attr in ["lm_head", "output", "embed_out", "cls", "transformer.wte",
              "model.embed_tokens"]:
    obj = model
    try:
        for part in _attr.split("."):
            obj = getattr(obj, part)
        if hasattr(obj, "weight") and obj.weight.ndim == 2:
            _unembed = obj.weight
            print(f"  found unembedding at: model.{_attr}.weight  shape={tuple(obj.weight.shape)}")
            break
    except AttributeError:
        continue
if _unembed is None:
    # last resort: scan all named parameters for (vocab_size, d_model) shaped matrix
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.shape[0] > 10000:
            _unembed = p
            print(f"  found unembedding via scan: {name}  shape={tuple(p.shape)}")
            break
if _unembed is None:
    raise RuntimeError("Cannot find unembedding matrix. Run with --skip-model and inspect the model manually.")
W_unembed = _unembed.detach().float().cpu()  # (vocab, d_model)
print(f"Unembedding shape: {W_unembed.shape}\n")
del model
import gc; gc.collect()
torch.cuda.empty_cache()

# ── compute vocab labels ──────────────────────────────────────────────────────
from cdg.interpret import feature_vocab_labels, print_feature_report

labels = feature_vocab_labels(
    sae, feature_ids, tokenizer, W_unembed,
    topk=args.topk, bottom=True,
)

print(print_feature_report(
    labels, n_tokens=10,
    header=f"Layer {args.layer}  SAE={args.sae_kind}  BvsC top features — vocab projection",
))

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs("analysis_output", exist_ok=True)
with open("analysis_output/vocab_labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2, ensure_ascii=False)
print(f"\n[saved] analysis_output/vocab_labels.json")
