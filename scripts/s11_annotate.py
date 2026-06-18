#!/usr/bin/env python3
"""Module 4 – Cluster-Centric Feature Annotation.

Annotates the co-activation modules produced by s10_modules.py.

Two-level analysis (matching the "family" intuition):
  1. Per-feature  – logit-lens vocab labels (what does this dimension promote/suppress?)
  2. Per-cluster  – aggregate top tokens across all features in the module,
                    then ask an LLM: "what is the unifying theme of this feature family?"

Logit-lens source (in priority order):
  A. Pre-computed vocab_labels.json from s4 (fast, no model load)
  B. Computed on-the-fly from SAE W_dec + lm_head weights (requires --lm-head-path)

Actual activation examples (optional enhancement):
  C. singlecell_matrix from sae_tokens at frac=1.0 (requires re-recorded data)

Usage
-----
# Minimal — use cached s4 vocab labels:
python scripts/s11_annotate.py outputs/ \
    --module-csv outputs/transcriptome/modules/feature_module.csv \
    --vocab-labels-json analysis_output/vocab_labels.json \
    --api-model deepseek-chat

# With actual activation examples (recommended when available):
python scripts/s11_annotate.py outputs_new2/ \
    --module-csv outputs/transcriptome/modules/feature_module.csv \
    --vocab-labels-json analysis_output/vocab_labels.json \
    --scope out_unmask --layer 16 \
    --tokenizer-path GSAI-ML/LLaDA-8B-Instruct \
    --api-model deepseek-chat

Outputs (under <out_dir>/transcriptome/annotation/)
---------
  feature_labels.csv       per-feature logit-lens vocab labels
  module_labels.csv        per-module LLM annotation + aggregated top tokens
  module_annotation.png    heatmap of module themes (optional viz)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Logit-lens vocabulary labels
# ---------------------------------------------------------------------------

def load_cached_vocab_labels(json_path: str) -> dict[int, dict]:
    """Load pre-computed logit-lens labels from s4 vocab_labels.json.

    Returns {feature_id -> {"top_tokens": [...], "bottom_tokens": [...]}}.
    """
    if not json_path or not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        raw = json.load(f)
    result = {}
    if isinstance(raw, list):
        for entry in raw:
            fid = entry.get("feature")
            if fid is not None:
                result[int(fid)] = entry
    elif isinstance(raw, dict):
        for fid, entry in raw.items():
            result[int(fid)] = entry
    return result


def compute_logit_lens(
    sae_ckpt_path: str,
    lm_head_path: str,
    feature_ids: list[int],
    tokenizer,
    topk: int = 10,
) -> dict[int, dict]:
    """Compute logit-lens labels on-the-fly from SAE weights + lm_head.

    lm_head_path: path to a .pt or .safetensors file containing the
    unembedding weight tensor (vocab_size, d_model), OR a directory
    with model.safetensors / pytorch_model.bin.
    """
    import torch
    from cdg.sae import load_sae
    from cdg.interpret import feature_vocab_labels

    sae = load_sae(sae_ckpt_path)
    sae.eval()

    # Try to load lm_head weight
    W = None
    if os.path.isfile(lm_head_path):
        if lm_head_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            st = load_file(lm_head_path)
            for key in ("lm_head.weight", "embed_tokens.weight",
                        "model.embed_tokens.weight"):
                if key in st:
                    W = st[key].float()
                    break
        else:
            obj = torch.load(lm_head_path, map_location="cpu", weights_only=False)
            if isinstance(obj, torch.Tensor):
                W = obj.float()
            elif isinstance(obj, dict):
                for key in ("lm_head.weight", "embed_tokens.weight"):
                    if key in obj:
                        W = obj[key].float()
                        break
    if W is None:
        raise ValueError(f"Cannot find lm_head.weight in {lm_head_path!r}")

    results = feature_vocab_labels(
        sae, feature_ids, tokenizer, unembed=W, topk=topk, bottom=True
    )
    return {int(r["feature"]): r for r in results}


# ---------------------------------------------------------------------------
# Cluster-level token aggregation
# ---------------------------------------------------------------------------

# Tokens that are structurally ubiquitous and uninformative for annotation.
# These include EOS/EOT/padding/pure-whitespace tokens that appear everywhere.
_SKIP_TOKENS = {
    "<|endoftext|>", "<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>",
    "\n", "\r\n", "\r", " ", "", "\t",
    "eter",   # partial artifact from subword tokenization
}


def aggregate_cluster_tokens(
    feature_ids: list[int],
    vocab_labels: dict[int, dict],
    actual_top_tokens: dict[int, list[str]],
    topk_per_feature: int = 10,
    min_feature_support: int = 2,
) -> dict:
    """Find tokens that co-occur across multiple features in a cluster.

    Returns {
        "top_union": [(token, count), ...],     # tokens from logit-lens top
        "bottom_union": [(token, count), ...],  # tokens from logit-lens bottom
        "actual_union": [(token, count), ...],  # tokens from real activations
    }
    """
    top_counter    = Counter()
    bottom_counter = Counter()
    actual_counter = Counter()

    for fid in feature_ids:
        # Logit-lens top tokens
        if fid in vocab_labels:
            for tok in (vocab_labels[fid].get("top_tokens") or [])[:topk_per_feature]:
                s = tok.get("token_str", "").strip()
                if s and s not in _SKIP_TOKENS:
                    top_counter[s] += 1
            for tok in (vocab_labels[fid].get("bottom_tokens") or [])[:topk_per_feature]:
                s = tok.get("token_str", "").strip()
                if s and s not in _SKIP_TOKENS:
                    bottom_counter[s] += 1
        # Actual activation examples
        for tok_str in actual_top_tokens.get(fid, []):
            s = tok_str.strip()
            if s and s not in _SKIP_TOKENS:
                actual_counter[s] += 1

    # Filter to tokens appearing in ≥ min_feature_support features
    def filter_and_rank(counter, min_sup):
        return [(t, c) for t, c in counter.most_common(30) if c >= min_sup]

    return {
        "top_union":    filter_and_rank(top_counter,    min_feature_support),
        "bottom_union": filter_and_rank(bottom_counter, min_feature_support),
        "actual_union": filter_and_rank(actual_counter, min_feature_support),
    }


# ---------------------------------------------------------------------------
# LLM annotation
# ---------------------------------------------------------------------------

ANNOTATE_SYSTEM = (
    "You are a neural network interpretability researcher studying a sparse "
    "autoencoder (SAE) trained on a Chinese diffusion language model (LLaDA-8B-Instruct). "
    "You will be given information about a GROUP of SAE features (a co-activation cluster). "
    "These features tend to fire together on the same inputs. "
    "Your job is to identify the shared semantic / syntactic / topical theme that unifies this cluster — "
    "what 'family' concept do these features collectively represent? "
    "Use three information sources provided: "
    "(1) logit-lens top tokens — what vocabulary items each feature geometrically promotes, "
    "(2) logit-lens bottom tokens — what each feature suppresses, "
    "(3) actual high-activation tokens — what tokens fired these features on real model outputs. "
    "Write ONE concise sentence for the cluster theme. Be specific and mechanistic. "
    "Then list 3-5 keywords. "
    'Respond ONLY with valid JSON: {"theme": "<one sentence>", "keywords": ["kw1","kw2",...], "confidence": "high"|"medium"|"low"}'
)


class _DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout: int = 60, max_retries: int = 3):
        import requests as _req
        self._req = _req
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def annotate_cluster(self, module_id: int, feature_ids: list[int],
                         agg: dict, feature_summaries: list[str]) -> dict:
        top_str    = ", ".join(f"{t!r}(×{c})" for t, c in agg["top_union"][:15])
        bottom_str = ", ".join(f"{t!r}(×{c})" for t, c in agg["bottom_union"][:10])
        actual_str = ", ".join(f"{t!r}(×{c})" for t, c in agg["actual_union"][:15])
        feat_str   = "\n".join(f"  • {s}" for s in feature_summaries[:10])
        user_msg = (
            f"=== Module {module_id} ({len(feature_ids)} features) ===\n\n"
            f"[LOGIT-LENS: tokens promoted across ≥2 features in this cluster]\n{top_str or '(none)'}\n\n"
            f"[LOGIT-LENS: tokens suppressed across ≥2 features]\n{bottom_str or '(none)'}\n\n"
            f"[ACTUAL HIGH-ACTIVATION TOKENS from real model outputs]\n{actual_str or '(none)'}\n\n"
            f"[PER-FEATURE summaries (top vocab tokens per feature)]\n{feat_str}\n\n"
            "What is the unifying theme of this feature cluster?"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ANNOTATE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._req.post(self.url, json=payload, headers=headers,
                                   timeout=self.timeout)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                import re
                content = re.sub(r"^```(json)?|```$", "", content,
                                 flags=re.MULTILINE).strip()
                d = json.loads(content)
                return {"theme":       str(d.get("theme", "")),
                        "keywords":    d.get("keywords", []),
                        "confidence":  str(d.get("confidence", "medium"))}
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        return {"theme": f"[api_error: {last}]", "keywords": [], "confidence": "low"}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_module_annotation(module_df: pd.DataFrame, save_path: str) -> None:
    """Bar chart of module sizes coloured by theme keyword count."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(module_df) * 0.45 + 2)))

    # Left: module sizes
    ax = axes[0]
    sizes = module_df["n_features"].tolist()
    mids  = [f"M{int(m)}" for m in module_df["module_id"].tolist()]
    colors = plt.cm.tab20(np.linspace(0, 1, len(mids)))
    bars = ax.barh(mids[::-1], sizes[::-1], color=colors[::-1], height=0.7)
    ax.set_xlabel("Number of features")
    ax.set_title("Module sizes")
    ax.tick_params(axis="y", labelsize=8)
    for bar, n in zip(bars, sizes[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(n), va="center", fontsize=7)

    # Right: themes as text table
    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    for _, row in module_df.iterrows():
        kws = ", ".join(row.get("keywords", [])[:4]) if row.get("keywords") else ""
        theme = str(row.get("theme", ""))[:70]
        rows.append([f"M{int(row['module_id'])}", theme, kws,
                     str(row.get("confidence", ""))])
    if rows:
        tbl = ax2.table(
            cellText=rows,
            colLabels=["Module", "Theme", "Keywords", "Conf"],
            cellLoc="left",
            loc="center",
            colWidths=[0.06, 0.55, 0.30, 0.09],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Annotation plot → {save_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Cluster-centric SAE feature annotation (module family themes)"
    )
    ap.add_argument("out_dir",
                    help="Run output directory containing manifest.jsonl")
    # Module CSV (from s10)
    ap.add_argument("--module-csv", default=None,
                    help="feature_module.csv from s10 (default: auto-detect)")
    # Vocab labels sources
    ap.add_argument("--vocab-labels-json", default=None,
                    help="analysis_output/vocab_labels.json from s4 (logit-lens cache)")
    ap.add_argument("--lm-head-path", default=None,
                    help="Path to lm_head weight file for on-the-fly logit-lens "
                         "(optional; falls back to vocab-labels-json)")
    ap.add_argument("--sae-ckpt", default=None,
                    help="SAE ae.pt path (needed with --lm-head-path)")
    # Actual activation data (optional)
    ap.add_argument("--scope",   default="out_unmask",
                    help="Scope for singlecell matrix (default: out_unmask)")
    ap.add_argument("--layer",   type=int, default=16)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--tokenizer-path", default=None,
                    help="HF tokenizer for decoding token IDs (recommended)")
    ap.add_argument("--n-features", type=int, default=None)
    # Annotation settings
    ap.add_argument("--top-per-module", type=int, default=8,
                    help="Number of representative features to show per module")
    ap.add_argument("--topk-vocab", type=int, default=10,
                    help="Top vocab tokens per feature from logit-lens")
    ap.add_argument("--min-support", type=int, default=2,
                    help="Min features a token must appear in to be listed as cluster-wide")
    # LLM settings
    ap.add_argument("--no-interp", action="store_true",
                    help="Skip LLM call; only save aggregated vocab labels")
    ap.add_argument("--api-model", default="deepseek-chat")
    ap.add_argument("--api-base",  default="https://api.deepseek.com")
    ap.add_argument("--api-key",   default=None)
    ap.add_argument("--sleep",     type=float, default=0.5)
    ap.add_argument("--save-dir",  default=None)
    args = ap.parse_args()

    base = args.save_dir or os.path.join(args.out_dir, "transcriptome", "annotation")
    os.makedirs(base, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load feature → module assignments
    # ------------------------------------------------------------------
    module_csv = args.module_csv
    if module_csv is None:
        # auto-detect from out_dir
        candidates = [
            os.path.join(args.out_dir, "transcriptome", "modules", "feature_module.csv"),
            "outputs/transcriptome/modules/feature_module.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                module_csv = c
                break
    if module_csv is None or not os.path.exists(module_csv):
        print("ERROR: feature_module.csv not found. Run s10_modules.py first.")
        sys.exit(1)

    fm = pd.read_csv(module_csv)
    modules = sorted(fm["module_id"].unique())
    print(f"[s11_annotate] Loaded {len(fm)} features across {len(modules)} modules "
          f"from {module_csv!r}")

    # ------------------------------------------------------------------
    # 2. Load logit-lens vocabulary labels
    # ------------------------------------------------------------------
    vocab_labels: dict[int, dict] = {}

    # Try cached s4 output first
    vl_json = args.vocab_labels_json
    if vl_json is None:
        auto = os.path.join(os.path.dirname(args.out_dir), "analysis_output",
                            "vocab_labels.json")
        if os.path.exists(auto):
            vl_json = auto
        elif os.path.exists("analysis_output/vocab_labels.json"):
            vl_json = "analysis_output/vocab_labels.json"
    if vl_json and os.path.exists(vl_json):
        vocab_labels = load_cached_vocab_labels(vl_json)
        print(f"  Logit-lens: loaded {len(vocab_labels)} pre-computed features "
              f"from {vl_json!r}")

    # Try on-the-fly computation
    if args.lm_head_path and args.sae_ckpt:
        tokenizer = None
        if args.tokenizer_path:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                           trust_remote_code=True)
            except Exception as e:
                warnings.warn(f"Could not load tokenizer: {e}")
        all_feature_ids = fm["feature_id"].tolist()
        missing = [f for f in all_feature_ids if f not in vocab_labels]
        if missing:
            print(f"  Computing logit-lens for {len(missing)} features "
                  f"(this may take a while)…")
            try:
                computed = compute_logit_lens(
                    args.sae_ckpt, args.lm_head_path,
                    missing, tokenizer, topk=args.topk_vocab
                )
                vocab_labels.update(computed)
                print(f"  Computed logit-lens for {len(computed)} features")
            except Exception as e:
                warnings.warn(f"On-the-fly logit-lens failed: {e}")

    n_with_labels = sum(1 for f in fm["feature_id"] if int(f) in vocab_labels)
    print(f"  Vocab labels coverage: {n_with_labels}/{len(fm)} features "
          f"({100*n_with_labels/max(len(fm),1):.0f}%)")

    # ------------------------------------------------------------------
    # 3. Load tokenizer and actual activation data (optional)
    # ------------------------------------------------------------------
    tokenizer = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                       trust_remote_code=True)
            print(f"  Tokenizer loaded from {args.tokenizer_path!r}")
        except Exception as e:
            warnings.warn(f"Could not load tokenizer: {e}")

    actual_top_tokens: dict[int, list[str]] = {}
    try:
        from cdg.analysis.io import singlecell_matrix
        X_sparse, obs_tok = singlecell_matrix(
            args.out_dir, scope=args.scope, layer=args.layer,
            n_features=args.n_features, model_name=args.model_name,
            tokenizer=tokenizer,
        )
        N_tok, N_feat = X_sparse.shape
        print(f"  Actual activations: {N_tok} tokens × {N_feat} features")
        # For each feature of interest, collect top-8 non-empty token strings
        all_feats_of_interest = fm["feature_id"].tolist()
        for fid in all_feats_of_interest:
            if fid >= N_feat:
                continue
            col = np.asarray(X_sparse[:, fid].todense()).ravel()
            top_idx = np.argsort(-col)[:40]
            top_idx = top_idx[col[top_idx] > 0]
            toks = []
            for i in top_idx:
                s = str(obs_tok.iloc[int(i)].get("token_str", "")).strip()
                if s and s not in _SKIP_TOKENS:
                    toks.append(s)
                if len(toks) >= 10:
                    break
            actual_top_tokens[int(fid)] = toks
        print(f"  Actual top tokens collected for "
              f"{sum(1 for v in actual_top_tokens.values() if v)} features")
    except ValueError as e:
        print(f"  No singlecell data available ({e}); using logit-lens only")
    except Exception as e:
        warnings.warn(f"singlecell_matrix error: {e}; using logit-lens only")

    # ------------------------------------------------------------------
    # 4. Per-feature label table
    # ------------------------------------------------------------------
    feat_rows = []
    for _, row in fm.iterrows():
        fid = int(row["feature_id"])
        mid = int(row["module_id"])
        vl = vocab_labels.get(fid, {})
        top_toks = [t.get("token_str", "") for t in
                    (vl.get("top_tokens") or [])[:args.topk_vocab]]
        bot_toks = [t.get("token_str", "") for t in
                    (vl.get("bottom_tokens") or [])[:5]]
        act_toks = actual_top_tokens.get(fid, [])[:5]
        feat_rows.append({
            "feature_id":       fid,
            "module_id":        mid,
            "frac_active":      row.get("frac_active", ""),
            "top_tokens_logit": json.dumps(top_toks, ensure_ascii=False),
            "bot_tokens_logit": json.dumps(bot_toks, ensure_ascii=False),
            "top_tokens_actual": json.dumps(act_toks, ensure_ascii=False),
        })
    feat_df = pd.DataFrame(feat_rows)
    feat_path = os.path.join(base, "feature_labels.csv")
    feat_df.to_csv(feat_path, index=False)
    print(f"\n  Per-feature labels → {feat_path}")

    # ------------------------------------------------------------------
    # 5. Per-cluster aggregation + LLM annotation
    # ------------------------------------------------------------------
    client = None
    if not args.no_interp:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("  [warn] No DEEPSEEK_API_KEY; skipping LLM. Use --no-interp to silence.")
        else:
            client = _DeepSeekClient(api_key=api_key, base_url=args.api_base,
                                     model=args.api_model)

    module_rows = []
    for mid in sorted(modules):
        mod_feats = fm[fm["module_id"] == mid].sort_values(
            "frac_active", ascending=False)
        feat_ids = mod_feats["feature_id"].tolist()
        top_feats = feat_ids[:args.top_per_module]

        # Aggregate tokens across all features in module
        agg = aggregate_cluster_tokens(
            feat_ids, vocab_labels, actual_top_tokens,
            topk_per_feature=args.topk_vocab,
            min_feature_support=args.min_support,
        )

        # Short per-feature summary for LLM context
        feat_summaries = []
        for fid in top_feats:
            vl = vocab_labels.get(int(fid), {})
            top_t = [t.get("token_str", "") for t in
                     (vl.get("top_tokens") or [])[:6]]
            act_t = actual_top_tokens.get(int(fid), [])[:4]
            s = f"feat {fid}: top_vocab=[{', '.join(repr(t) for t in top_t)}]"
            if act_t:
                s += f"  actual=[{', '.join(repr(t) for t in act_t)}]"
            feat_summaries.append(s)

        print(f"\n  Module {mid:2d} ({len(feat_ids):4d} features):")
        print(f"    Top vocab tokens: "
              f"{[t for t, _ in agg['top_union'][:8]]}")
        print(f"    Actual tokens:    "
              f"{[t for t, _ in agg['actual_union'][:8]]}")

        theme, keywords, conf = "", [], "low"
        if client:
            result = client.annotate_cluster(mid, feat_ids, agg, feat_summaries)
            theme    = result["theme"]
            keywords = result["keywords"]
            conf     = result["confidence"]
            print(f"    Theme: {theme[:90]!r}  [{conf}]")
            time.sleep(args.sleep)

        module_rows.append({
            "module_id":         mid,
            "n_features":        len(feat_ids),
            "top_vocab_tokens":  json.dumps([t for t, _ in agg["top_union"][:15]],
                                             ensure_ascii=False),
            "bottom_vocab_tokens": json.dumps([t for t, _ in agg["bottom_union"][:10]],
                                              ensure_ascii=False),
            "actual_tokens":     json.dumps([t for t, _ in agg["actual_union"][:15]],
                                             ensure_ascii=False),
            "theme":             theme,
            "keywords":          json.dumps(keywords, ensure_ascii=False),
            "confidence":        conf,
        })

    mod_df = pd.DataFrame(module_rows)
    mod_path = os.path.join(base, "module_labels.csv")
    mod_df.to_csv(mod_path, index=False)
    print(f"\n  Module labels → {mod_path}")

    # ------------------------------------------------------------------
    # 6. Visualization
    # ------------------------------------------------------------------
    plot_module_annotation(mod_df, os.path.join(base, "module_annotation.png"))

    print("\n[s11_annotate] Done.")


if __name__ == "__main__":
    main()
