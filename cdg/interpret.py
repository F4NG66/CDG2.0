from __future__ import annotations
"""Mechanistic interpretability utilities.

Three pillars:
  1. feature_vocab_labels   – project SAE decoder directions through the model's
                               unembedding matrix to find which tokens a feature
                               "writes towards" (logit-lens / decoder projection).
  2. feature_atlas_data     – pairwise cosine-similarity matrix + clustering for
                               the "Feature Atlas" figure.
  3. print_feature_report   – human-readable printout of the top features for a
                               given (layer, frac, group pair) with their vocab labels.

All three work offline from saved SAE weights; no live model forward pass needed
for vocab labels (only the unembedding matrix is required).
"""
import json
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Vocabulary label projection
# ---------------------------------------------------------------------------

def _get_unembed(model_or_weight):
    """Extract the unembedding matrix from a model or accept a raw tensor."""
    if isinstance(model_or_weight, torch.Tensor):
        return model_or_weight.float()
    # try common attribute paths
    for attr in ("lm_head.weight", "embed_out.weight", "output.weight",
                 "transformer.wte.weight"):
        obj = model_or_weight
        found = True
        for part in attr.split("."):
            if not hasattr(obj, part):
                found = False; break
            obj = getattr(obj, part)
        if found and isinstance(obj, torch.Tensor):
            return obj.detach().float()
    raise AttributeError(
        "Cannot find unembedding matrix. Pass model.lm_head.weight directly."
    )


def feature_vocab_labels(
    sae,
    feature_ids,
    tokenizer,
    unembed,          # model, or model.lm_head.weight tensor  (vocab_size, d_model)
    topk: int = 10,
    bottom: bool = True,
) -> list[dict]:
    """For each feature, return the top (and bottom) vocabulary tokens.

    Method: project the SAE decoder direction w = W_dec[:, f]  (shape d_model)
    through the unembedding:  scores = W_unembed @ w  (shape vocab_size).
    High-scoring tokens are what this feature direction "promotes"; low-scoring
    are what it suppresses.

    Args:
        sae:         TopKSAE, already on CPU.
        feature_ids: iterable of int feature indices.
        tokenizer:   HuggingFace tokenizer (for decode).
        unembed:     (vocab_size, d_model) tensor, or the model itself.
        topk:        how many top/bottom tokens to return per feature.
        bottom:      if True, also return the bottom-k (suppressed) tokens.

    Returns:
        list of dicts, one per feature, with keys:
            feature, top_tokens, bottom_tokens (if bottom=True), norm
        Each token entry: {token_id, token_str, score}
    """
    W_u = _get_unembed(unembed)      # (V, d)
    W_dec = sae.W_dec.detach().float()  # (d, n_features)

    results = []
    for f in feature_ids:
        f = int(f)
        direction = W_dec[:, f]                # (d,)
        norm = float(direction.norm())
        direction_unit = direction / (norm + 1e-8)
        scores = W_u @ direction_unit          # (V,)  logit-lens projection

        top_vals, top_ids = torch.topk(scores, topk)
        top_tokens = [
            {"token_id": int(tid), "token_str": tokenizer.decode([int(tid)]),
             "score": float(s)}
            for tid, s in zip(top_ids.tolist(), top_vals.tolist())
        ]
        entry = {"feature": f, "norm": norm, "top_tokens": top_tokens}

        if bottom:
            bot_vals, bot_ids = torch.topk(-scores, topk)
            entry["bottom_tokens"] = [
                {"token_id": int(tid), "token_str": tokenizer.decode([int(tid)]),
                 "score": float(-s)}
                for tid, s in zip(bot_ids.tolist(), bot_vals.tolist())
            ]
        results.append(entry)
    return results


def print_feature_report(feature_labels: list[dict], *, n_tokens: int = 8,
                         header: str = "") -> str:
    """Format feature vocab labels as a readable string for quick inspection.

    Example output:
        Feature 1234 (norm=3.21)
          TOP:    " harmful"(2.3)  " injec"(2.1)  "tion"(1.9) ...
          BOTTOM: " safe"(-1.2)   " benign"(-1.1) ...
    """
    lines = []
    if header:
        lines.append(header)
        lines.append("=" * len(header))
    for entry in feature_labels:
        f = entry["feature"]
        norm = entry.get("norm", float("nan"))
        top_str = "  ".join(
            f'"{t["token_str"]}"({t["score"]:.2f})'
            for t in entry.get("top_tokens", [])[:n_tokens]
        )
        lines.append(f"Feature {f:>6}  (|w|={norm:.3f})")
        lines.append(f"  TOP:    {top_str}")
        if "bottom_tokens" in entry:
            bot_str = "  ".join(
                f'"{t["token_str"]}"({t["score"]:.2f})'
                for t in entry["bottom_tokens"][:n_tokens]
            )
            lines.append(f"  BOTTOM: {bot_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Feature Atlas
# ---------------------------------------------------------------------------

def feature_atlas_data(sae, feature_ids) -> dict:
    """Compute pairwise cosine-similarity + hierarchical clustering order for
    the Feature Atlas figure.

    Returns:
        {
          "feature_ids": [...],   # input feature_ids (as ints)
          "cosine_sim":  ndarray  # (N, N) symmetric float32
          "order":       ndarray  # (N,) int, hierarchical-clustering row order
                                  #  (None if scipy unavailable)
          "clusters":    ndarray  # (N,) int cluster labels (k-means, k=auto)
                                  #  (None if sklearn unavailable)
        }
    """
    feature_ids = [int(f) for f in feature_ids]
    W_dec = sae.W_dec.detach().float()  # (d, n_features)
    dirs = W_dec[:, feature_ids].t()    # (N, d)
    norms = dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dirs_unit = dirs / norms
    sim = (dirs_unit @ dirs_unit.t()).cpu().numpy().astype(np.float32)
    np.fill_diagonal(sim, 1.0)

    order = None
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = np.clip(1.0 - sim, 0, 2)
        condensed = squareform((dist + dist.T) / 2, checks=False)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    except ImportError:
        pass

    clusters = None
    try:
        from sklearn.cluster import KMeans
        n_clusters = max(2, min(8, len(feature_ids) // 4))
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        clusters = km.fit_predict(dirs_unit.numpy()).tolist()
    except (ImportError, Exception):
        pass

    return {
        "feature_ids": feature_ids,
        "cosine_sim": sim,
        "order": order.tolist() if order is not None else None,
        "clusters": clusters,
    }


def save_atlas(atlas: dict, path: str) -> str:
    """Save atlas data as JSON (sim matrix stored as nested list)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {k: v for k, v in atlas.items() if k != "cosine_sim"}
    out["cosine_sim"] = atlas["cosine_sim"].tolist()
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    return path


# ---------------------------------------------------------------------------
# 3. Feature activation correlation (co-activation matrix)
# ---------------------------------------------------------------------------

def coactivation_matrix(records, *, feature_ids, scope: str = "tpl_mask",
                        frac: float = 0.10, layer: int = 16,
                        groups=("B", "C")) -> np.ndarray:
    """Compute feature co-activation correlation matrix from recorded SAE acts.

    Unlike the decoder-direction cosine similarity (which is geometry), this
    shows whether features *empirically* fire together on the same tokens.
    Combining both reveals whether geometrically-similar features are also
    functionally-similar.

    Returns: (N, N) Pearson-correlation matrix over cases in `groups`.
    """
    from .probe import stack_group
    X, _ = stack_group(records, groups=groups, scope=scope, frac=frac,
                       layer=layer, space="sae")
    if X is None:
        raise ValueError(f"No data for groups={groups} scope={scope} frac={frac} layer={layer}")
    feat_ids = [int(f) for f in feature_ids]
    X_sub = X[:, feat_ids].float().numpy()   # (n_cases, k)
    # Pearson correlation
    X_c = X_sub - X_sub.mean(0, keepdims=True)
    std = X_c.std(0) + 1e-8
    X_n = X_c / std
    C = (X_n.T @ X_n) / max(1, X_sub.shape[0] - 1)
    return C.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Mechanism summary: tie everything together
# ---------------------------------------------------------------------------

def mechanism_summary(
    records,
    sae,
    tokenizer,
    unembed,
    *,
    scope: str = "tpl_mask",
    frac: float = 0.10,
    layer: int = 16,
    pairs: tuple = (("B", "C"), ("A", "B")),
    topk_features: int = 20,
    topk_tokens: int = 8,
    stat: str = "mean_gap",
    save_dir: Optional[str] = None,
) -> dict:
    """Full pipeline: find top differential features → label with vocab → build atlas.

    Returns a dict with everything needed to write the mechanism section of the paper.
    If save_dir is given, writes JSON files there.
    """
    from .probe import top_diff_features

    diff_rows = top_diff_features(
        records, scope=scope, fracs=(frac,), layers=(layer,),
        pairs=pairs, k=topk_features, stat=stat,
    )

    # collect union of top features across pairs
    union_feats = set()
    for r in diff_rows:
        union_feats.update(f["feature"] for f in r["features"])
    union_feats = sorted(union_feats)

    # vocab labels
    labels = feature_vocab_labels(sae, union_feats, tokenizer, unembed,
                                   topk=topk_tokens)
    label_map = {e["feature"]: e for e in labels}

    # decorate diff rows with vocab labels
    for r in diff_rows:
        for feat_entry in r["features"]:
            fi = feat_entry["feature"]
            lbl = label_map.get(fi, {})
            feat_entry["top_tokens"] = [t["token_str"]
                                        for t in lbl.get("top_tokens", [])[:5]]
            feat_entry["norm"] = lbl.get("norm")

    # atlas
    atlas = feature_atlas_data(sae, union_feats)

    result = {
        "layer": layer, "frac": frac, "scope": scope,
        "diff_rows": diff_rows,
        "vocab_labels": labels,
        "atlas": atlas,
    }

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        json_safe = {k: v for k, v in result.items() if k != "atlas"}
        json_safe["atlas"] = {kk: vv if not isinstance(vv, np.ndarray)
                              else vv.tolist()
                              for kk, vv in atlas.items()}
        with open(os.path.join(save_dir, "mechanism_summary.json"), "w") as f:
            json.dump(json_safe, f, indent=2, ensure_ascii=False, default=float)

    return result
