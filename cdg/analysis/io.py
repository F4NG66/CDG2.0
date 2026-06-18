"""cdg.analysis.io — shared data-loading layer for transcriptomics-style analysis.

Provides:
  load_manifest      – read manifest.jsonl → pd.DataFrame (with group letter, judge fields)
  load_record        – torch.load a .pt record dict
  pseudobulk_matrix  – build (X[N_gen, n_features], obs) from pooled SAE activations
  singlecell_matrix  – build (X_sparse[N_tok, n_features], obs_tok) from sae_tokens

Statistical unit rule: pseudobulk rows = one generation.  Token rows are only for
annotation / clustering exploration, never for inter-group DE tests.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _group_letter(variant: str) -> str:
    """'B_harmful_injected' → 'B'."""
    return (variant or "")[:1].upper()


def _match_frac(store: dict, frac: float):
    """Find the key in store closest to frac (handles float repr noise)."""
    if frac in store:
        return frac
    for k in store:
        try:
            if abs(float(k) - float(frac)) < 1e-6:
                return k
        except (TypeError, ValueError):
            continue
    return None


def _judge_fields(judge: Optional[dict]) -> tuple:
    """Return (judge_success, judge_label) from a judge dict or None."""
    if judge is None:
        return None, None
    return judge.get("success"), judge.get("label")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_manifest(out_dir: str) -> pd.DataFrame:
    """Read manifest.jsonl and return a tidy DataFrame.

    Added columns: group (A/B/C/D), judge_success (0/1/None), judge_label.
    """
    path = os.path.join(out_dir, "manifest.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["group"] = df["variant"].apply(_group_letter)
    js, jl = zip(*df["judge"].apply(_judge_fields)) if len(df) else ([], [])
    df["judge_success"] = list(js)
    df["judge_label"] = list(jl)
    return df


def load_record(path: str) -> dict:
    """Load a .pt generation record."""
    return torch.load(path, map_location="cpu", weights_only=False)


def pseudobulk_matrix(
    out_dir: str,
    *,
    scope: str,
    layer: int,
    frac: float,
    model_name: Optional[str] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a pseudobulk feature matrix from pooled SAE activations.

    Parameters
    ----------
    out_dir    : directory containing manifest.jsonl and *.pt records
    scope      : scope name used during recording, e.g. "out_mask", "tpl_mask"
    layer      : transformer layer index (int)
    frac       : denoising fraction at which activations were recorded
    model_name : filter to a single model (optional)

    Returns
    -------
    X   : float32 ndarray [N_gen, n_features]  (pooled SAE activations)
    obs : DataFrame [N_gen] with columns:
          case_id, variant, group, content_type, has_template, attack_method,
          is_neutral, model_name, seed, judge_success, judge_label
    """
    manifest = load_manifest(out_dir)
    if model_name is not None:
        manifest = manifest[manifest["model_name"] == model_name].copy()

    vecs, obs_rows = [], []
    n_skip_nofile = 0
    n_skip_nan = 0
    n_skip_noscope = 0

    for _, mrow in manifest.iterrows():
        path = mrow.get("path")
        if not path or not os.path.exists(path):
            n_skip_nofile += 1
            continue
        try:
            rec = load_record(path)
        except Exception as e:
            warnings.warn(f"Could not load {path}: {e}")
            n_skip_nofile += 1
            continue

        sae_store = rec.get("sae", {}).get(scope, {})
        fk = _match_frac(sae_store, frac)
        if fk is None:
            n_skip_noscope += 1
            continue
        vec = sae_store[fk].get(layer)
        if vec is None:
            n_skip_noscope += 1
            continue
        v = vec.float().numpy()
        if np.any(np.isnan(v)):
            n_skip_nan += 1
            continue

        vecs.append(v)
        obs_rows.append({
            "case_id":      mrow.get("case_id"),
            "variant":      mrow.get("variant"),
            "group":        mrow.get("group"),
            "content_type": mrow.get("content_type"),
            "has_template": mrow.get("has_template"),
            "attack_method": mrow.get("attack_method"),
            "is_neutral":   mrow.get("is_neutral"),
            "model_name":   mrow.get("model_name"),
            "seed":         mrow.get("seed"),
            "judge_success": mrow.get("judge_success"),
            "judge_label":   mrow.get("judge_label"),
        })

    if n_skip_nofile or n_skip_nan or n_skip_noscope:
        warnings.warn(
            f"pseudobulk_matrix: skipped {n_skip_nofile} missing files, "
            f"{n_skip_nan} NaN rows (empty scope), "
            f"{n_skip_noscope} rows with no scope/layer/frac match. "
            f"Kept {len(vecs)} generations."
        )

    if not vecs:
        raise ValueError(
            f"No valid rows for scope={scope!r} layer={layer} frac={frac}. "
            "Check that records contain this scope/layer/frac combination."
        )

    X = np.stack(vecs, axis=0).astype(np.float32)
    obs = pd.DataFrame(obs_rows).reset_index(drop=True)
    return X, obs


def singlecell_matrix(
    out_dir: str,
    *,
    scope: str,
    layer: int,
    tokenizer=None,
    n_features: Optional[int] = None,
    model_name: Optional[str] = None,
) -> tuple:  # (csr_matrix, pd.DataFrame)
    """Build a token-level sparse feature matrix from sae_tokens records.

    Requires records to have been captured with record_token_level=True.

    Parameters
    ----------
    out_dir    : directory with manifest.jsonl and .pt records
    scope      : scope name, e.g. "out_unmask" (must be an unmask scope)
    layer      : transformer layer index
    tokenizer  : HuggingFace tokenizer for decoding token_ids → strings (optional)
    n_features : number of SAE features; inferred from data if None
    model_name : filter to a single model (optional)

    Returns
    -------
    X_sparse : csr_matrix [N_tok, n_features]  (float32)
    obs_tok  : DataFrame [N_tok] with columns:
               case_id, variant, group, token_id, token_str,
               judge_success, position
    """
    from scipy import sparse

    manifest = load_manifest(out_dir)
    if model_name is not None:
        manifest = manifest[manifest["model_name"] == model_name].copy()

    data_vals, data_rows, data_cols = [], [], []
    obs_rows = []
    row_ptr = 0
    n_skip = 0
    max_col = 0

    for _, mrow in manifest.iterrows():
        path = mrow.get("path")
        if not path or not os.path.exists(path):
            n_skip += 1
            continue
        try:
            rec = load_record(path)
        except Exception as e:
            warnings.warn(f"Could not load {path}: {e}")
            n_skip += 1
            continue

        tok_store = rec.get("sae_tokens", {}).get(scope)
        if tok_store is None or layer not in tok_store:
            n_skip += 1
            continue

        entry = tok_store[layer]
        token_ids_t = entry["token_ids"]   # (T,)  int32 tensor
        idx_t = entry["idx"]               # (T, k) int32 tensor
        val_t = entry["val"]               # (T, k) float16 tensor

        T = token_ids_t.shape[0]
        token_ids = token_ids_t.numpy()
        idx_np = idx_t.numpy()
        val_np = val_t.float().numpy()

        judge_s = mrow.get("judge_success")
        variant = mrow.get("variant", "")
        group = mrow.get("group", "")

        for t in range(T):
            tok_id = int(token_ids[t])
            if tokenizer is not None:
                tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
            else:
                tok_str = str(tok_id)

            obs_rows.append({
                "case_id":      mrow.get("case_id"),
                "variant":      variant,
                "group":        group,
                "token_id":     tok_id,
                "token_str":    tok_str,
                "judge_success": judge_s,
                "position":     t,
            })

            for ki in range(idx_np.shape[1]):
                feat_idx = int(idx_np[t, ki])
                feat_val = float(val_np[t, ki])
                if feat_val != 0.0:
                    data_rows.append(row_ptr)
                    data_cols.append(feat_idx)
                    data_vals.append(feat_val)
                    if feat_idx > max_col:
                        max_col = feat_idx

            row_ptr += 1

    if n_skip:
        warnings.warn(
            f"singlecell_matrix: skipped {n_skip} records "
            f"(missing file / no sae_tokens for scope={scope!r} layer={layer})."
        )

    if row_ptr == 0:
        raise ValueError(
            f"No token-level data found for scope={scope!r} layer={layer}. "
            "Re-run recording with record_token_level=True."
        )

    _n_feat = n_features if n_features is not None else (max_col + 1)
    X_sparse = sparse.csr_matrix(
        (np.array(data_vals, dtype=np.float32),
         (np.array(data_rows, dtype=np.int32),
          np.array(data_cols, dtype=np.int32))),
        shape=(row_ptr, _n_feat),
        dtype=np.float32,
    )
    obs_tok = pd.DataFrame(obs_rows).reset_index(drop=True)
    return X_sparse, obs_tok
