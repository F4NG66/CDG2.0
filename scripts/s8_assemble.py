#!/usr/bin/env python3
"""Module 1 – Assemble & QC.

Builds a pseudobulk feature matrix from pooled SAE activations, runs feature-level
QC (dead / ubiquitous), and saves the assembled data for downstream modules.

Usage
-----
python scripts/s8_assemble.py outputs/ \
    --scope out_mask --layer 16 --frac 0.10 \
    --out-dir analysis_output/assembled

Outputs (under --out-dir)
---------
  <scope>_L<layer>_f<frac>.npy      float32 X matrix [N, n_features]
  <scope>_L<layer>_f<frac>_obs.csv  observation metadata [N rows]
  <scope>_L<layer>_f<frac>_var.csv  feature metadata [n_features rows]

Design invariant: NO normalization here.  Raw activation scale is preserved for
delta_de.py and steering.  log1p is only applied in modules.py for clustering.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cdg.analysis.io import pseudobulk_matrix


# ---------------------------------------------------------------------------
# feature QC
# ---------------------------------------------------------------------------

def feature_qc(X: np.ndarray, ubiq_frac: float = 0.90, ubiq_var_thresh: float = 1e-4
               ) -> pd.DataFrame:
    """Compute per-feature QC metrics.

    Parameters
    ----------
    X            : [N, n_features] raw activations
    ubiq_frac    : fraction of samples that must be active to be called ubiquitous
    ubiq_var_thresh : variance threshold below which an active feature is ubiquitous

    Returns
    -------
    var DataFrame with columns:
      feature_id, mean_act, frac_active, is_dead, is_ubiquitous
    """
    n_samples, n_features = X.shape
    mean_act = X.mean(axis=0)
    frac_active = (X > 0).mean(axis=0)
    var_act = X.var(axis=0)

    is_dead = (frac_active == 0.0)
    # ubiquitous: active in >ubiq_frac of samples AND very low variance
    is_ubiquitous = (frac_active > ubiq_frac) & (var_act < ubiq_var_thresh)

    var_df = pd.DataFrame({
        "feature_id":    np.arange(n_features, dtype=np.int32),
        "mean_act":      mean_act.astype(np.float32),
        "frac_active":   frac_active.astype(np.float32),
        "var_act":       var_act.astype(np.float32),
        "is_dead":       is_dead,
        "is_ubiquitous": is_ubiquitous,
    })
    return var_df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SAE pseudobulk assembly + QC")
    ap.add_argument("out_dir", help="Run output directory (contains manifest.jsonl)")
    ap.add_argument("--scope",   default="out_mask",
                    help="Scope name (default: out_mask). Use 'tpl_mask' for "
                         "template-injection-specific analysis (only B/C cases).")
    ap.add_argument("--layer",  type=int, default=16)
    ap.add_argument("--frac",   type=float, default=0.10)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--save-dir", default=None,
                    help="Output directory (default: <out_dir>/assembled)")
    ap.add_argument("--ubiq-frac", type=float, default=0.90,
                    help="Fraction of samples active to call a feature ubiquitous")
    ap.add_argument("--ubiq-var", type=float, default=1e-4,
                    help="Max variance for ubiquitous feature call")
    ap.add_argument("--anndata", action="store_true",
                    help="Also export an AnnData .h5ad file (requires anndata)")
    args = ap.parse_args()

    save_dir = args.save_dir or os.path.join(args.out_dir, "assembled")
    os.makedirs(save_dir, exist_ok=True)

    tag = f"{args.scope}_L{args.layer}_f{args.frac:.2f}".replace(".", "p")

    print(f"[s8_assemble] Loading pseudobulk: scope={args.scope!r} "
          f"layer={args.layer} frac={args.frac}")
    X, obs = pseudobulk_matrix(
        args.out_dir,
        scope=args.scope,
        layer=args.layer,
        frac=args.frac,
        model_name=args.model_name,
    )
    N, F = X.shape
    print(f"  → {N} generations × {F} features")

    # QC
    var_df = feature_qc(X, ubiq_frac=args.ubiq_frac,
                        ubiq_var_thresh=args.ubiq_var)
    n_dead  = var_df["is_dead"].sum()
    n_ubiq  = var_df["is_ubiquitous"].sum()
    n_live  = (~var_df["is_dead"]).sum()
    print(f"  QC: {n_dead} dead ({100*n_dead/F:.1f}%), "
          f"{n_ubiq} ubiquitous, "
          f"{n_live} live features")

    # Save
    x_path   = os.path.join(save_dir, f"{tag}.npy")
    obs_path = os.path.join(save_dir, f"{tag}_obs.csv")
    var_path = os.path.join(save_dir, f"{tag}_var.csv")

    np.save(x_path, X)
    obs.to_csv(obs_path, index=False)
    var_df.to_csv(var_path, index=False)
    print(f"  Saved: {x_path}")
    print(f"         {obs_path}")
    print(f"         {var_path}")

    # Optional AnnData export
    if args.anndata:
        try:
            import anndata as ad
            adata = ad.AnnData(X=X, obs=obs.set_index("case_id"),
                               var=var_df.set_index("feature_id").astype(str))
            h5_path = os.path.join(save_dir, f"{tag}.h5ad")
            adata.write_h5ad(h5_path)
            print(f"         {h5_path}  (AnnData)")
        except ImportError:
            print("  [warn] anndata not installed; skipping .h5ad export")

    # Summary stats
    n_groups = obs["group"].value_counts()
    print("\n  Group composition:")
    for g, cnt in n_groups.sort_index().items():
        print(f"    {g}: {cnt} generations")

    active_per_gen = (X > 0).sum(axis=1)
    print(f"\n  Active features / generation: "
          f"mean={active_per_gen.mean():.1f} "
          f"median={np.median(active_per_gen):.1f} "
          f"max={active_per_gen.max():.0f}")
    return X, obs, var_df


if __name__ == "__main__":
    main()
