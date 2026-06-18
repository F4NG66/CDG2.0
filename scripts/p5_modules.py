#!/usr/bin/env python3
"""Module 3 – Co-activation Modules + Hierarchical Clustering + Enrichment.

Finds "feature programs" (co-activating sets of SAE features analogous to gene
modules in scRNA-seq) and tests whether jailbreak vs sycophancy markers fall in
different modules (module-level orthogonality).

Memory note: n_features=16384 × n_features feature-correlation matrix is ~1 GB.
We default to working on the ACTIVE feature subset (features with frac_active > 0
in the pseudobulk; typically < 2000 for our small N).  Adjust --min-frac-active.

log1p normalization is applied here (for clustering ONLY); it is NOT written back
to the saved X matrix or used for Δ/steering.

Usage
-----
python scripts/s10_modules.py outputs/ \
    --scope out_mask --layer 16 --frac 0.10 \
    --marker-dir outputs/transcriptome/markers

Outputs (under <out_dir>/analysis/modules/)
---------
  feature_module.csv            feature_id → module_id (all active features)
  module_attack_enrichment.csv  hypergeometric enrichment per (module, marker_set)
  linkage.npy                   scipy linkage matrix for the active feature subset
  corr_matrix.npy               feature×feature Pearson correlation (active subset)
  silhouette_scores.json        silhouette vs n_clusters for elbow selection
  corr_heatmap.png              clustered correlation matrix + dendrogram
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cdg.analysis.io import pseudobulk_matrix


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

def pearson_feature_corr(X_log: np.ndarray) -> np.ndarray:
    """Compute feature×feature Pearson correlation from log1p-transformed X.

    X_log : [N_samples, n_active_features]
    Returns [n_active, n_active] correlation matrix.
    Degenerate (zero-variance) features yield 0 correlation to everything.
    """
    X_c = X_log - X_log.mean(axis=0, keepdims=True)
    std = X_c.std(axis=0) + 1e-12
    X_z = X_c / std
    corr = (X_z.T @ X_z) / X_log.shape[0]
    np.clip(corr, -1, 1, out=corr)
    return corr.astype(np.float32)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def hierarchical_cluster(
    corr: np.ndarray,
    n_clusters_range: range,
    linkage_method: str = "ward",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict, int]:
    """Run hierarchical clustering and pick n_clusters by silhouette.

    Parameters
    ----------
    corr             : [n, n] correlation matrix
    n_clusters_range : range of n_clusters to evaluate
    linkage_method   : "ward" | "average" | "complete"

    Returns
    -------
    linkage_mat  : scipy linkage matrix
    labels       : cluster labels [n] for the best n_clusters
    sil_scores   : {n_clusters: silhouette_score}
    best_k       : chosen n_clusters
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Convert correlation to distance
    dist = np.clip(1 - corr, 0, 2)
    np.fill_diagonal(dist, 0)

    # squareform expects a condensed distance vector
    cond = squareform(dist, checks=False)
    link = linkage(cond, method=linkage_method)

    sil_scores = {}
    best_k = list(n_clusters_range)[0]
    best_sil = -1.0

    try:
        from sklearn.metrics import silhouette_score
        for k in n_clusters_range:
            if k >= corr.shape[0]:
                continue
            labs = fcluster(link, k, criterion="maxclust")
            if len(set(labs)) < 2:
                continue
            s = silhouette_score(dist, labs, metric="precomputed")
            sil_scores[int(k)] = float(s)
            if s > best_sil:
                best_sil = s
                best_k = int(k)
    except ImportError:
        # fallback: just use midpoint of range
        warnings.warn("sklearn not available; silhouette skipped, using default k")
        best_k = list(n_clusters_range)[len(n_clusters_range) // 2]

    labels = fcluster(link, best_k, criterion="maxclust")
    return link, labels, sil_scores, best_k


# ---------------------------------------------------------------------------
# hypergeometric enrichment
# ---------------------------------------------------------------------------

def hypergeometric_pval(k: int, M: int, n: int, N: int) -> float:
    """One-sided (upper tail) P(X >= k) for Hypergeometric(M, n, N).

    M : total population (n_active_features)
    n : number of successes in population (marker set size)
    N : number of draws (module size)
    k : observed successes (markers in module)

    Returns 1.0 if k == 0 (no enrichment), or n == 0 (no markers in population).
    """
    if k == 0 or n == 0:
        return 1.0
    try:
        from scipy.stats import hypergeom
        return float(hypergeom.sf(k - 1, M, n, N))
    except ImportError:
        p = n / max(M, 1)
        from math import comb
        total = sum(comb(N, i) * (p ** i) * ((1 - p) ** (N - i))
                    for i in range(k, N + 1))
        return float(total)


def plot_clustered_heatmap(
    corr: np.ndarray,
    labels: np.ndarray,
    linkage_mat: np.ndarray,
    feature_ids: np.ndarray,
    save_path: str,
    max_features: int = 200,
) -> None:
    """Draw dendrogram + clustered correlation heatmap and save to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from scipy.cluster.hierarchy import dendrogram
        from scipy.spatial.distance import squareform
    except ImportError:
        warnings.warn("matplotlib not available; skipping heatmap")
        return

    n = corr.shape[0]
    # Subsample if too large for readable plot
    if n > max_features:
        rng = np.random.default_rng(0)
        # Keep a proportional sample from each module
        keep = []
        for mid in sorted(set(labels)):
            mod_idx = np.where(labels == mid)[0]
            n_keep = max(1, int(max_features * len(mod_idx) / n))
            keep.extend(rng.choice(mod_idx, size=min(n_keep, len(mod_idx)),
                                   replace=False).tolist())
        keep = sorted(keep)
        corr = corr[np.ix_(keep, keep)]
        labels = labels[keep]
        feature_ids = feature_ids[keep]
        # rebuild linkage from subsampled corr for dendrogram leaf ordering
        from scipy.cluster.hierarchy import linkage as _link
        from scipy.spatial.distance import squareform as _sq
        dist = np.clip(1 - corr, 0, 2)
        np.fill_diagonal(dist, 0)
        linkage_mat = _link(_sq(dist, checks=False), method="ward")

    # Leaf ordering from dendrogram
    dg = dendrogram(linkage_mat, no_plot=True)
    order = dg["leaves"]
    corr_ord = corr[np.ix_(order, order)]
    labels_ord = labels[order]

    # Module boundary positions
    boundaries = []
    prev = labels_ord[0]
    for i, m in enumerate(labels_ord):
        if m != prev:
            boundaries.append(i)
            prev = m

    # ── Layout: dendrogram left + heatmap right ──────────────────────────
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, width_ratios=[1, 8], height_ratios=[1, 8],
                  hspace=0.02, wspace=0.02)
    ax_dg_left = fig.add_subplot(gs[1, 0])
    ax_dg_top  = fig.add_subplot(gs[0, 1])
    ax_heat    = fig.add_subplot(gs[1, 1])

    # Dendrogram (left, horizontal)
    dendrogram(linkage_mat, ax=ax_dg_left, orientation="left",
               no_labels=True, link_color_func=lambda k: "#555555",
               above_threshold_color="#555555")
    ax_dg_left.set_xticks([])
    ax_dg_left.set_yticks([])
    ax_dg_left.axis("off")

    # Dendrogram (top, vertical) — same tree mirrored
    dendrogram(linkage_mat, ax=ax_dg_top, orientation="top",
               no_labels=True, link_color_func=lambda k: "#555555",
               above_threshold_color="#555555")
    ax_dg_top.set_xticks([])
    ax_dg_top.set_yticks([])
    ax_dg_top.axis("off")

    # Heatmap
    im = ax_heat.imshow(corr_ord, aspect="auto", cmap="RdBu_r",
                        vmin=-1, vmax=1, interpolation="nearest")
    for b in boundaries:
        ax_heat.axvline(b - 0.5, color="black", lw=0.6, alpha=0.7)
        ax_heat.axhline(b - 0.5, color="black", lw=0.6, alpha=0.7)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])

    # Module labels on right axis
    module_ids = sorted(set(labels_ord))
    for mid in module_ids:
        positions = np.where(labels_ord == mid)[0]
        center = positions.mean()
        ax_heat.text(len(order) + 1, center, f"M{mid}",
                     va="center", ha="left", fontsize=7,
                     color="black")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.01)
    cbar.set_label("Pearson r (log1p)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"SAE Feature Co-activation Modules (n={len(order)} active features, "
        f"k={len(set(labels_ord))} modules)",
        fontsize=10, y=0.98,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {save_path}")


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    adj = pvals * n / ranks
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return np.clip(adj, 0, 1)


def module_enrichment(
    feature_ids: np.ndarray,
    labels: np.ndarray,
    marker_sets_dict: dict[str, set],
) -> pd.DataFrame:
    """Hypergeometric enrichment of marker sets in each module.

    Parameters
    ----------
    feature_ids     : [n_active] global feature indices
    labels          : [n_active] cluster labels (1-based from scipy)
    marker_sets_dict: {marker_set_name → set of global feature_ids}

    Returns
    -------
    DataFrame: module_id, marker_set, n_markers_in_module, module_size,
               marker_set_size, n_active, pval, fdr, enrichment_ratio
    """
    M = len(feature_ids)
    modules = sorted(set(labels))
    rows = []

    all_pvals = []
    for mod_id in modules:
        mod_mask = labels == mod_id
        mod_features = set(feature_ids[mod_mask].tolist())
        N = len(mod_features)
        for ms_name, ms_feats in marker_sets_dict.items():
            n = len(ms_feats & set(feature_ids.tolist()))  # markers that are active
            k = len(mod_features & ms_feats)
            p = hypergeometric_pval(k, M, n, N)
            expected = N * n / max(M, 1)
            enr = k / expected if expected > 0 else 0.0
            rows.append({
                "module_id":        int(mod_id),
                "marker_set":       ms_name,
                "n_markers_in_module": k,
                "module_size":      N,
                "marker_set_size":  n,
                "n_active":         M,
                "pval":             p,
                "enrichment_ratio": float(enr),
            })
            all_pvals.append(p)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["fdr"] = _bh_fdr(np.array(all_pvals))
    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Feature co-activation modules")
    ap.add_argument("out_dir")
    ap.add_argument("--scope",   default="tpl_mask",
                    help="Scope for co-activation analysis. Default: tpl_mask "
                         "(injection blanks — captures the template-injection signal "
                         "directly). Note: tpl_mask is only non-NaN for groups B and C "
                         "(N=6). Use out_mask for all 4 groups (N=12).")
    ap.add_argument("--layer",   type=int, default=16)
    ap.add_argument("--frac",    type=float, default=0.10)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--marker-dir", default=None,
                    help="Directory containing <pair>_up.csv / _dn.csv from s9_delta_de")
    ap.add_argument("--min-frac-active", type=float, default=0.0,
                    help="Only cluster features active in >this fraction of samples "
                         "(default=0, i.e. any non-dead feature)")
    ap.add_argument("--linkage-method", default="ward",
                    choices=["ward", "average", "complete"])
    ap.add_argument("--k-min",  type=int, default=3,
                    help="Min n_clusters to evaluate (silhouette)")
    ap.add_argument("--k-max",  type=int, default=15,
                    help="Max n_clusters to evaluate")
    ap.add_argument("--save-dir", default=None,
                    help="Output dir (default: <out_dir>/analysis/modules)")
    args = ap.parse_args()

    base = args.save_dir or os.path.join(args.out_dir, "transcriptome", "modules")
    os.makedirs(base, exist_ok=True)

    print(f"[s10_modules] Loading pseudobulk: scope={args.scope!r} "
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

    # Filter to active features
    frac_active = (X > 0).mean(axis=0)
    active_mask = frac_active > args.min_frac_active
    active_ids = np.where(active_mask)[0]
    X_active = X[:, active_mask]
    n_active = X_active.shape[1]
    print(f"  Active features (frac_active > {args.min_frac_active}): {n_active}")

    if n_active < 4:
        print("  Too few active features for clustering; exiting.")
        return

    # log1p normalization (clustering only — NOT written back to delta)
    X_log = np.log1p(X_active)

    # -----------------------------------------------------------------------
    # 1. Feature correlation matrix
    # -----------------------------------------------------------------------
    print("  Computing feature×feature Pearson correlation (log1p)…")
    corr = pearson_feature_corr(X_log)
    np.save(os.path.join(base, "corr_matrix.npy"), corr)
    print(f"  Correlation matrix: {corr.shape} → {os.path.join(base, 'corr_matrix.npy')}")

    # -----------------------------------------------------------------------
    # 2. Hierarchical clustering + silhouette selection
    # -----------------------------------------------------------------------
    k_range = range(args.k_min, min(args.k_max + 1, n_active))
    print(f"  Hierarchical clustering (method={args.linkage_method!r}, "
          f"k={args.k_min}…{args.k_max})…")
    link, labels, sil_scores, best_k = hierarchical_cluster(
        corr, k_range, linkage_method=args.linkage_method
    )
    np.save(os.path.join(base, "linkage.npy"), link)

    # Clustered correlation heatmap
    heatmap_path = os.path.join(base, "corr_heatmap.png")
    plot_clustered_heatmap(corr, labels, link, active_ids, heatmap_path)

    sil_path = os.path.join(base, "silhouette_scores.json")
    with open(sil_path, "w") as f:
        json.dump({"best_k": best_k, "scores": sil_scores}, f, indent=2)
    print(f"  Best k = {best_k}  (silhouette → {sil_path})")

    # Save feature → module mapping (all active features)
    module_df = pd.DataFrame({
        "feature_id": active_ids.astype(np.int32),
        "module_id":  labels.astype(np.int32),
        "frac_active": frac_active[active_mask].astype(np.float32),
    })
    module_df.to_csv(os.path.join(base, "feature_module.csv"), index=False)
    print(f"  Module assignments → {os.path.join(base, 'feature_module.csv')}")

    # Module summary
    mod_counts = pd.Series(labels).value_counts().sort_index()
    print(f"\n  Module sizes (top 10):")
    for mid, cnt in mod_counts.head(10).items():
        print(f"    module {mid}: {cnt} features")

    # -----------------------------------------------------------------------
    # 3. Marker enrichment
    # -----------------------------------------------------------------------
    marker_sets_dict = {}
    marker_dir = args.marker_dir or os.path.join(args.out_dir, "transcriptome", "markers")
    if os.path.isdir(marker_dir):
        for fname in sorted(os.listdir(marker_dir)):
            if not fname.endswith("_up.csv") and not fname.endswith("_dn.csv"):
                continue
            try:
                mdf = pd.read_csv(os.path.join(marker_dir, fname))
                key = fname.replace(".csv", "")
                marker_sets_dict[key] = set(mdf["feature_id"].tolist())
            except Exception as e:
                warnings.warn(f"Could not load {fname}: {e}")
    else:
        print(f"  [warn] Marker directory not found: {marker_dir}; skipping enrichment")

    if marker_sets_dict:
        print(f"\n  Enrichment analysis ({len(marker_sets_dict)} marker sets)…")
        enr_df = module_enrichment(active_ids, labels, marker_sets_dict)

        enr_path = os.path.join(base, "module_attack_enrichment.csv")
        enr_df.to_csv(enr_path, index=False)
        print(f"  Enrichment → {enr_path}")

        # Report top hits
        sig = enr_df[enr_df["fdr"] < 0.05].sort_values("enrichment_ratio",
                                                         ascending=False)
        if len(sig):
            print(f"\n  Significant enrichments (FDR<0.05):")
            for _, r in sig.head(10).iterrows():
                print(f"    module {r['module_id']:3d} × {r['marker_set']:30s} "
                      f"k={r['n_markers_in_module']:3d}/{r['module_size']:3d}  "
                      f"enr={r['enrichment_ratio']:.2f}  FDR={r['fdr']:.3g}")
        else:
            print("  No significant enrichments (FDR<0.05); likely insufficient N.")
            sig_nom = enr_df[enr_df["pval"] < 0.05].sort_values("enrichment_ratio",
                                                                  ascending=False)
            if len(sig_nom):
                print(f"  Nominally significant (p<0.05, top 5):")
                for _, r in sig_nom.head(5).iterrows():
                    print(f"    module {r['module_id']:3d} × {r['marker_set']:30s} "
                          f"enr={r['enrichment_ratio']:.2f}  p={r['pval']:.3g}")
    else:
        print("  No marker sets found; skipping enrichment.")

    print("\n[s10_modules] Done.")


if __name__ == "__main__":
    main()
