#!/usr/bin/env python3
"""Module 2 – Δ Vectors + Differential Feature Analysis + Orthogonality.

Upgrades the existing cosine Feature Atlas with:
  1. Per-attack Δ vectors (raw activation scale, for steering)
  2. Per-feature DE: Mann–Whitney U + BH-FDR per (attack vs baseline) pair
  3. Marker feature sets (FDR<0.05, |effect|>threshold)
  4. Cosine orthogonality matrix between Δ vectors
  5. Jaccard overlap between marker sets + bootstrap CI

Statistical unit: ONE GENERATION (pseudobulk row).  Never token-level.

Attack/baseline mapping for the 2×2 design (A/B/C/D):
  B (harmful_injected)  vs A (harmful_clean)     → injection effect on harm
  C (neutral_injected)  vs D (neutral_clean)     → injection effect on neutral
  B vs C                                         → harmful vs neutral within injection
  (Add more pairs via --pairs if dataset grows)

Usage
-----
python scripts/s9_delta_de.py outputs/ \
    --scope out_mask --layer 16 --frac 0.10

Outputs (all under <out_dir>/transcriptome/)
---------
  delta/Delta_<pair>.npy          per-pair Δ vector [n_features]
  delta/Delta_attack.npy          mean of all attack Δ vectors
  markers/<pair>_up.csv           up-regulated markers (FDR<0.05, effect>thresh)
  markers/<pair>_dn.csv           down-regulated markers
  atlas/cosine_atlas.csv          Δ-vector cosine similarity matrix
  atlas/group_overlap.json        Jaccard + bootstrap CI results
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
# statistical helpers
# ---------------------------------------------------------------------------

def mannwhitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann–Whitney U test; returns (U_stat, p_value).

    Uses scipy if available, falls back to a vectorised exact-ish approximation.
    """
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        return float(stat), float(p)
    except ImportError:
        # Normal approximation fallback
        nx, ny = len(x), len(y)
        U = float(np.sum(x[:, None] > y[None, :]) +
                  0.5 * np.sum(x[:, None] == y[None, :]))
        mean_U = nx * ny / 2
        std_U = np.sqrt(nx * ny * (nx + ny + 1) / 12)
        z = (U - mean_U) / (std_U + 1e-12)
        from math import erfc, sqrt
        p = float(erfc(abs(z) / sqrt(2)))
        return U, p


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation as effect size for Mann–Whitney U."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    U, _ = mannwhitney_u(x, y)
    return float(2 * U / (nx * ny) - 1)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-SD Cohen's d (signed: positive = x > y)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float(np.mean(x) - np.mean(y))
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) +
                      (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2) + 1e-12)
    return float((np.mean(x) - np.mean(y)) / pooled)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    adj = pvals * n / ranks
    # enforce monotonicity from the right
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return np.clip(adj, 0, 1)


def de_per_feature(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    effect: str = "rank_biserial",
) -> pd.DataFrame:
    """Run per-feature DE between two pseudobulk groups.

    Parameters
    ----------
    X_pos / X_neg : [n_samples, n_features]
    effect        : "rank_biserial" or "cohens_d"

    Returns
    -------
    DataFrame with columns: feature_id, mean_pos, mean_neg, delta,
                             effect_size, pval, fdr
    Sorted by FDR then |effect_size|.
    """
    if X_pos.shape[0] < 2 or X_neg.shape[0] < 2:
        warnings.warn(
            f"DE has only {X_pos.shape[0]} pos and {X_neg.shape[0]} neg samples; "
            "p-values will be unreliable."
        )

    n_feat = X_pos.shape[1]
    mean_pos = X_pos.mean(axis=0)
    mean_neg = X_neg.mean(axis=0)
    delta = mean_pos - mean_neg

    pvals = np.ones(n_feat)
    effect_sizes = np.zeros(n_feat)

    for f in range(n_feat):
        xp = X_pos[:, f]
        xn = X_neg[:, f]
        if np.all(xp == 0) and np.all(xn == 0):
            continue
        _, p = mannwhitney_u(xp, xn)
        pvals[f] = p
        if effect == "rank_biserial":
            effect_sizes[f] = rank_biserial(xp, xn)
        else:
            effect_sizes[f] = cohens_d(xp, xn)

    fdr = bh_fdr(pvals)

    df = pd.DataFrame({
        "feature_id":   np.arange(n_feat, dtype=np.int32),
        "mean_pos":     mean_pos.astype(np.float32),
        "mean_neg":     mean_neg.astype(np.float32),
        "delta":        delta.astype(np.float32),
        "effect_size":  effect_sizes.astype(np.float32),
        "pval":         pvals.astype(np.float64),
        "fdr":          fdr.astype(np.float64),
    })
    df = df.sort_values(["fdr", "effect_size"], key=lambda c: c.abs() if c.name == "effect_size" else c,
                        ascending=[True, False]).reset_index(drop=True)
    return df


def marker_sets(de_df: pd.DataFrame, fdr_thresh: float = 0.05,
                effect_thresh: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DE results into up/down marker sets."""
    sig = de_df[(de_df["fdr"] < fdr_thresh) & (de_df["effect_size"].abs() > effect_thresh)]
    up = sig[sig["effect_size"] > 0].copy()
    dn = sig[sig["effect_size"] < 0].copy()
    return up, dn


# ---------------------------------------------------------------------------
# orthogonality
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard index; returns NaN if both sets are empty (no markers found)."""
    if not set_a and not set_b:
        return float("nan")
    union = len(set_a | set_b)
    if union == 0:
        return float("nan")
    return len(set_a & set_b) / union


def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the mean of a list of scalars."""
    rng = np.random.default_rng(seed)
    v = np.array(values, dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    boots = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 - (1 - ci) / 2))
    return {"mean": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# All pairwise comparisons in the 2×2 design.
# Format: (attack_group, baseline_group, label)
#
#  Injection effect:  B_vs_A (harmful), C_vs_D (neutral)
#  Content effect:    B_vs_C (within injection), A_vs_D (without injection)
#  Cross effects:     B_vs_D (full attack vs double baseline),
#                     A_vs_C (harmful-clean vs neutral-injected — tests whether
#                             neutral injection alone is "harder" than clean harmful)
DEFAULT_PAIRS = [
    ("B", "A", "B_vs_A"),   # injection + content  vs  clean baseline
    ("C", "D", "C_vs_D"),   # injection (neutral)  vs  neutral clean
    ("B", "C", "B_vs_C"),   # harmful  vs  neutral  (within injection)
    ("A", "D", "A_vs_D"),   # harmful  vs  neutral  (both clean, no template)
    ("B", "D", "B_vs_D"),   # full attack  vs  double baseline
    ("A", "C", "A_vs_C"),   # harmful clean  vs  neutral injected
]


def main():
    ap = argparse.ArgumentParser(description="Delta vectors + Differential Feature Analysis")
    ap.add_argument("out_dir")
    ap.add_argument("--scope",   default="out_mask")
    ap.add_argument("--layer",   type=int, default=16)
    ap.add_argument("--frac",    type=float, default=0.10)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--fdr-thresh",    type=float, default=0.05)
    ap.add_argument("--effect-thresh", type=float, default=0.1,
                    help="Min |rank-biserial| to be a marker")
    ap.add_argument("--effect-type",   default="rank_biserial",
                    choices=["rank_biserial", "cohens_d"])
    ap.add_argument("--n-boot",  type=int, default=2000)
    ap.add_argument("--save-dir", default=None,
                    help="Base output directory (default: <out_dir>/transcriptome)")
    args = ap.parse_args()

    base = args.save_dir or os.path.join(args.out_dir, "transcriptome")
    delta_dir   = os.path.join(base, "delta")
    marker_dir  = os.path.join(base, "markers")
    atlas_dir   = os.path.join(base, "atlas")
    for d in (delta_dir, marker_dir, atlas_dir):
        os.makedirs(d, exist_ok=True)

    print(f"[s9_delta_de] Loading pseudobulk: scope={args.scope!r} "
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

    # Index observations by group for easy access
    group_idx = {g: obs.index[obs["group"] == g].tolist()
                 for g in obs["group"].unique()}

    # -----------------------------------------------------------------------
    # 1. Per-pair Δ and DE
    # -----------------------------------------------------------------------
    delta_vecs = {}
    all_markers_up = {}
    all_markers_dn = {}

    for atk, base_grp, label in DEFAULT_PAIRS:
        pos_idx = group_idx.get(atk, [])
        neg_idx = group_idx.get(base_grp, [])
        if not pos_idx or not neg_idx:
            print(f"  [skip] {label}: missing group {atk!r} or {base_grp!r}")
            continue

        X_pos = X[pos_idx]
        X_neg = X[neg_idx]
        delta = X_pos.mean(axis=0) - X_neg.mean(axis=0)
        delta_vecs[label] = delta

        # Save Δ (raw scale)
        np.save(os.path.join(delta_dir, f"Delta_{label}.npy"), delta)
        print(f"  {label}: Δ norm={np.linalg.norm(delta):.4f}  "
              f"(n_pos={len(pos_idx)}, n_neg={len(neg_idx)})")

        # DE
        de_df = de_per_feature(X_pos, X_neg, effect=args.effect_type)
        de_df.to_csv(os.path.join(marker_dir, f"{label}_de.csv"), index=False)

        up, dn = marker_sets(de_df, fdr_thresh=args.fdr_thresh,
                             effect_thresh=args.effect_thresh)
        up.to_csv(os.path.join(marker_dir, f"{label}_up.csv"), index=False)
        dn.to_csv(os.path.join(marker_dir, f"{label}_dn.csv"), index=False)
        all_markers_up[label] = set(up["feature_id"].tolist())
        all_markers_dn[label] = set(dn["feature_id"].tolist())
        print(f"    markers: {len(up)} up, {len(dn)} down "
              f"(FDR<{args.fdr_thresh}, |effect|>{args.effect_thresh})")

        # Warn about sample size
        if len(pos_idx) < 5 or len(neg_idx) < 5:
            print(f"    ⚠ Small-N warning: only {len(pos_idx)}+{len(neg_idx)} samples; "
                  "statistical power is very low.  Effect sizes are more informative "
                  "than p-values here.")

    # -----------------------------------------------------------------------
    # 2. Group Δ: mean of primary attack Δ vectors (for steering)
    #    Primary attacks: B_vs_A (injection on harm), C_vs_D (injection on neutral)
    # -----------------------------------------------------------------------
    injection_pairs = [l for l in ("B_vs_A", "C_vs_D") if l in delta_vecs]
    if injection_pairs:
        delta_attack = np.stack([delta_vecs[l] for l in injection_pairs]).mean(axis=0)
        np.save(os.path.join(delta_dir, "Delta_attack.npy"), delta_attack)
        print(f"\n  Delta_attack (mean of {injection_pairs}): norm={np.linalg.norm(delta_attack):.4f}")

    # -----------------------------------------------------------------------
    # 3. Cosine atlas between Δ vectors
    # -----------------------------------------------------------------------
    if len(delta_vecs) >= 2:
        labels = sorted(delta_vecs.keys())
        n = len(labels)
        cos_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cos_mat[i, j] = cosine_sim(delta_vecs[labels[i]],
                                           delta_vecs[labels[j]])
        atlas_df = pd.DataFrame(cos_mat, index=labels, columns=labels)
        atlas_path = os.path.join(atlas_dir, "cosine_atlas.csv")
        atlas_df.to_csv(atlas_path)
        print(f"\n  Δ-vector cosine atlas ({n}×{n}) → {atlas_path}")
        print(atlas_df.round(3).to_string())

    # -----------------------------------------------------------------------
    # 4. Jaccard overlap between marker sets
    # -----------------------------------------------------------------------
    group_overlap = {}
    pair_labels = sorted(all_markers_up.keys())

    # Define attack groups for within/between comparisons
    # "injection" pairs: B_vs_A, C_vs_D  (both measure injection effect)
    # "content" pair: B_vs_C             (harmful vs neutral within injection)
    injection_markers = {l: (all_markers_up[l] | all_markers_dn[l])
                         for l in ["B_vs_A", "C_vs_D"] if l in all_markers_up}
    content_markers   = {l: (all_markers_up[l] | all_markers_dn[l])
                         for l in ["B_vs_C"]           if l in all_markers_up}

    # Within-injection Jaccard
    inj_keys = sorted(injection_markers.keys())
    within_jac = []
    for i in range(len(inj_keys)):
        for j in range(i + 1, len(inj_keys)):
            j_val = jaccard(injection_markers[inj_keys[i]],
                            injection_markers[inj_keys[j]])
            within_jac.append(j_val)
            group_overlap[f"within_injection_{inj_keys[i]}_{inj_keys[j]}"] = j_val

    # Between injection and content markers
    between_jac = []
    for il in inj_keys:
        for cl in sorted(content_markers.keys()):
            j_val = jaccard(injection_markers[il], content_markers[cl])
            between_jac.append(j_val)
            group_overlap[f"between_{il}_{cl}"] = j_val

    # Bootstrap CI
    group_overlap["within_injection_ci"] = bootstrap_ci(within_jac, args.n_boot)
    group_overlap["between_ci"] = bootstrap_ci(between_jac, args.n_boot)

    # Pairwise for all marker sets
    for i in range(len(pair_labels)):
        for j in range(i + 1, len(pair_labels)):
            li, lj = pair_labels[i], pair_labels[j]
            mi = all_markers_up.get(li, set()) | all_markers_dn.get(li, set())
            mj = all_markers_up.get(lj, set()) | all_markers_dn.get(lj, set())
            group_overlap[f"jaccard_{li}_{lj}"] = jaccard(mi, mj)

    overlap_path = os.path.join(atlas_dir, "group_overlap.json")
    with open(overlap_path, "w") as f:
        json.dump(group_overlap, f, indent=2, default=float)
    print(f"\n  Jaccard / bootstrap → {overlap_path}")
    for k, v in group_overlap.items():
        if isinstance(v, dict):
            print(f"    {k}: mean={v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
        else:
            print(f"    {k}: {v:.3f}")

    print("\n[s9_delta_de] Done.")


if __name__ == "__main__":
    main()
