#!/usr/bin/env python3
"""p4_delta.py — Δ Vectors + Differential Feature Analysis + Visualizations.

Merges the functionality of old s3_diff_features.py (quick mean-gap ranking)
and s9_delta_de.py (statistical DE + cosine orthogonality atlas).

Two modes:
  --quick   Fast: rank features by mean activation gap. Reads Phase-1 records
            directly. Good for exploration. (was s3_diff_features.py)
  --full    Statistical: Mann-Whitney U + BH-FDR + Δ vectors + atlas.
            Reads assembled matrices from p2_assemble.py. (was s9_delta_de.py)

Usage
-----
python scripts/p4_delta.py outputs/ --quick --scope tpl_mask --layer 16
python scripts/p4_delta.py outputs/ --full  --scope out_mask --layer 16 --frac 0.10

Outputs (under <out_dir>/analysis/delta/)
------------------------------------------
  Delta_{pair}.npy          Δ vector per comparison pair [n_features]
  Delta_attack.npy          mean injection Δ (for steering)
  {pair}_de.csv             full DE table per pair
  {pair}_up.csv / _dn.csv   significant marker sets
  cosine_atlas.csv          Δ-vector cosine similarity matrix (6×6)
  delta_atlas.png           cosine atlas heatmap
  top_features.png          top-20 features by |Δ| per key pair
  diff_features.json        quick ranking output (--quick mode)
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Statistical helpers (from s9_delta_de.py)
# ---------------------------------------------------------------------------

def mannwhitney_u(x, y):
    try:
        from scipy.stats import mannwhitneyu
        return mannwhitneyu(x, y, alternative="two-sided", method="auto")
    except ImportError:
        nx, ny = len(x), len(y)
        U = float(np.sum(x[:, None] > y[None, :]) +
                  0.5 * np.sum(x[:, None] == y[None, :]))
        mean_U = nx * ny / 2
        std_U  = np.sqrt(nx * ny * (nx + ny + 1) / 12)
        z = (U - mean_U) / (std_U + 1e-12)
        from math import erfc, sqrt
        return type("R", (), {"statistic": U, "pvalue": float(erfc(abs(z)/sqrt(2)))})()


def rank_biserial(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return 0.0
    U = mannwhitney_u(x, y).statistic
    return float(2 * U / (nx * ny) - 1)


def bh_fdr(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    adj = np.clip(pvals * n / ranks, 0, 1)
    adj_s = adj[order]
    for i in range(n - 2, -1, -1):
        adj_s[i] = min(adj_s[i], adj_s[i + 1])
    adj[order] = adj_s
    return adj


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def jaccard(sa, sb):
    if not sa and not sb: return float("nan")
    u = len(sa | sb)
    return len(sa & sb) / u if u else float("nan")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_cosine_atlas(atlas_df: pd.DataFrame, save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(atlas_df.index)
        mat = atlas_df.values.astype(float)
        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(6, n * 0.8 + 1), max(5, n * 0.8)))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black" if abs(v) < 0.7 else "white",
                        fontweight="bold" if i == j else "normal")
        plt.colorbar(im, ax=ax, label="Cosine similarity", fraction=0.035)
        ax.set_title("Δ-vector Cosine Orthogonality Atlas\n"
                     "Off-diagonal near 0 = independent feature programs", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Atlas heatmap → {save_path}")
    except ImportError:
        pass


def plot_top_features(delta_vecs: dict, pairs_to_show: list,
                      save_path: str, topk: int = 20) -> None:
    """Bar chart: top features by |Δ| for key comparison pairs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_pairs = len(pairs_to_show)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 6), squeeze=False)

        for pi, label in enumerate(pairs_to_show):
            ax = axes[0][pi]
            delta = delta_vecs.get(label)
            if delta is None:
                ax.set_visible(False)
                continue
            order = np.argsort(-np.abs(delta))[:topk]
            feat_ids = order[::-1]
            vals = delta[feat_ids]
            colors = ["#e15759" if v > 0 else "#4e79a7" for v in vals]
            ax.barh([f"feat {i}" for i in feat_ids], vals, color=colors, height=0.7)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("Δ (attack − baseline)")
            ax.set_title(f"{label}\ntop {topk} features by |Δ|", fontsize=9)
            ax.tick_params(axis="y", labelsize=7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Top-features bar chart → {save_path}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# All 6 comparison pairs
# ---------------------------------------------------------------------------

ALL_PAIRS = [
    ("B", "A", "B_vs_A"),   # injection + harmful  vs  harmful clean
    ("C", "D", "C_vs_D"),   # injection + neutral  vs  neutral clean
    ("B", "C", "B_vs_C"),   # harmful  vs  neutral  (within injection)
    ("A", "D", "A_vs_D"),   # harmful  vs  neutral  (both clean)
    ("B", "D", "B_vs_D"),   # full attack  vs  double baseline
    ("A", "C", "A_vs_C"),   # harmful clean  vs  neutral injected
]

INJECTION_PAIRS = ["B_vs_A", "C_vs_D"]


# ---------------------------------------------------------------------------
# Quick mode (reads Phase-1 records directly)
# ---------------------------------------------------------------------------

def run_quick(out_dir, scope, layer, frac, topk, save_dir):
    from cdg.probe import load_records, top_diff_features
    from cdg.config import get_backend_config
    cfg = get_backend_config("llada_attack")
    records = load_records(out_dir, cfg.name)
    print(f"  Loaded {len(records)} Phase-1 records")

    pairs = [("B","C"), ("A","B"), ("C","D"), ("A","D")]
    rows = top_diff_features(records, scope=scope, pairs=pairs, k=topk)
    focus = [r for r in rows
             if r["layer"] == layer and abs(r["frac"] - frac) < 1e-4]

    if not focus:
        print(f"  No data for scope={scope} layer={layer} frac={frac}")
        return

    print(f"\n  {'pair':10s}  {'feature':>8}  {'gap':>10}  {'pos':>8}  {'neg':>8}")
    for row in focus:
        print(f"\n  {row['pair']:10s} (n={row['n_pos']}+{row['n_neg']})")
        for f in row["features"][:topk]:
            mark = " ←" if f["score"] > 0 and row["pair"] == "BvsC" else ""
            print(f"  {'':10s}  {f['feature']:8d}  {f['score']:+10.4f}  "
                  f"{f['pos_mean']:8.4f}  {f['neg_mean']:8.4f}{mark}")

    out = {"layer": layer, "frac": frac, "scope": scope,
           "rows": rows, "focus_rows": focus}
    with open(os.path.join(save_dir, "diff_features.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    # also write to legacy path for s6_activation_trace / p6
    os.makedirs("analysis_output", exist_ok=True)
    with open("analysis_output/diff_features.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  Saved diff_features.json")


# ---------------------------------------------------------------------------
# Full mode (reads assembled matrices from p2)
# ---------------------------------------------------------------------------

def run_full(out_dir, scope, layer, frac, fdr_thresh, effect_thresh, save_dir):
    from cdg.analysis.io import pseudobulk_matrix

    # Try assembled matrix first, fall back to direct load
    assembled_x = os.path.join(out_dir, "assembled",
                                f"X_{scope}_L{layer}_f{frac:.2f}".replace(".","p") + ".npy")
    assembled_obs = assembled_x.replace("X_", "obs_").replace(".npy", ".csv")

    if os.path.exists(assembled_x):
        X = np.load(assembled_x).astype(np.float32)
        obs = pd.read_csv(assembled_obs)
        print(f"  Loaded assembled matrix: {X.shape}")
    else:
        print(f"  Assembled matrix not found; loading directly from records…")
        X, obs = pseudobulk_matrix(out_dir, scope=scope, layer=layer, frac=frac)
        print(f"  Loaded: {X.shape}")

    group_idx = {g: obs.index[obs["group"] == g].tolist()
                 for g in obs["group"].unique()}

    delta_vecs = {}
    marker_up = {}
    marker_dn = {}

    for atk, bas, label in ALL_PAIRS:
        pi = group_idx.get(atk, [])
        ni = group_idx.get(bas, [])
        if not pi or not ni:
            continue
        Xp, Xn = X[pi], X[ni]
        delta = Xp.mean(0) - Xn.mean(0)
        delta_vecs[label] = delta
        np.save(os.path.join(save_dir, f"Delta_{label}.npy"), delta)

        # Statistical DE
        n_feat = X.shape[1]
        pvals = np.ones(n_feat)
        effects = np.zeros(n_feat)
        if len(pi) >= 2 and len(ni) >= 2:
            for f in range(n_feat):
                xp, xn = Xp[:, f], Xn[:, f]
                if np.all(xp == 0) and np.all(xn == 0):
                    continue
                res = mannwhitney_u(xp, xn)
                pvals[f] = res.pvalue
                effects[f] = rank_biserial(xp, xn)
        fdr = bh_fdr(pvals)

        de_df = pd.DataFrame({
            "feature_id":  np.arange(n_feat, dtype=np.int32),
            "mean_pos":    Xp.mean(0).astype(np.float32),
            "mean_neg":    Xn.mean(0).astype(np.float32),
            "delta":       delta.astype(np.float32),
            "effect_size": effects.astype(np.float32),
            "pval":        pvals,
            "fdr":         fdr,
        })
        de_df.to_csv(os.path.join(save_dir, f"{label}_de.csv"), index=False)

        sig = de_df[(de_df["fdr"] < fdr_thresh) & (de_df["effect_size"].abs() > effect_thresh)]
        up = sig[sig["effect_size"] > 0]
        dn = sig[sig["effect_size"] < 0]
        up.to_csv(os.path.join(save_dir, f"{label}_up.csv"), index=False)
        dn.to_csv(os.path.join(save_dir, f"{label}_dn.csv"), index=False)
        marker_up[label] = set(up["feature_id"].tolist())
        marker_dn[label] = set(dn["feature_id"].tolist())

        n_warn = " ⚠ small-N" if min(len(pi), len(ni)) < 5 else ""
        print(f"  {label}: ‖Δ‖={np.linalg.norm(delta):.3f}  "
              f"markers={len(up)}↑ {len(dn)}↓{n_warn}")

    # Group Δ for steering
    inj_available = [l for l in INJECTION_PAIRS if l in delta_vecs]
    if inj_available:
        attack_delta = np.stack([delta_vecs[l] for l in inj_available]).mean(0)
        np.save(os.path.join(save_dir, "Delta_attack.npy"), attack_delta)
        print(f"\n  Delta_attack (mean of {inj_available}): ‖Δ‖={np.linalg.norm(attack_delta):.3f}")

    # Cosine atlas
    if len(delta_vecs) >= 2:
        labels = sorted(delta_vecs.keys())
        n = len(labels)
        cos_mat = np.array([[cosine_sim(delta_vecs[a], delta_vecs[b])
                             for b in labels] for a in labels])
        atlas_df = pd.DataFrame(cos_mat, index=labels, columns=labels)
        atlas_df.to_csv(os.path.join(save_dir, "cosine_atlas.csv"))
        print(f"\n  Cosine atlas ({n}×{n}):")
        print(atlas_df.round(3).to_string())
        plot_cosine_atlas(atlas_df, os.path.join(save_dir, "delta_atlas.png"))

    # Top-features bar chart for the key pairs
    key_pairs = [l for l in ["B_vs_C", "B_vs_A", "C_vs_D"] if l in delta_vecs]
    if key_pairs:
        plot_top_features(delta_vecs, key_pairs[:3],
                          os.path.join(save_dir, "top_features.png"), topk=20)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Δ vectors + DE + orthogonality")
    ap.add_argument("out_dir")
    ap.add_argument("--scope",  default="out_mask",
                    help="SAE scope (default: out_mask). "
                         "tpl_mask for injection-specific (B+C only).")
    ap.add_argument("--layer",  type=int,   default=16)
    ap.add_argument("--frac",   type=float, default=0.10)
    ap.add_argument("--topk",   type=int,   default=20, help="Features per pair (quick mode)")
    ap.add_argument("--quick",  action="store_true",
                    help="Quick mean-gap ranking (no statistics, reads records directly)")
    ap.add_argument("--full",   action="store_true",
                    help="Full statistical DE (reads assembled matrices from p2)")
    ap.add_argument("--fdr-thresh",    type=float, default=0.05)
    ap.add_argument("--effect-thresh", type=float, default=0.1)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    if not args.quick and not args.full:
        args.quick = True   # default to quick if neither specified

    save_dir = args.save_dir or os.path.join(args.out_dir, "analysis", "delta")
    os.makedirs(save_dir, exist_ok=True)

    if args.quick:
        print(f"[p4_delta] QUICK mode: scope={args.scope!r} layer={args.layer} frac={args.frac}")
        run_quick(args.out_dir, args.scope, args.layer, args.frac, args.topk, save_dir)

    if args.full:
        print(f"\n[p4_delta] FULL mode: scope={args.scope!r} layer={args.layer} frac={args.frac}")
        run_full(args.out_dir, args.scope, args.layer, args.frac,
                 args.fdr_thresh, args.effect_thresh, save_dir)

    print(f"\n[p4_delta] Done → {save_dir}")
    print(f"  Next: python scripts/p5_modules.py {args.out_dir}")


if __name__ == "__main__":
    main()
