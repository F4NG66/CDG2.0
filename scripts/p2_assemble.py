#!/usr/bin/env python3
"""p2_assemble.py — Batch Pseudobulk Assembly.

Scans all available scopes / layers / fracs from Phase-1 records and builds
pseudobulk feature matrices for each combination.  Downstream scripts read
from assembled/ instead of touching .pt files directly.

Usage
-----
python scripts/p2_assemble.py outputs/              # assemble everything
python scripts/p2_assemble.py outputs/ \\
    --scopes tpl_mask out_mask \\
    --layers 16 \\
    --fracs 0.10                                    # targeted subset

Outputs (under <out_dir>/assembled/)
-------------------------------------
  X_{scope}_L{layer}_f{frac}.npy      float32 [N, n_features]
  obs_{scope}_L{layer}_f{frac}.csv    observation metadata
  var_{scope}_L{layer}_f{frac}.csv    feature QC (dead, ubiquitous, frac_active)
  manifest.csv                         index of all assembled matrices + dims
  qc_overview.png                      feature activity overview plot
"""
from __future__ import annotations
import argparse, os, sys, warnings, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cdg.analysis.io import load_manifest, pseudobulk_matrix

ANALYSIS_DIR = "analysis"


def feature_qc(X: np.ndarray, ubiq_frac: float = 0.90,
               ubiq_var: float = 1e-4) -> pd.DataFrame:
    n, f = X.shape
    mean_act   = X.mean(0)
    frac_active = (X > 0).mean(0)
    var_act    = X.var(0)
    return pd.DataFrame({
        "feature_id":    np.arange(f, dtype=np.int32),
        "mean_act":      mean_act.astype(np.float32),
        "frac_active":   frac_active.astype(np.float32),
        "var_act":       var_act.astype(np.float32),
        "is_dead":       frac_active == 0.0,
        "is_ubiquitous": (frac_active > ubiq_frac) & (var_act < ubiq_var),
    })


def discover_axes(out_dir: str, model_name: str | None = None
                  ) -> tuple[list[str], list[int], list[float]]:
    """Probe one record to discover available scopes / layers / fracs."""
    import torch
    mf = load_manifest(out_dir)
    if model_name:
        mf = mf[mf["model_name"] == model_name]
    for _, row in mf.iterrows():
        path = row.get("path")
        if not path or not os.path.exists(path):
            continue
        try:
            rec = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        sae = rec.get("sae", {})
        if not sae:
            continue
        scopes = sorted(sae.keys())
        # pick first non-empty scope for layer/frac discovery
        for sc in scopes:
            fracs_d = sae[sc]
            if not fracs_d:
                continue
            fracs  = sorted(float(f) for f in fracs_d.keys())
            first_frac = next(iter(fracs_d.values()))
            layers = sorted(first_frac.keys())
            return scopes, layers, fracs
    return [], [], []


def plot_qc(var_df: pd.DataFrame, tag: str, save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        # 1. frac_active distribution (live features only)
        live = var_df[~var_df["is_dead"]]
        axes[0].hist(live["frac_active"], bins=20, color="#4e79a7", edgecolor="white")
        axes[0].set_xlabel("Fraction of samples active")
        axes[0].set_ylabel("# features")
        axes[0].set_title(f"Activity distribution\n({len(live)} live / {len(var_df)} total)")

        # 2. mean activation (live)
        axes[1].hist(np.log1p(live["mean_act"]), bins=30,
                     color="#f28e2b", edgecolor="white")
        axes[1].set_xlabel("log1p(mean activation)")
        axes[1].set_ylabel("# features")
        axes[1].set_title("Mean activation (log1p)")

        # 3. pie: dead / ubiquitous / live
        n_dead = var_df["is_dead"].sum()
        n_ubiq = var_df["is_ubiquitous"].sum()
        n_live = len(var_df) - n_dead
        axes[2].pie(
            [n_dead, n_ubiq, max(0, n_live - n_ubiq)],
            labels=["Dead", "Ubiquitous", "Live"],
            colors=["#e15759", "#f28e2b", "#4e79a7"],
            autopct="%1.0f%%", startangle=90,
        )
        axes[2].set_title("Feature QC breakdown")

        fig.suptitle(f"Feature QC — {tag}", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description="Batch pseudobulk assembly")
    ap.add_argument("out_dir")
    ap.add_argument("--scopes",  nargs="+", default=None,
                    help="Scopes to assemble (default: tpl_mask out_mask out_unmask)")
    ap.add_argument("--layers",  nargs="+", type=int, default=None,
                    help="Layers (default: all recorded)")
    ap.add_argument("--fracs",   nargs="+", type=float, default=None,
                    help="Fracs (default: main frac only = 0.10)")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--min-frac-active", type=float, default=0.0)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    save_dir = args.save_dir or os.path.join(args.out_dir, "assembled")
    os.makedirs(save_dir, exist_ok=True)

    # Discover axes if not specified
    all_scopes, all_layers, all_fracs = discover_axes(args.out_dir, args.model_name)
    scopes = args.scopes or [s for s in ["tpl_mask", "out_mask"] if s in all_scopes]
    layers = args.layers or [16] if 16 in all_layers else all_layers[:1]
    fracs  = args.fracs  or [0.10]
    print(f"[p2_assemble] Assembling: scopes={scopes} layers={layers} fracs={fracs}")

    manifest_rows = []
    for scope in scopes:
        for layer in layers:
            for frac in fracs:
                tag = f"{scope}_L{layer}_f{frac:.2f}".replace(".", "p")
                print(f"\n  ── {tag} ──")
                try:
                    X, obs = pseudobulk_matrix(
                        args.out_dir, scope=scope, layer=layer, frac=frac,
                        model_name=args.model_name,
                    )
                except (ValueError, Exception) as e:
                    warnings.warn(f"  Skipped {tag}: {e}")
                    continue

                N, F = X.shape
                var_df = feature_qc(X)
                n_dead = var_df["is_dead"].sum()
                n_live = F - n_dead
                print(f"  {N} generations × {F} features | live={n_live} dead={n_dead} "
                      f"({100*n_dead/F:.0f}%)")
                print(f"  Groups: {obs['group'].value_counts().to_dict()}")

                np.save(os.path.join(save_dir, f"X_{tag}.npy"), X)
                obs.to_csv(os.path.join(save_dir, f"obs_{tag}.csv"), index=False)
                var_df.to_csv(os.path.join(save_dir, f"var_{tag}.csv"), index=False)

                qc_path = os.path.join(save_dir, f"qc_{tag}.png")
                plot_qc(var_df, tag, qc_path)

                manifest_rows.append({
                    "tag": tag, "scope": scope, "layer": layer, "frac": frac,
                    "n_gen": N, "n_features": F, "n_live": int(n_live),
                    "n_dead": int(n_dead),
                    "groups": obs["group"].value_counts().to_dict(),
                    "x_path": os.path.join(save_dir, f"X_{tag}.npy"),
                    "obs_path": os.path.join(save_dir, f"obs_{tag}.csv"),
                    "var_path": os.path.join(save_dir, f"var_{tag}.csv"),
                })

    mf_path = os.path.join(save_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(mf_path, index=False)
    print(f"\n[p2_assemble] Done. manifest → {mf_path}")
    print(f"  Next: python scripts/p3_probe.py {args.out_dir}")
    print(f"        python scripts/p4_delta.py {args.out_dir}")


if __name__ == "__main__":
    main()
