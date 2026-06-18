#!/usr/bin/env python3
"""
s5_atlas.py
-----------
Build the Feature Atlas:
  1. Decoder cosine-similarity matrix  — geometric independence of feature directions
  2. Co-activation Pearson matrix      — functional co-firing on actual data
  3. Hierarchy / clustering            — which features cluster together

Reads feature IDs from analysis_output/diff_features.json (s3 output).

Output: analysis_output/atlas.json + feature_atlas.png

Run: python scripts/s5_atlas.py [--layer 16] [--frac 0.10]
"""
import os, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int,   default=16)
ap.add_argument("--frac",  type=float, default=0.10)
ap.add_argument("--scope", default="tpl_mask")
ap.add_argument("--no-plot", action="store_true")
args = ap.parse_args()

from cdg.probe   import load_records
from cdg.sae     import load_sae, sae_ckpt_path
from cdg.config  import get_backend_config
from cdg.interpret import feature_atlas_data, coactivation_matrix, save_atlas

cfg     = get_backend_config("llada_attack")
records = load_records("outputs", cfg.name)
print(f"Loaded {len(records)} records.")

# ── load feature IDs ──────────────────────────────────────────────────────────
diff_path = "analysis_output/diff_features.json"
if not os.path.exists(diff_path):
    print(f"[error] Run s3_diff_features.py first.")
    raise SystemExit(1)

diff        = json.load(open(diff_path))
union_feats = diff["union_feature_ids"]
print(f"Union feature set: {len(union_feats)} features across all pairs\n")

# ── load SAE ─────────────────────────────────────────────────────────────────
ckpt, cfgp = sae_ckpt_path(os.path.join("saes", "llada_mask"),
                            layer=args.layer, trainer=1)
sae = load_sae(ckpt, config_path=cfgp)
print(f"SAE: d={sae.d_model}  n={sae.n_features}  k={sae.k}")

# ── feature atlas (geometric) ─────────────────────────────────────────────────
atlas = feature_atlas_data(sae, union_feats)
sim   = atlas["cosine_sim"]
order = atlas["order"] or list(range(len(union_feats)))
feat_ord = [union_feats[i] for i in order]

import numpy as np
sim_ord = sim[np.ix_(order, order)]

print(f"\n-- Decoder direction cosine similarity (off-diagonal stats) --")
mask_off = ~np.eye(len(union_feats), dtype=bool)
off = sim[mask_off]
print(f"  mean |cos| = {np.abs(off).mean():.4f}   "
      f"max |cos| = {np.abs(off).max():.4f}   "
      f"frac > 0.3 = {(np.abs(off) > 0.3).mean():.2%}")
print("  (ideal: mean near 0 = features are near-orthogonal = monosemantic)")

if atlas["clusters"]:
    from collections import defaultdict as DD
    cl = DD(list)
    for fid, cid in zip(union_feats, atlas["clusters"]):
        cl[cid].append(fid)
    print(f"\n-- K-means clusters ({len(cl)} clusters) --")
    for cid in sorted(cl):
        print(f"  cluster {cid}: {cl[cid]}")

# ── co-activation matrix (functional) ─────────────────────────────────────────
try:
    C_mat = coactivation_matrix(
        records, feature_ids=union_feats,
        scope=args.scope, frac=args.frac, layer=args.layer,
        groups=("B", "C"),
    )
    C_ord = C_mat[np.ix_(order, order)]
    off_c = C_mat[mask_off]
    print(f"\n-- Co-activation Pearson (off-diagonal stats, B+C combined) --")
    print(f"  mean |r| = {np.abs(off_c).mean():.4f}   "
          f"max |r| = {np.abs(off_c).max():.4f}   "
          f"frac > 0.5 = {(np.abs(off_c) > 0.5).mean():.2%}")
    has_coact = True
except Exception as e:
    print(f"\n[skip] co-activation matrix: {e}")
    C_ord = None
    has_coact = False

# ── plot ──────────────────────────────────────────────────────────────────────
if not args.no_plot:
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (no display needed)
        import matplotlib.pyplot as plt

        n_plots = 2 if has_coact else 1
        fig, axes = plt.subplots(1, n_plots,
                                 figsize=(7 * n_plots, 6),
                                 squeeze=False)

        ax = axes[0][0]
        im = ax.imshow(sim_ord, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(feat_ord)))
        ax.set_xticklabels(feat_ord, rotation=90, fontsize=7)
        ax.set_yticks(range(len(feat_ord)))
        ax.set_yticklabels(feat_ord, fontsize=7)
        ax.set_title("Decoder Cosine Sim (geometry)")
        plt.colorbar(im, ax=ax)

        if has_coact:
            ax2 = axes[0][1]
            im2 = ax2.imshow(C_ord, vmin=-1, vmax=1, cmap="RdBu_r")
            ax2.set_xticks(range(len(feat_ord)))
            ax2.set_xticklabels(feat_ord, rotation=90, fontsize=7)
            ax2.set_yticks(range(len(feat_ord)))
            ax2.set_yticklabels(feat_ord, fontsize=7)
            ax2.set_title("Co-activation Pearson (function)")
            plt.colorbar(im2, ax=ax2)

        plt.suptitle(f"Feature Atlas — layer={args.layer} frac={args.frac} scope={args.scope}")
        plt.tight_layout()
        os.makedirs("analysis_output", exist_ok=True)
        plt.savefig("analysis_output/feature_atlas.png", dpi=150, bbox_inches="tight")
        print(f"\n[saved] analysis_output/feature_atlas.png")
    except ImportError:
        print("[skip] matplotlib not available — run `pip install matplotlib` for plots")

# ── save atlas JSON ───────────────────────────────────────────────────────────
os.makedirs("analysis_output", exist_ok=True)
atlas_save = {k: v for k, v in atlas.items() if k != "cosine_sim"}
atlas_save["cosine_sim"] = sim.tolist()
if has_coact:
    atlas_save["coactivation"] = C_mat.tolist()
with open("analysis_output/atlas.json", "w") as f:
    json.dump(atlas_save, f, indent=2, default=float)
print(f"[saved] analysis_output/atlas.json")
