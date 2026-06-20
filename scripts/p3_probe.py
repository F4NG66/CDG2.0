#!/usr/bin/env python3
"""p3_probe.py — Linear Probe Sweep (B vs C).

Grid-search over (scope × layer × frac × space) to find when and where the
injection signal is most linearly decodable from model activations.

Runs on Phase-1 records directly (no assembled matrices needed).
AUC → 1.0 means perfect separation of harmful-injection (B) from
neutral-injection (C) at that (scope, layer, frac, space).

Outputs:
  analysis_output/probe_sweep.json   — full results
  analysis_output/probe_heatmap.png  — AUC heatmap per scope

Run: python scripts/p3_probe.py [--out-dir outputs]
"""
import os, json, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--out-dir",  default="outputs")
ap.add_argument("--scopes",   nargs="+",
                default=["tpl_mask", "tpl_ctx", "harm", "out_unmask", "out_mask"])
ap.add_argument("--spaces",   nargs="+", default=["hidden", "sae"])
ap.add_argument("--no-plot",  action="store_true")
args = ap.parse_args()

from cdg.probe  import load_records, probe_sweep
from cdg.config import get_backend_config

cfg     = get_backend_config("llada_attack")
records = load_records(args.out_dir, cfg.name)
print(f"Loaded {len(records)} records.\n")

fracs  = [0.05, 0.10, 0.20, 0.35, 0.50, 1.00]
layers = list(cfg.record_layers)   # [11, 16, 26]

all_results = []
for space in args.spaces:
    print(f"── space={space} ──")
    rows = probe_sweep(
        records,
        scopes=args.scopes,
        fracs=fracs,
        layers=layers,
        space=space,
        pos_groups=("B",),
        neg_groups=("C",),
    )
    for r in rows:
        r["space"] = space
    all_results.extend(rows)

# Sort by AUC descending
all_results.sort(key=lambda r: r.get("auc") or 0, reverse=True)

print(f"\n{'space':7} {'scope':12} {'frac':6} {'layer':6} {'AUC':8} {'F1':8} {'n':5}")
print("─" * 70)
for r in all_results:
    auc = r.get("auc", float("nan"))
    f1  = r.get("macro_f1", float("nan"))
    note = r.get("note", "")
    print(f"{r.get('space','?'):7} {r['scope']:12} {r['frac']:6.2f} {r['layer']:6d} "
          f"{auc:8.3f} {f1:8.3f} {r.get('n',0):5d}  {note}")

best = all_results[0]
print(f"\n>> Best: space={best.get('space')}  scope={best['scope']}  "
      f"frac={best['frac']:.2f}  layer={best['layer']}  "
      f"AUC={best.get('auc', float('nan')):.3f}")

os.makedirs("analysis_output", exist_ok=True)
with open("analysis_output/probe_sweep.json", "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\n[saved] analysis_output/probe_sweep.json")

# ── Heatmap visualization ─────────────────────────────────────────────────────
if not args.no_plot:
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for space in args.spaces:
            space_rows = [r for r in all_results if r.get("space") == space]
            scopes_used = [s for s in args.scopes
                           if any(r["scope"] == s for r in space_rows
                                  if r.get("auc") is not None)]
            if not scopes_used:
                continue

            fig, axes = plt.subplots(1, len(layers),
                                     figsize=(5 * len(layers), 4 * len(scopes_used) // 3 + 2),
                                     squeeze=False)

            for li, layer in enumerate(layers):
                ax = axes[0][li]
                mat = np.full((len(scopes_used), len(fracs)), float("nan"))
                for si, scope in enumerate(scopes_used):
                    for fi, frac in enumerate(fracs):
                        hit = next((r for r in space_rows
                                    if r["scope"] == scope and r["layer"] == layer
                                    and abs(r["frac"] - frac) < 1e-4), None)
                        if hit and hit.get("auc") is not None:
                            mat[si, fi] = hit["auc"]

                im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                               aspect="auto", interpolation="nearest")
                ax.set_xticks(range(len(fracs)))
                ax.set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=8)
                ax.set_yticks(range(len(scopes_used)))
                ax.set_yticklabels(scopes_used, fontsize=8)
                ax.set_xlabel("Denoising fraction")
                ax.set_title(f"Layer {layer}", fontsize=10)

                for si in range(len(scopes_used)):
                    for fi in range(len(fracs)):
                        v = mat[si, fi]
                        if not np.isnan(v):
                            ax.text(fi, si, f"{v:.2f}", ha="center", va="center",
                                    fontsize=7,
                                    color="black" if v < 0.85 else "white")

            fig.colorbar(im, ax=axes[0], label="AUC (B vs C)", fraction=0.02)
            fig.suptitle(f"Linear Probe Sweep — space={space}  (B harmful_injected vs C neutral_injected)",
                         fontsize=10)
            plt.tight_layout()
            save_path = f"analysis_output/probe_heatmap_{space}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {save_path}")

    except ImportError as e:
        print(f"[skip plot] {e}")
