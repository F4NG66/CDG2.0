#!/usr/bin/env python3
"""p12_dir_stability.py — Cross-frac Direction Stability Analysis (Experiment Group D).

Answers: does the B-vs-A / C-vs-D / shared injection direction rotate significantly
across denoising steps (fracs)?  If directions at early steps (frac=0.05) and late
steps (frac=0.50+) are nearly orthogonal, a single-frac steering vector will be
ineffective for most of the denoising trajectory.

Method
------
For each frac in record_fractions × each layer in [11,16,26]:
  1. Load mean hidden states per group (A/B/C/D) from existing Phase-1 records
     (no model re-run needed).
  2. Compute v_BA(frac,layer) = mean(B) - mean(A)  (injection mechanism, harmful)
               v_CD(frac,layer) = mean(C) - mean(D)  (injection mechanism, neutral)
               v_shared via SVD of stacked [u_BA, u_CD]  (shared subspace)
  3. Record norms and cos(BA,CD) / var_explained at every (frac, layer).
  4. Build cross-frac cosine similarity matrices for v_shared per layer:
       cos(v_shared(f_i), v_shared(f_j))  for all pairs (f_i, f_j)
     and the consecutive-step cos (f[i] → f[i+1]).

Interpretation thresholds
-------------------------
  cos ≥ 0.7  : direction is stable — single frac=0.05 extraction is fine
  0.3–0.7    : moderate drift — consider 2-3 segment direction schedule
  < 0.3      : near-orthogonal — need frac-conditional direction switching

Outputs
-------
  analysis_output/p12_dir_stability.json   — full numeric data
  analysis_output/p12_dir_stability.png    — heatmap per layer (if matplotlib available)
  (also prints a formatted table to stdout)

Run
---
  python scripts/p12_dir_stability.py [--out-dir outputs] [--layers 11 16 26]
"""
from __future__ import annotations
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch

from cdg.probe import load_records, stack_group, group_letter as _gl

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out-dir",  default="outputs",
                help="Phase-1 records directory (manifest.jsonl)")
ap.add_argument("--layers",   nargs="+", type=int, default=[11, 16, 26])
ap.add_argument("--scope",    default="harm",
                help="Scope to extract hidden states from (default: harm)")
ap.add_argument("--no-plot",  action="store_true")
ap.add_argument("--out-json", default="analysis_output/p12_dir_stability.json")
args = ap.parse_args()

os.makedirs("analysis_output", exist_ok=True)

# ── load records ──────────────────────────────────────────────────────────────
print(f"Loading records from {args.out_dir}/ ...")
records = load_records(args.out_dir)
group_n = {}
for r in records:
    g = _gl(r)
    group_n[g] = group_n.get(g, 0) + 1
print(f"  {group_n}  (scope={args.scope})\n")

# ── discover available fracs ──────────────────────────────────────────────────
fracs_set: set = set()
for r in records:
    rec = r.get("_rec")
    if rec is None:
        continue
    store = rec.get("hidden", {}).get(args.scope, {})
    for k in store:
        try:
            fracs_set.add(float(k))
        except (TypeError, ValueError):
            pass
    break  # one record is enough to get the frac list

ALL_FRACS = sorted(fracs_set)
print(f"Available fracs in scope={args.scope}: {ALL_FRACS}\n")

if not ALL_FRACS:
    print(f"[error] No hidden states found for scope='{args.scope}'.")
    print("  Check that Phase-1 recording included this scope.")
    sys.exit(1)


# ── direction extraction helper ───────────────────────────────────────────────
def _make_direction(layer: int, frac: float) -> dict | None:
    """Compute v_BA, v_CD, shared direction for one (layer, frac) point."""
    mus = {}
    for grp in ("A", "B", "C", "D"):
        X, _ = stack_group(records, groups=(grp,), scope=args.scope,
                            frac=frac, layer=layer, space="hidden")
        if X is not None and X.shape[0] > 0:
            mus[grp] = X.float().mean(0)

    if not all(g in mus for g in ("A", "B", "C", "D")):
        return None

    v_BA = mus["B"] - mus["A"]
    v_CD = mus["C"] - mus["D"]
    u_BA = v_BA / (v_BA.norm() + 1e-8)
    u_CD = v_CD / (v_CD.norm() + 1e-8)

    M = torch.stack([u_BA, u_CD]).numpy()
    _, S, Vh = np.linalg.svd(M, full_matrices=False)
    shared = torch.from_numpy(Vh[0]).float()
    if torch.dot(shared, u_BA) < 0:
        shared = -shared

    return {
        "frac":          float(frac),
        "layer":         int(layer),
        "norm_BA":       float(v_BA.norm()),
        "norm_CD":       float(v_CD.norm()),
        "cos_BA_CD":     float(torch.dot(u_BA, u_CD)),
        "var_explained": float(S[0] ** 2 / (S**2).sum()),
        "sing_values":   S.tolist(),
        "v_BA":          v_BA,
        "v_CD":          v_CD,
        "shared":        shared,
    }


# ── compute directions at every (frac, layer) ─────────────────────────────────
print("Computing directions across fracs and layers...")
results: dict = {}   # layer -> frac -> direction_dict

for layer in args.layers:
    results[layer] = {}
    print(f"\n{'━'*58}")
    print(f"  Layer {layer}")
    print(f"{'━'*58}")
    print(f"  {'frac':>6}  {'|v_BA|':>8}  {'|v_CD|':>8}  "
          f"{'cos(BA,CD)':>11}  {'var_expl':>9}  {'S[0]':>7}  {'S[1]':>7}")
    print("  " + "─" * 56)

    for frac in ALL_FRACS:
        d = _make_direction(layer, frac)
        if d is None:
            print(f"  {frac:>6.2f}  [no data]")
            continue
        results[layer][frac] = d
        print(f"  {frac:>6.2f}  {d['norm_BA']:>8.4f}  {d['norm_CD']:>8.4f}  "
              f"{d['cos_BA_CD']:>11.4f}  {d['var_explained']:>9.4f}  "
              f"{d['sing_values'][0]:>7.4f}  {d['sing_values'][1]:>7.4f}")


# ── cross-frac cosine similarity matrices ─────────────────────────────────────
print("\n\n" + "═"*70)
print("  Cross-frac Cosine Similarity of v_shared(frac_i, frac_j)")
print("  (diagonal=1.0 by definition; check off-diagonals for drift)")
print("═"*70)

INTERP_THRESHOLDS = [(0.7, "stable ✓"), (0.3, "moderate drift ⚠"),
                     (0.0, "near-orthogonal ✗")]

def _interp(cos_val: float) -> str:
    for thresh, label in INTERP_THRESHOLDS:
        if cos_val >= thresh:
            return label
    return "near-orthogonal ✗"

cross_frac_data: dict = {}   # layer -> {(fi, fj): cos}

for layer in args.layers:
    layer_fracs = sorted(results[layer].keys())
    if len(layer_fracs) < 2:
        continue

    print(f"\n  Layer {layer}:")
    # Header
    hdr = "".join(f"  {f:>5.2f}" for f in layer_fracs)
    print(f"  {'frac':>6}{hdr}")
    print("  " + "─" * (8 + 7 * len(layer_fracs)))

    cos_matrix: dict = {}
    for fi in layer_fracs:
        row_str = f"  {fi:>6.2f}"
        for fj in layer_fracs:
            vi = results[layer][fi]["shared"]
            vj = results[layer][fj]["shared"]
            cos_val = float(torch.dot(vi / (vi.norm()+1e-8),
                                      vj / (vj.norm()+1e-8)))
            cos_matrix[(fi, fj)] = cos_val
            row_str += f"  {cos_val:>5.3f}"
        print(row_str)
    cross_frac_data[layer] = cos_matrix

    # Consecutive-step analysis
    print(f"\n  Consecutive-step drift (Layer {layer}):")
    print(f"  {'step':>14}  {'cos':>7}  {'interpretation'}")
    print("  " + "─" * 42)
    for i in range(len(layer_fracs) - 1):
        fi, fj = layer_fracs[i], layer_fracs[i+1]
        c = cos_matrix[(fi, fj)]
        print(f"  {fi:.2f} → {fj:.2f}   {c:>7.3f}  {_interp(c)}")

    # Key comparison: frac=0.05 vs all others
    f0 = layer_fracs[0]
    print(f"\n  frac={f0:.2f} vs later steps (Layer {layer}):")
    for fj in layer_fracs[1:]:
        c = cos_matrix[(f0, fj)]
        print(f"  {f0:.2f} → {fj:.2f}   {c:>7.3f}  {_interp(c)}")

# ── overall verdict ───────────────────────────────────────────────────────────
print("\n\n" + "═"*70)
print("  VERDICT (averaged across layers)")
print("═"*70)

# Find the frac closest to 0.5 for the key comparison
target_frac = 0.50

for layer in args.layers:
    layer_fracs = sorted(results[layer].keys())
    if not layer_fracs:
        continue
    f0 = layer_fracs[0]  # earliest frac (0.05)
    avail_later = [f for f in layer_fracs if f >= 0.40]
    if avail_later:
        f_mid = min(avail_later, key=lambda f: abs(f - target_frac))
        c_mid = cross_frac_data[layer].get((f0, f_mid), float("nan"))
        f_late = layer_fracs[-1]
        c_late = cross_frac_data[layer].get((f0, f_late), float("nan"))
        print(f"\n  Layer {layer}:")
        print(f"    cos(frac={f0:.2f}, frac={f_mid:.2f}) = {c_mid:.3f}  {_interp(c_mid)}")
        print(f"    cos(frac={f0:.2f}, frac={f_late:.2f}) = {c_late:.3f}  {_interp(c_late)}")
        if c_mid >= 0.7:
            print(f"    → Single frac=0.05 direction generalises well across steps.")
        elif c_mid >= 0.3:
            print(f"    → Moderate drift. Consider 2-segment direction schedule.")
        else:
            print(f"    → Near-orthogonal. Frac-conditional direction switching needed.")

# ── save results ──────────────────────────────────────────────────────────────
save_data = {
    "scope":   args.scope,
    "layers":  args.layers,
    "fracs":   ALL_FRACS,
    "per_layer": {
        str(layer): {
            str(frac): {
                k: v for k, v in d.items()
                if not isinstance(v, torch.Tensor)   # exclude tensors
            }
            for frac, d in fdata.items()
        }
        for layer, fdata in results.items()
    },
    "cross_frac_cosine": {
        str(layer): {
            f"{fi:.2f}_vs_{fj:.2f}": cos
            for (fi, fj), cos in cdata.items()
        }
        for layer, cdata in cross_frac_data.items()
    },
}

with open(args.out_json, "w") as fh:
    json.dump(save_data, fh, indent=2, default=float)
print(f"\n[saved] {args.out_json}")

# ── heatmap plot ──────────────────────────────────────────────────────────────
if not args.no_plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_layers = len(args.layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5),
                                  squeeze=False)

        for li, layer in enumerate(args.layers):
            ax = axes[0][li]
            layer_fracs = sorted(results[layer].keys())
            n = len(layer_fracs)
            mat = np.full((n, n), float("nan"))
            for i, fi in enumerate(layer_fracs):
                for j, fj in enumerate(layer_fracs):
                    v = cross_frac_data.get(layer, {}).get((fi, fj))
                    if v is not None:
                        mat[i, j] = v

            im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks(range(n))
            ax.set_xticklabels([f"{f:.2f}" for f in layer_fracs], fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels([f"{f:.2f}" for f in layer_fracs], fontsize=8)
            ax.set_xlabel("frac (j)", fontsize=9)
            ax.set_ylabel("frac (i)", fontsize=9)
            ax.set_title(f"cos(v_shared(i), v_shared(j))\nLayer {layer}", fontsize=9)

            for i in range(n):
                for j in range(n):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                                fontsize=7,
                                color="white" if mat[i, j] < 0.5 else "black")

        fig.colorbar(im, ax=axes[0], label="cosine similarity", fraction=0.02)
        fig.suptitle(
            f"Cross-frac Direction Stability  (scope={args.scope})\n"
            "v_shared = shared injection-mechanism direction at each denoising fraction",
            fontsize=9)
        plt.tight_layout()
        plot_path = "analysis_output/p12_dir_stability.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {plot_path}")

    except ImportError as e:
        print(f"[skip plot] {e}")
