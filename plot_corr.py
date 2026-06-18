#!/usr/bin/env python3
"""Plot raw correlation matrix heatmap (no clustering, no dendrogram)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

corr_path = "outputs/analysis/modules/corr_matrix.npy"
out_path  = "outputs/analysis/modules/corr_raw_heatmap.png"

C = np.load(corr_path)
n = C.shape[0]
print(f"Matrix: {n}×{n}  range=[{C.min():.4f}, {C.max():.4f}]")

# Subsample for readability — keep every stride-th feature
# 8732 × 8732 at 1px per cell = 8732px, too big; downsample to ~1000
stride = max(1, n // 1000)
Cs = C[::stride, ::stride]
ns = Cs.shape[0]
print(f"Subsampled: {ns}×{ns} (stride={stride})")

# Colour scale: symmetric around 0, clipped at ±0.5 (most values are small)
vmax = 0.5

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(Cs, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               aspect="auto", interpolation="nearest")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label("Pearson r  (log1p-transformed activations)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_xlabel(f"SAE feature index  (subsampled 1-in-{stride}, {ns} shown)", fontsize=9)
ax.set_ylabel(f"SAE feature index  (subsampled 1-in-{stride}, {ns} shown)", fontsize=9)
ax.set_title(
    f"SAE Feature Co-activation Correlation Matrix\n"
    f"tpl_mask scope · layer 16 · frac=0.10 · {n} active features · N=200 generations",
    fontsize=10,
)
ax.tick_params(labelsize=7)

# Light diagonal reference
ax.plot([0, ns-1], [0, ns-1], color="black", lw=0.4, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
