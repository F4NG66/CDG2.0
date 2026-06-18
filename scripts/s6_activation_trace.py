#!/usr/bin/env python3
"""
s6_activation_trace.py
----------------------
Track how the top injection features' activations evolve across denoising
steps (frac=0.05 → 1.00) for groups B and C separately.

This tells you: at which point in the diffusion process does the model
"commit" to the harmful path?  Early divergence (frac≤0.10) means the
injection hijacks the model's computation very early.

Output: analysis_output/activation_trace.json + activation_trace.png

Run: python scripts/s6_activation_trace.py [--layer 16] [--top 5]
"""
import os, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int,   default=16)
ap.add_argument("--scope", default="tpl_mask")
ap.add_argument("--top",   type=int,   default=5,
                help="How many top BvsC features to trace")
ap.add_argument("--no-plot", action="store_true")
args = ap.parse_args()

from cdg.probe  import load_records, stack_group
from cdg.config import get_backend_config
import numpy as np, json as _json

cfg     = get_backend_config("llada_attack")
records = load_records("outputs", cfg.name)

# ── load top feature IDs ──────────────────────────────────────────────────────
diff_path = "analysis_output/diff_features.json"
if not os.path.exists(diff_path):
    print("[error] Run s3_diff_features.py first.")
    raise SystemExit(1)

diff = _json.load(open(diff_path))
bc_row = next((r for r in diff["focus_rows"] if r["pair"] == "BvsC"), None)
if bc_row is None:
    print("[error] No BvsC row found in diff_features.json.")
    raise SystemExit(1)

focus_feats = [f["feature"] for f in bc_row["features"][:args.top]]
print(f"Tracing features: {focus_feats}  (layer={args.layer}, scope={args.scope})\n")

# ── discover available fracs from records ─────────────────────────────────────
all_fracs = []
for r in records:
    rec = r.get("_rec") or {}
    store = rec.get("sae", {}).get(args.scope, {})
    if store:
        all_fracs = sorted(float(k) for k in store.keys())
        break

if not all_fracs:
    print(f"[error] No SAE data for scope={args.scope}. "
          "Check Phase-1 was run with this scope.")
    raise SystemExit(1)

print(f"Fracs available: {all_fracs}\n")

# ── compute mean activation per group per frac ────────────────────────────────
trace = {fid: {"B": [], "C": []} for fid in focus_feats}

for frac in all_fracs:
    XB, _ = stack_group(records, groups=("B",), scope=args.scope,
                        frac=frac, layer=args.layer, space="sae")
    XC, _ = stack_group(records, groups=("C",), scope=args.scope,
                        frac=frac, layer=args.layer, space="sae")
    for fid in focus_feats:
        mb = float(XB[:, fid].mean()) if XB is not None else float("nan")
        mc = float(XC[:, fid].mean()) if XC is not None else float("nan")
        trace[fid]["B"].append(mb)
        trace[fid]["C"].append(mc)

# ── print table ──────────────────────────────────────────────────────────────
print(f"{'frac':>6}  " + "  ".join(f"feat{f:>6}(B/C)" for f in focus_feats))
print("-" * (8 + 20 * len(focus_feats)))
for i, frac in enumerate(all_fracs):
    row = f"{frac:6.2f}  "
    for fid in focus_feats:
        b = trace[fid]["B"][i]
        c = trace[fid]["C"][i]
        row += f"{b:7.4f}/{c:<7.4f}  "
    print(row)

# ── plot ──────────────────────────────────────────────────────────────────────
if not args.no_plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10.colors

        for fi, fid in enumerate(focus_feats):
            ax.plot(all_fracs, trace[fid]["B"], color=colors[fi],
                    linewidth=2.5, label=f"feat {fid} [B harmful]")
            ax.plot(all_fracs, trace[fid]["C"], color=colors[fi],
                    linewidth=1.5, linestyle="--", alpha=0.65,
                    label=f"feat {fid} [C neutral]")

        ax.axvline(bc_row["frac"], color="gray", linestyle=":", linewidth=1.5,
                   label=f"focus frac={bc_row['frac']}")
        ax.set_xlabel("Denoising fraction (0=start, 1=complete)")
        ax.set_ylabel("Mean SAE feature activation")
        ax.set_title(f"Injection feature activations over denoising — "
                     f"Layer {args.layer}, scope={args.scope}\n"
                     "Solid=harmful(B), Dashed=neutral(C)")
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("analysis_output/activation_trace.png", dpi=150)
        print("[saved] analysis_output/activation_trace.png")
    except ImportError:
        print("[skip] matplotlib not available")

# ── save ──────────────────────────────────────────────────────────────────────
out = {"layer": args.layer, "scope": args.scope,
       "fracs": all_fracs, "features": focus_feats,
       "trace": {str(fid): trace[fid] for fid in focus_feats}}
os.makedirs("analysis_output", exist_ok=True)
with open("analysis_output/activation_trace.json", "w") as f:
    _json.dump(out, f, indent=2)
print("[saved] analysis_output/activation_trace.json")
