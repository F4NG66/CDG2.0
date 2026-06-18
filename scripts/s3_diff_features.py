#!/usr/bin/env python3
"""
s3_diff_features.py
-------------------
Find the SAE features with the largest mean-activation gap between group pairs.

For each (layer x frac x pair), rank all 16384 SAE features by |gap|.
A feature with large B-C gap means: it fires strongly when the model is being
coerced into harmful completion (B) but NOT when it processes the same
injection scaffold with benign content (C).  These are the "injection features".

Output: diff_features.json + printed table

Run: python scripts/s3_diff_features.py [--layer 16] [--frac 0.10] [--topk 20]
"""
import os, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int,   default=16)
ap.add_argument("--frac",  type=float, default=0.10)
ap.add_argument("--scope", default="tpl_mask")
ap.add_argument("--topk",  type=int,   default=20)
args = ap.parse_args()

from cdg.probe import load_records, top_diff_features
from cdg.config import get_backend_config

cfg     = get_backend_config("llada_attack")
records = load_records("outputs", cfg.name)
print(f"Loaded {len(records)} records.\n")

pairs = [("B","C"), ("A","B"), ("C","D")]
rows  = top_diff_features(
    records,
    scope=args.scope,
    pairs=pairs,
    k=args.topk,
    stat="mean_gap",
)

if not rows:
    print(f"No data found for scope={args.scope}. "
          "Make sure Phase-1 recorded this scope.")
    raise SystemExit(1)

# ── print results for the focus (layer, frac) ───────────────────────────────
focus_rows = [r for r in rows
              if r["layer"] == args.layer
              and abs(r["frac"] - args.frac) < 1e-4]

if not focus_rows:
    available_layers = sorted({r["layer"] for r in rows})
    available_fracs  = sorted({r["frac"]  for r in rows})
    print(f"No rows for layer={args.layer} frac={args.frac}.")
    print(f"Available layers: {available_layers}")
    print(f"Available fracs:  {available_fracs}")
    print("Re-run with --layer / --frac from the lists above.")
    raise SystemExit(1)

for row in focus_rows:
    pair = row["pair"]
    print(f"\n{'='*60}")
    print(f"  Pair={pair}  layer={row['layer']}  frac={row['frac']:.2f}  "
          f"scope={args.scope}")
    print(f"  n_pos={row['n_pos']}  n_neg={row['n_neg']}")
    print(f"{'='*60}")
    print(f"  {'feature':>8}  {'gap':>10}  {'pos_mean':>10}  {'neg_mean':>10}")
    print(f"  {'-'*44}")
    for f in row["features"]:
        marker = " <-- injection feature" if f["score"] > 0 and pair == "BvsC" else ""
        print(f"  {f['feature']:8d}  {f['score']:+10.4f}  "
              f"{f['pos_mean']:10.4f}  {f['neg_mean']:10.4f}{marker}")

# ── union of features appearing in multiple pairs ───────────────────────────
from collections import defaultdict
feat_pairs = defaultdict(list)
for row in focus_rows:
    for f in row["features"]:
        feat_pairs[f["feature"]].append(row["pair"])

shared = [(fid, ps) for fid, ps in feat_pairs.items() if len(ps) > 1]
shared.sort(key=lambda x: -len(x[1]))

if shared:
    print(f"\n-- Features shared across multiple pairs (layer={args.layer} frac={args.frac}) --")
    for fid, ps in shared[:10]:
        print(f"  feature {fid:6d}  appears in: {ps}")

# ── save ─────────────────────────────────────────────────────────────────────
os.makedirs("analysis_output", exist_ok=True)
out = {
    "layer": args.layer, "frac": args.frac, "scope": args.scope,
    "rows": rows,
    "focus_rows": focus_rows,
    "union_feature_ids": sorted(feat_pairs.keys()),
}
with open("analysis_output/diff_features.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n[saved] analysis_output/diff_features.json")

bc = next((r for r in focus_rows if r["pair"] == "BvsC"), None)
if bc:
    top_ids = [f["feature"] for f in bc["features"][:10]]
    print(f"\n>> Top-10 injection features (B vs C, layer={args.layer}, frac={args.frac}):")
    print(f"   {top_ids}")
    print("   Pass these to s4_vocab_labels.py via --features")
