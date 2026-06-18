#!/usr/bin/env python3
"""
s2_probe_sweep.py
-----------------
Grid-search over (scope x layer x frac) using a linear probe (logistic regression).
Tells you WHEN and WHERE in the model the injection signal is most linearly decodable.

AUC close to 1.0 means: at that position/layer/step, the hidden state alone can
distinguish harmful-injection (B) from neutral-injection (C) nearly perfectly.
This is the right place to aim steering.

Output: probe_sweep.json + printed table

Run: python scripts/s2_probe_sweep.py
"""
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)

from cdg.probe import load_records, probe_sweep
from cdg.config import get_backend_config

cfg     = get_backend_config("llada_attack")
records = load_records("outputs", cfg.name)
print(f"Loaded {len(records)} records.\n")

sweep = probe_sweep(
    records,
    scopes=["tpl_mask", "tpl_ctx", "harm", "out_unmask"],
    fracs=[0.05, 0.10, 0.20, 0.35, 0.50, 1.00],
    layers=list(cfg.record_layers),   # [11, 16, 26]
    space="hidden",
    pos_groups=("B",),
    neg_groups=("C",),
)

# sort by AUC descending
sweep.sort(key=lambda r: r.get("auc") or 0, reverse=True)

# print top results
print(f"{'scope':12} {'frac':6} {'layer':6} {'AUC':8} {'F1':8} {'n':4}  note")
print("-" * 65)
for r in sweep:
    auc  = r.get("auc",      float("nan"))
    f1   = r.get("macro_f1", float("nan"))
    note = r.get("note", "")
    print(f"{r['scope']:12} {r['frac']:6.2f} {r['layer']:6d} "
          f"{auc:8.3f} {f1:8.3f} {r.get('n',0):4d}  {note}")

# save
os.makedirs("analysis_output", exist_ok=True)
with open("analysis_output/probe_sweep.json", "w") as f:
    json.dump(sweep, f, indent=2, default=float)

print(f"\n[saved] analysis_output/probe_sweep.json")
print("\n>> Best (layer, frac, scope) for steering:")
best = sweep[0]
print(f"   scope={best['scope']}  frac={best['frac']:.2f}  layer={best['layer']}  "
      f"AUC={best.get('auc', float('nan')):.3f}")
