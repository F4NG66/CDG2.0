#!/usr/bin/env python3
"""clockv2/make_summary.py — compact, VERSION-CONTROLLABLE summary of every probe reading.

WHY: results/ is 414MB of per-step JSONL living only on /scratch, which is purgeable, and
/home is at 99% (516MB free) so it cannot hold a copy. Every load-bearing table in REPORT.md
step-averages over the 128 denoising steps, so the per-step detail is not needed to
regenerate them. This collapses each (file, id, cond, seed, layer) to its step-averaged
projections -> ~1MB, which fits in git.

WHAT IT PRESERVES: §1 headline, §2/§2a split, §3 localization + band tables, §3a seed
robustness, §7 two-arm agreement. i.e. every number the paper rests on.

WHAT IT DOES NOT PRESERVE (stated so nobody assumes otherwise):
  * per-step trajectories — §4's tau table samples steps 0/32/64/96/127 individually, and
    §3's physical-canvas table needs per-step mask_ratio. Both concern tau, which is
    NULL/circular and not in the paper. Regenerate from raw if ever needed.
  * anything requiring the raw hidden states (they were never written to disk at all).

    python clockv2/make_summary.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from common import RESULTS_DIR

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary")
PROJ = ["vshared_pmean", "vshared_last", "vshared_rmean",
        "vrefusal_pmean", "vrefusal_last", "vrefusal_rmean",
        "vnull_pmean", "vnull_last", "vnull_rmean"]
META = ["n_inject", "prompt_len"]


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "probe_readings*.jsonl")))
    acc = defaultdict(lambda: defaultdict(list))
    meta = {}
    for f in files:
        base = os.path.basename(f)
        n = 0
        for line in open(f):
            r = json.loads(line)
            key = (base, r["id"], r["cond"], r.get("seed", 0),
                   r.get("temperature", 0.0), r["layer"])
            for p in PROJ:
                if p in r:
                    acc[key][p].append(r[p])
            meta[key] = tuple(r.get(m) for m in META)
            n += 1
        print(f"  read {base}: {n} rows")

    path = os.path.join(OUT, "probe_summary.csv")
    cols = ["file", "id", "cond", "seed", "temperature", "layer"] + META + \
           [f"{p}_mean" for p in PROJ] + ["n_steps"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for key in sorted(acc):
            d = acc[key]
            row = list(key) + list(meta[key])
            nst = 0
            for p in PROJ:
                v = d.get(p)
                if v:
                    row.append(round(float(np.mean(v)), 6))
                    nst = max(nst, len(v))
                else:
                    row.append("")
            row.append(nst)
            w.writerow(row)
    sz = os.path.getsize(path)
    print(f"\nwrote {path}  ({sz/1024:.0f} KB, {len(acc)} rows)")
    print("  each row = one (file, id, cond, seed, temperature, layer), projections averaged")
    print("  over all 128 denoising steps. Regenerates every table in REPORT.md except the")
    print("  per-step tau/canvas tables (§3 Table 1, §4), which are NULL/circular and not in")
    print("  the paper.")


if __name__ == "__main__":
    main()
