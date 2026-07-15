#!/usr/bin/env python
from pathlib import Path
import json
import pandas as pd

ROOT = Path("/path/to/rrae_steering_work_v2")
BASE = ROOT / "external/MedSafetyBench"

files = []
for ext in ["*.csv", "*.jsonl", "*.json", "*.parquet"]:
    files.extend(BASE.rglob(ext))

print("found files:", len(files))

for p in files:
    print("\n===== FILE =====")
    print(p)
    try:
        if p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix == ".jsonl":
            rows = []
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 2000:
                        break
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            df = pd.DataFrame(rows)
        elif p.suffix == ".json":
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict):
                list_values = [v for v in obj.values() if isinstance(v, list)]
                df = pd.DataFrame(list_values[0]) if list_values else pd.DataFrame([obj])
            else:
                continue
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            continue

        print("shape:", df.shape)
        print("columns:", list(df.columns))

        for c in df.columns:
            if df[c].dtype == "object":
                nunique = df[c].nunique(dropna=True)
                avg_len = df[c].fillna("").astype(str).str.len().mean()
                print(f"  object_col={c!r} nunique={nunique} avg_len={avg_len:.1f}")
    except Exception as e:
        print("ERROR:", repr(e))
