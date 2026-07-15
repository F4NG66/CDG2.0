#!/usr/bin/env python
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/path/to/rrae_steering_work_v2")
BASE = ROOT / "external/MedSafetyBench"
OUT = ROOT / "data/source_extra/medsafetybench_extra_health_harmful.csv"
AUDIT = ROOT / "data/audit/medsafetybench_extra_health_harmful_summary.json"

TARGET = 80

PROMPT_COL_CANDIDATES = [
    "harmful_request", "unsafe_request", "request", "prompt", "question",
    "instruction", "user_prompt", "query", "input", "goal"
]

def norm(s):
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def read_file(p):
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if p.suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            list_values = [v for v in obj.values() if isinstance(v, list)]
            return pd.DataFrame(list_values[0]) if list_values else pd.DataFrame([obj])
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"unsupported: {p}")

def choose_prompt_col(df):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in PROMPT_COL_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand]

    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        return None

    # choose text-like column with reasonably long average text
    return max(object_cols, key=lambda c: df[c].fillna("").astype(str).str.len().mean())

rows = []
seen = set()
file_stats = []

files = []
for ext in ["*.csv", "*.jsonl", "*.json", "*.parquet"]:
    files.extend(BASE.rglob(ext))

for p in files:
    try:
        df = read_file(p)
    except Exception as e:
        file_stats.append({"file": str(p), "status": "read_failed", "error": repr(e)})
        continue

    if df.empty:
        file_stats.append({"file": str(p), "status": "empty"})
        continue

    col = choose_prompt_col(df)
    if not col:
        file_stats.append({"file": str(p), "status": "no_prompt_col", "columns": list(df.columns)})
        continue

    added = 0
    for i, r in df.iterrows():
        prompt = norm(r.get(col, ""))
        if len(prompt) < 15:
            continue

        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "prompt": prompt,
            "source_dataset": "MedSafetyBench",
            "source_file": str(p),
            "source_id": f"{p.name}_row_{i}",
            "domain": "health_related",
            "label": "harmful",
            "variant": "clean",
            "bucket": "A",
            "needs_manual_audit": True,
        })
        added += 1

        if len(rows) >= TARGET:
            break

    file_stats.append({
        "file": str(p),
        "status": "ok",
        "shape": list(df.shape),
        "prompt_col": col,
        "added": added,
        "columns": list(df.columns),
    })

    if len(rows) >= TARGET:
        break

out_df = pd.DataFrame(rows[:TARGET])
OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT, index=False)

summary = {
    "target": TARGET,
    "selected": int(len(out_df)),
    "output_csv": str(OUT),
    "file_stats": file_stats,
}

AUDIT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps({
    "target": TARGET,
    "selected": int(len(out_df)),
    "output_csv": str(OUT),
    "audit_json": str(AUDIT),
}, indent=2, ensure_ascii=False))

if len(out_df) < TARGET:
    raise SystemExit("Not enough extra health harmful examples from MedSafetyBench.")
