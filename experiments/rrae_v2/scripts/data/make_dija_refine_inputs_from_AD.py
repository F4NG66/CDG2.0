#!/usr/bin/env python
import json
import pandas as pd
from pathlib import Path

ROOT = Path("/path/to/rrae_steering_work_v2")

A_PATH = ROOT / "data/group_A/A_harmful_clean_FINAL_400_health_100_nonhealth.jsonl"
D_PATH = ROOT / "data/group_D/D_benign_clean_deepseek_chat_FULL_500.jsonl"

OUT_A = ROOT / "data/dija_inputs/A_for_DIJA_refine.csv"
OUT_D = ROOT / "data/dija_inputs/D_for_DIJA_refine.csv"

AUDIT_JSON = ROOT / "data/audit/DIJA_refine_input_build_summary.json"

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

A = read_jsonl(A_PATH)
D = read_jsonl(D_PATH)

assert len(A) == 500, f"Expected 500 A rows, got {len(A)}"
assert len(D) == 500, f"Expected 500 D rows, got {len(D)}"

A_ids = [r["id"] for r in A]
D_pair_ids = [r["paired_A_id"] for r in D]

assert len(set(A_ids)) == 500, "A ids are not unique"
assert len(set(D_pair_ids)) == 500, "D paired_A_id values are not unique"
assert set(A_ids) == set(D_pair_ids), "A ids and D paired_A_id do not match"

A_rows = []
for r in A:
    A_rows.append({
        "BehaviorID": r["id"],
        "FunctionalCategory": "standard",
        "SemanticCategory": r.get("domain", ""),
        "Behavior": r["prompt"],
    })

D_rows = []
for r in D:
    D_rows.append({
        "BehaviorID": r["id"],
        "FunctionalCategory": "standard",
        "SemanticCategory": r.get("domain", ""),
        "Behavior": r["prompt"],
    })

A_df = pd.DataFrame(A_rows)
D_df = pd.DataFrame(D_rows)

A_df.to_csv(OUT_A, index=False)
D_df.to_csv(OUT_D, index=False)

summary = {
    "A_input": str(A_PATH),
    "D_input": str(D_PATH),
    "A_output_csv": str(OUT_A),
    "D_output_csv": str(OUT_D),
    "A_rows": len(A_df),
    "D_rows": len(D_df),
    "A_domain_counts": A_df["SemanticCategory"].value_counts().to_dict(),
    "D_domain_counts": D_df["SemanticCategory"].value_counts().to_dict(),
    "columns": list(A_df.columns),
}

AUDIT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
