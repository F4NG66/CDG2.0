#!/usr/bin/env python

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("/path/to/rrae_steering_work_v2")

A_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HARMFUL_CLEAN_FINAL_500.jsonl"
)

D_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_D_benign_clean/"
      "D_TEST_BENIGN_CLEAN_FINAL_500.jsonl"
)

OUT_DIR = ROOT / "data/TEST_HELDOUT_V1/dija_inputs"

A_CSV = OUT_DIR / "A_TEST_for_DIJA_refine.csv"
D_CSV = OUT_DIR / "D_TEST_for_DIJA_refine.csv"

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "DIJA_TEST_refine_input_summary.json"
)


def read_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


A = read_jsonl(A_PATH)
D = read_jsonl(D_PATH)

if len(A) != 500:
    raise RuntimeError(f"Expected 500 A_TEST rows, found {len(A)}")

if len(D) != 500:
    raise RuntimeError(f"Expected 500 D_TEST rows, found {len(D)}")

A_pair_ids = [row["pair_id"] for row in A]
D_pair_ids = [row["pair_id"] for row in D]

if len(set(A_pair_ids)) != 500:
    raise RuntimeError("A_TEST pair IDs are not unique")

if len(set(D_pair_ids)) != 500:
    raise RuntimeError("D_TEST pair IDs are not unique")

if set(A_pair_ids) != set(D_pair_ids):
    missing_in_D = sorted(set(A_pair_ids) - set(D_pair_ids))
    missing_in_A = sorted(set(D_pair_ids) - set(A_pair_ids))

    raise RuntimeError(
        "A_TEST and D_TEST pair IDs do not match\n"
        f"missing in D: {missing_in_D[:10]}\n"
        f"missing in A: {missing_in_A[:10]}"
    )

A_by_pair = {
    row["pair_id"]: row
    for row in A
}

D_by_pair = {
    row["pair_id"]: row
    for row in D
}

ordered_pair_ids = [
    f"TEST_PAIR_{index:04d}"
    for index in range(500)
]

if set(ordered_pair_ids) != set(A_pair_ids):
    raise RuntimeError(
        "Expected pair IDs TEST_PAIR_0000 to TEST_PAIR_0499"
    )

A_rows = []
D_rows = []

for pair_id in ordered_pair_ids:
    a = A_by_pair[pair_id]
    d = D_by_pair[pair_id]

    A_rows.append({
        "BehaviorID": pair_id,
        "FunctionalCategory": "standard",
        "SemanticCategory": a["domain"],
        "Behavior": a["prompt"],
    })

    D_rows.append({
        "BehaviorID": pair_id,
        "FunctionalCategory": "standard",
        "SemanticCategory": d["domain"],
        "Behavior": d["prompt"],
    })

A_df = pd.DataFrame(A_rows)
D_df = pd.DataFrame(D_rows)

expected_columns = [
    "BehaviorID",
    "FunctionalCategory",
    "SemanticCategory",
    "Behavior",
]

if list(A_df.columns) != expected_columns:
    raise RuntimeError(f"Unexpected A columns: {list(A_df.columns)}")

if list(D_df.columns) != expected_columns:
    raise RuntimeError(f"Unexpected D columns: {list(D_df.columns)}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

A_df.to_csv(A_CSV, index=False)
D_df.to_csv(D_CSV, index=False)

summary = {
    "A_input": str(A_PATH),
    "D_input": str(D_PATH),
    "A_output": str(A_CSV),
    "D_output": str(D_CSV),
    "A_rows": len(A_df),
    "D_rows": len(D_df),
    "A_unique_behavior_ids": int(A_df["BehaviorID"].nunique()),
    "D_unique_behavior_ids": int(D_df["BehaviorID"].nunique()),
    "A_domain_counts": dict(Counter(A_df["SemanticCategory"])),
    "D_domain_counts": dict(Counter(D_df["SemanticCategory"])),
    "columns": expected_columns,
    "first_behavior_id": ordered_pair_ids[0],
    "last_behavior_id": ordered_pair_ids[-1],
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== DIJA TEST inputs prepared =====")
print("A rows:", len(A_df))
print("D rows:", len(D_df))
print("A domains:", Counter(A_df["SemanticCategory"]))
print("D domains:", Counter(D_df["SemanticCategory"]))
print(
    "BehaviorID range:",
    ordered_pair_ids[0],
    "to",
    ordered_pair_ids[-1],
)
print("A CSV:", A_CSV)
print("D CSV:", D_CSV)
print("summary:", SUMMARY_PATH)
