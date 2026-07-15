#!/usr/bin/env python

import json
import re
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

B_RAW = (
    ROOT
    / "data/TEST_HELDOUT_V1/dija_raw/"
      "B_TEST_DIJA_Qwen_raw.json"
)

C_RAW = (
    ROOT
    / "data/TEST_HELDOUT_V1/dija_raw/"
      "C_TEST_DIJA_Qwen_raw.json"
)

B_JSONL = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_B_harmful_injected/"
      "B_TEST_HARMFUL_DIJA_FINAL_500.jsonl"
)

C_JSONL = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_C_benign_injected/"
      "C_TEST_BENIGN_DIJA_FINAL_500.jsonl"
)

ABCD_JSONL = (
    ROOT
    / "data/TEST_HELDOUT_V1/abcd/"
      "ABCD_TEST_HELDOUT_V1_N2000.jsonl"
)

ABCD_CSV = (
    ROOT
    / "data/TEST_HELDOUT_V1/abcd/"
      "ABCD_TEST_HELDOUT_V1_N2000.csv"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "ABCD_TEST_HELDOUT_V1_N2000_summary.json"
)


def read_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mask_stats(text):
    nums = [
        int(x)
        for x in re.findall(r"<mask:(\d+)>", str(text or ""))
    ]

    return {
        "mask_segment_count": len(nums),
        "mask_token_total_declared": sum(nums),
        "mask_lengths": nums,
    }


A = read_jsonl(A_PATH)
D = read_jsonl(D_PATH)

B_raw = json.loads(B_RAW.read_text(encoding="utf-8"))
C_raw = json.loads(C_RAW.read_text(encoding="utf-8"))

if len(A) != 500:
    raise RuntimeError(f"A expected 500, found {len(A)}")

if len(D) != 500:
    raise RuntimeError(f"D expected 500, found {len(D)}")

if len(B_raw) != 500:
    raise RuntimeError(f"B raw expected 500, found {len(B_raw)}")

if len(C_raw) != 500:
    raise RuntimeError(f"C raw expected 500, found {len(C_raw)}")

A_by_pair = {row["pair_id"]: row for row in A}
D_by_pair = {row["pair_id"]: row for row in D}

if len(A_by_pair) != 500 or len(D_by_pair) != 500:
    raise RuntimeError("A or D pair IDs are not unique")

if set(A_by_pair) != set(D_by_pair):
    raise RuntimeError("A and D pair IDs do not match")

B_raw_by_pair = {
    str(row["BehaviorID"]): row
    for row in B_raw
}

C_raw_by_pair = {
    str(row["BehaviorID"]): row
    for row in C_raw
}

if set(B_raw_by_pair) != set(A_by_pair):
    raise RuntimeError("B raw pair IDs do not match A")

if set(C_raw_by_pair) != set(D_by_pair):
    raise RuntimeError("C raw pair IDs do not match D")

ordered_pairs = [
    f"TEST_PAIR_{i:04d}"
    for i in range(500)
]

B = []
C = []

for index, pair_id in enumerate(ordered_pairs):
    a = A_by_pair[pair_id]
    d = D_by_pair[pair_id]

    b_raw = B_raw_by_pair[pair_id]
    c_raw = C_raw_by_pair[pair_id]

    b_prompt = str(
        b_raw.get("Refined_behavior", "")
    ).strip()

    c_prompt = str(
        c_raw.get("Refined_behavior", "")
    ).strip()

    b_stats = mask_stats(b_prompt)
    c_stats = mask_stats(c_prompt)

    if b_stats["mask_segment_count"] == 0:
        raise RuntimeError(f"B missing mask: {pair_id}")

    if c_stats["mask_segment_count"] == 0:
        raise RuntimeError(f"C missing mask: {pair_id}")

    B.append({
        "id": f"B_TEST_{index:04d}",
        "pair_id": pair_id,
        "paired_A_id": a["id"],
        "paired_D_id": d["id"],
        "bucket": "B",
        "label": "harmful",
        "variant": "injected",
        "injection_type": "DIJA_Qwen_refined_maskN",
        "domain": a["domain"],
        "prompt": b_prompt,
        "clean_prompt": a["prompt"],
        "source_dataset": a.get("source_dataset"),
        "dija_behavior_id": pair_id,
        "mask_segment_count": b_stats["mask_segment_count"],
        "mask_token_total_declared": (
            b_stats["mask_token_total_declared"]
        ),
        "mask_lengths": b_stats["mask_lengths"],
        "validation_status": "dija_generated_and_mask_verified",
    })

    C.append({
        "id": f"C_TEST_{index:04d}",
        "pair_id": pair_id,
        "paired_A_id": a["id"],
        "paired_D_id": d["id"],
        "bucket": "C",
        "label": "benign",
        "variant": "injected",
        "injection_type": "DIJA_Qwen_refined_maskN",
        "domain": d["domain"],
        "prompt": c_prompt,
        "clean_prompt": d["prompt"],
        "source_dataset": d.get("source_dataset"),
        "dija_behavior_id": pair_id,
        "mask_segment_count": c_stats["mask_segment_count"],
        "mask_token_total_declared": (
            c_stats["mask_token_total_declared"]
        ),
        "mask_lengths": c_stats["mask_lengths"],
        "validation_status": "dija_generated_and_mask_verified",
    })

write_jsonl(B, B_JSONL)
write_jsonl(C, C_JSONL)

ABCD = []

for row in A:
    ABCD.append({
        **row,
        "variant": "clean",
        "injection_type": "none",
        "clean_prompt": row["prompt"],
    })

ABCD.extend(B)
ABCD.extend(C)

for row in D:
    ABCD.append({
        **row,
        "variant": "clean",
        "injection_type": "none",
        "clean_prompt": row["prompt"],
    })

if len(ABCD) != 2000:
    raise RuntimeError(
        f"ABCD expected 2000 rows, found {len(ABCD)}"
    )

df = pd.DataFrame(ABCD)

bucket_counts = Counter(df["bucket"])
label_counts = Counter(df["label"])
domain_counts = Counter(
    zip(df["bucket"], df["domain"])
)

if bucket_counts != Counter({
    "A": 500,
    "B": 500,
    "C": 500,
    "D": 500,
}):
    raise RuntimeError(
        f"Unexpected bucket counts: {bucket_counts}"
    )

if label_counts != Counter({
    "harmful": 1000,
    "benign": 1000,
}):
    raise RuntimeError(
        f"Unexpected label counts: {label_counts}"
    )

if df["id"].nunique() != 2000:
    raise RuntimeError("ABCD IDs are not unique")

pair_counts = Counter(df["pair_id"])

if set(pair_counts.values()) != {4}:
    raise RuntimeError(
        "Each pair_id must appear exactly four times"
    )

write_jsonl(ABCD, ABCD_JSONL)

ABCD_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(ABCD_CSV, index=False)

summary = {
    "total_rows": len(ABCD),
    "bucket_counts": dict(bucket_counts),
    "label_counts": dict(label_counts),
    "unique_ids": int(df["id"].nunique()),
    "unique_pair_ids": int(df["pair_id"].nunique()),
    "rows_per_pair_counts": {
        str(k): int(v)
        for k, v in Counter(pair_counts.values()).items()
    },
    "domain_by_bucket": {
        f"{bucket}|{domain}": int(count)
        for (bucket, domain), count in domain_counts.items()
    },
    "B_mask_segment_counts": {
        str(k): int(v)
        for k, v in Counter(
            row["mask_segment_count"] for row in B
        ).items()
    },
    "C_mask_segment_counts": {
        str(k): int(v)
        for k, v in Counter(
            row["mask_segment_count"] for row in C
        ).items()
    },
    "B_no_mask_rows": sum(
        row["mask_segment_count"] == 0
        for row in B
    ),
    "C_no_mask_rows": sum(
        row["mask_segment_count"] == 0
        for row in C
    ),
    "outputs": {
        "B": str(B_JSONL),
        "C": str(C_JSONL),
        "ABCD_jsonl": str(ABCD_JSONL),
        "ABCD_csv": str(ABCD_CSV),
    },
}

SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== ABCD TEST finalized =====")
print("total rows:", len(ABCD))
print("buckets:", bucket_counts)
print("labels:", label_counts)
print("unique IDs:", df["id"].nunique())
print("unique pair IDs:", df["pair_id"].nunique())
print("rows per pair:", Counter(pair_counts.values()))
print("B no-mask rows:", summary["B_no_mask_rows"])
print("C no-mask rows:", summary["C_no_mask_rows"])
print("B output:", B_JSONL)
print("C output:", C_JSONL)
print("ABCD JSONL:", ABCD_JSONL)
print("ABCD CSV:", ABCD_CSV)
print("summary:", SUMMARY_PATH)
