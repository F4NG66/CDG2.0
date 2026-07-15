#!/usr/bin/env python

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/path/to/rrae_steering_work_v2")

HEALTH_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_FINAL_400.jsonl"
)

NONHEALTH_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_NONHEALTH_EGIDA_FINAL_100.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HARMFUL_CLEAN_FINAL_500.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_HARMFUL_CLEAN_FINAL_500_summary.json"
)


def load_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


health_rows = load_jsonl(HEALTH_PATH)
nonhealth_rows = load_jsonl(NONHEALTH_PATH)

if len(health_rows) != 400:
    raise RuntimeError(
        f"Expected 400 health rows, found {len(health_rows)}"
    )

if len(nonhealth_rows) != 100:
    raise RuntimeError(
        f"Expected 100 non-health rows, found {len(nonhealth_rows)}"
    )

final_rows = health_rows + nonhealth_rows

if len(final_rows) != 500:
    raise RuntimeError(
        f"Expected 500 total rows, found {len(final_rows)}"
    )

# Standardize and verify IDs.
for index, row in enumerate(final_rows):
    row["id"] = f"A_TEST_{index:04d}"
    row["pair_id"] = f"TEST_PAIR_{index:04d}"
    row["bucket"] = "A"
    row["label"] = "harmful"

    if index < 400:
        row["domain"] = "health_related"
    else:
        row["domain"] = "non_health_control"

ids = [row["id"] for row in final_rows]
pair_ids = [row["pair_id"] for row in final_rows]
prompts = [normalize(row["prompt"]) for row in final_rows]

if len(set(ids)) != 500:
    raise RuntimeError("Duplicate A_TEST IDs detected")

if len(set(pair_ids)) != 500:
    raise RuntimeError("Duplicate TEST pair IDs detected")

if len(set(prompts)) != 500:
    duplicates = [
        prompt
        for prompt, count in Counter(prompts).items()
        if count > 1
    ]

    raise RuntimeError(
        f"Duplicate prompts detected: {len(duplicates)}"
    )

domain_counts = Counter(
    row["domain"] for row in final_rows
)

label_counts = Counter(
    row["label"] for row in final_rows
)

bucket_counts = Counter(
    row["bucket"] for row in final_rows
)

source_dataset_counts = Counter(
    row.get("source_dataset", "unknown")
    for row in final_rows
)

validation_counts = Counter(
    row.get("validation_status", "unknown")
    for row in final_rows
)

expected_domains = Counter({
    "health_related": 400,
    "non_health_control": 100,
})

if domain_counts != expected_domains:
    raise RuntimeError(
        f"Unexpected domain balance: {domain_counts}"
    )

if label_counts != Counter({"harmful": 500}):
    raise RuntimeError(
        f"Unexpected labels: {label_counts}"
    )

if bucket_counts != Counter({"A": 500}):
    raise RuntimeError(
        f"Unexpected buckets: {bucket_counts}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in final_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "total_rows": len(final_rows),
    "domain_counts": dict(domain_counts),
    "label_counts": dict(label_counts),
    "bucket_counts": dict(bucket_counts),
    "source_dataset_counts": dict(source_dataset_counts),
    "validation_counts": dict(validation_counts),
    "unique_ids": len(set(ids)),
    "unique_pair_ids": len(set(pair_ids)),
    "unique_prompts": len(set(prompts)),
    "first_id": ids[0],
    "last_id": ids[-1],
    "output": str(OUTPUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== A_TEST finalized =====")
print("total rows:", len(final_rows))
print("domains:", domain_counts)
print("labels:", label_counts)
print("buckets:", bucket_counts)
print("source datasets:", source_dataset_counts)
print("validation:", validation_counts)
print("unique ids:", len(set(ids)))
print("unique pair ids:", len(set(pair_ids)))
print("unique prompts:", len(set(prompts)))
print("ID range:", ids[0], "to", ids[-1])
print("output:", OUTPUT_PATH)
print("summary:", SUMMARY_PATH)
