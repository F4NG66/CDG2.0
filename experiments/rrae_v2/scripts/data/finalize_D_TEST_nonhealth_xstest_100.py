#!/usr/bin/env python

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/path/to/rrae_steering_work_v2")

INPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "D_TEST_NONHEALTH_XSTEST_DEEPSEEK_AUDIT.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_D_benign_clean/"
      "D_TEST_NONHEALTH_XSTEST_FINAL_100.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "D_TEST_NONHEALTH_XSTEST_FINAL_100_summary.json"
)

TYPE_QUOTAS = {
    "homonyms": 10,
    "figurative_language": 11,
    "safe_targets": 10,
    "safe_contexts": 11,
    "definitions": 10,
    "nons_group_real_discr": 10,
    "real_group_nons_discr": 10,
    "historical_events": 10,
    "privacy_public": 8,
    "privacy_fictional": 10,
}

TARGET_TOTAL = 100

if sum(TYPE_QUOTAS.values()) != TARGET_TOTAL:
    raise RuntimeError(
        f"Type quotas sum to {sum(TYPE_QUOTAS.values())}, "
        f"expected {TARGET_TOTAL}"
    )


def load_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def normalize(text):
    return re.sub(r"\s+", " ", text.lower().strip())


rows = load_jsonl(INPUT_PATH)

accepted = [
    row
    for row in rows
    if row.get("audit_classification") == "ACCEPT_SAFE_NONHEALTH"
]

excluded = [
    row
    for row in rows
    if row.get("audit_classification") != "ACCEPT_SAFE_NONHEALTH"
]

by_type = defaultdict(list)

for row in accepted:
    by_type[row["xstest_type"]].append(row)

selected = []
type_counts = Counter()
opening_counts = Counter()

for type_name, quota in TYPE_QUOTAS.items():
    candidates = by_type[type_name][:]

    if len(candidates) < quota:
        raise RuntimeError(
            f"{type_name}: only {len(candidates)} accepted rows, "
            f"need {quota}"
        )

    # Prefer a spread of question openings within each type.
    while type_counts[type_name] < quota:
        row = min(
            candidates,
            key=lambda candidate: (
                opening_counts[candidate["opening_type"]],
                abs(len(candidate["prompt"].split()) - 11),
                candidate["source_id"],
            ),
        )

        selected.append(row)
        type_counts[type_name] += 1
        opening_counts[row["opening_type"]] += 1
        candidates.remove(row)

if len(selected) != TARGET_TOTAL:
    raise RuntimeError(
        f"Selected {len(selected)} rows, expected {TARGET_TOTAL}"
    )

final_rows = []

for offset, row in enumerate(selected):
    index = 400 + offset

    final_rows.append({
        "id": f"D_TEST_{index:04d}",
        "pair_id": f"TEST_PAIR_{index:04d}",
        "bucket": "D",
        "label": "benign",
        "domain": "non_health_control",
        "source_dataset": "XSTest",
        "source_split": "safe",
        "source_id": row["source_id"],
        "xstest_type": row["xstest_type"],
        "focus": row.get("focus"),
        "note": row.get("note"),
        "opening_type": row["opening_type"],
        "prompt": row["prompt"],
        "validation_status": "accepted_by_deepseek",
        "audit_confidence": row.get("audit_confidence"),
        "audit_reason": row.get("audit_reason"),
    })

normalized_prompts = [
    normalize(row["prompt"])
    for row in final_rows
]

if len(set(normalized_prompts)) != TARGET_TOTAL:
    raise RuntimeError("Duplicate prompts detected")

ids = [row["id"] for row in final_rows]
pair_ids = [row["pair_id"] for row in final_rows]

if len(set(ids)) != TARGET_TOTAL:
    raise RuntimeError("Duplicate D_TEST IDs detected")

if len(set(pair_ids)) != TARGET_TOTAL:
    raise RuntimeError("Duplicate TEST pair IDs detected")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in final_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "total_rows": len(final_rows),
    "type_counts": dict(type_counts),
    "opening_counts": dict(opening_counts),
    "validation_counts": dict(Counter(
        row["validation_status"]
        for row in final_rows
    )),
    "confidence_counts": dict(Counter(
        row["audit_confidence"]
        for row in final_rows
    )),
    "unique_prompts": len(set(normalized_prompts)),
    "unique_ids": len(set(ids)),
    "unique_pair_ids": len(set(pair_ids)),
    "excluded_rows": len(excluded),
    "first_id": ids[0],
    "last_id": ids[-1],
    "output": str(OUTPUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== D_TEST non-health finalized =====")
print("total rows:", len(final_rows))
print("types:", type_counts)
print("openings:", opening_counts)
print("validation:", Counter(
    row["validation_status"]
    for row in final_rows
))
print("confidence:", Counter(
    row["audit_confidence"]
    for row in final_rows
))
print("unique prompts:", len(set(normalized_prompts)))
print("unique ids:", len(set(ids)))
print("unique pair ids:", len(set(pair_ids)))
print("ID range:", ids[0], "to", ids[-1])
print("excluded rows:", len(excluded))
print("output:", OUTPUT_PATH)
print("summary:", SUMMARY_PATH)
