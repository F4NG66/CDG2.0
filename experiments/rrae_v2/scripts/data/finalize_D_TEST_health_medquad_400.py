#!/usr/bin/env python

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/path/to/rrae_steering_work_v2")

INPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "D_TEST_HEALTH_MEDQUAD_DEEPSEEK_AUDIT.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_D_benign_clean/"
      "D_TEST_HEALTH_MEDQUAD_FINAL_400.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "D_TEST_HEALTH_MEDQUAD_FINAL_400_summary.json"
)

TARGET_TOTAL = 400

SOURCE_QUOTAS = {
    "10_MPlus_ADAM_QA": 60,
    "11_MPlusDrugs_QA": 52,
    "2_GARD_QA": 44,
    "3_GHR_QA": 44,
    "6_NINDS_QA": 36,
    "12_MPlusHerbsSupplements_QA": 32,
    "4_MPlus_Health_Topics_QA": 32,
    "5_NIDDK_QA": 28,
    "1_CancerGov_QA": 24,
    "8_NHLBI_QA_XML": 20,
    "9_CDC_QA": 16,
    "7_SeniorHealth_QA": 12,
}

if sum(SOURCE_QUOTAS.values()) != TARGET_TOTAL:
    raise RuntimeError(
        f"Source quotas sum to {sum(SOURCE_QUOTAS.values())}, "
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
    if row.get("audit_classification") == "BENIGN_WELL_FORMED"
]

rejected = [
    row
    for row in rows
    if row.get("audit_classification") != "BENIGN_WELL_FORMED"
]

if len(accepted) < TARGET_TOTAL:
    raise RuntimeError(
        f"Only {len(accepted)} accepted rows; need {TARGET_TOTAL}"
    )

by_source = defaultdict(list)

for row in accepted:
    by_source[row["source_folder"]].append(row)

selected = []
selected_ids = set()
source_counts = Counter()
type_counts = Counter()
opening_counts = Counter()

for source, quota in SOURCE_QUOTAS.items():
    remaining = by_source[source][:]

    source_selected = 0
    target_for_source = min(quota, len(remaining))

    while source_selected < target_for_source:
        valid = [
            row
            for row in remaining
            if row["temporary_id"] not in selected_ids
        ]

        if not valid:
            break

        row = min(
            valid,
            key=lambda candidate: (
                opening_counts[candidate["opening_type"]],
                type_counts[candidate["question_type"]],
                abs(len(candidate["prompt"].split()) - 14),
                candidate["temporary_id"],
            ),
        )

        selected.append(row)
        selected_ids.add(row["temporary_id"])
        source_counts[source] += 1
        type_counts[row["question_type"]] += 1
        opening_counts[row["opening_type"]] += 1
        source_selected += 1

        remaining.remove(row)

# Fill any remaining slots from all still-unused accepted rows.
if len(selected) < TARGET_TOTAL:
    remaining_global = [
        row
        for row in accepted
        if row["temporary_id"] not in selected_ids
    ]

    while len(selected) < TARGET_TOTAL:
        if not remaining_global:
            raise RuntimeError(
                f"Could only select {len(selected)} rows; "
                f"need {TARGET_TOTAL}"
            )

        row = min(
            remaining_global,
            key=lambda candidate: (
                source_counts[candidate["source_folder"]],
                opening_counts[candidate["opening_type"]],
                type_counts[candidate["question_type"]],
                abs(len(candidate["prompt"].split()) - 14),
                candidate["temporary_id"],
            ),
        )

        selected.append(row)
        selected_ids.add(row["temporary_id"])
        source_counts[row["source_folder"]] += 1
        type_counts[row["question_type"]] += 1
        opening_counts[row["opening_type"]] += 1

        remaining_global.remove(row)

if len(selected) != TARGET_TOTAL:
    raise RuntimeError(
        f"Selected {len(selected)} rows, expected {TARGET_TOTAL}"
    )

final_rows = []

for index, row in enumerate(selected):
    final_rows.append({
        "id": f"D_TEST_{index:04d}",
        "pair_id": f"TEST_PAIR_{index:04d}",
        "bucket": "D",
        "label": "benign",
        "domain": "health_related",
        "source_dataset": "MedQuAD",
        "source_file": row["source_file"],
        "source_folder": row["source_folder"],
        "focus": row.get("focus"),
        "category": row.get("category"),
        "question_type": row["question_type"],
        "opening_type": row["opening_type"],
        "prompt": row["prompt"],
        "validation_status": "accepted_by_deepseek",
        "audit_confidence": row.get("audit_confidence"),
        "audit_reason": row.get("audit_reason"),
    })

unique_prompts = {
    normalize(row["prompt"])
    for row in final_rows
}

if len(unique_prompts) != TARGET_TOTAL:
    raise RuntimeError(
        f"Expected {TARGET_TOTAL} unique prompts, "
        f"found {len(unique_prompts)}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in final_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "total_rows": len(final_rows),
    "source_counts": dict(source_counts),
    "question_type_counts": dict(type_counts),
    "opening_counts": dict(opening_counts),
    "category_counts": dict(Counter(
        row.get("category") or "UNKNOWN"
        for row in final_rows
    )),
    "validation_counts": dict(Counter(
        row["validation_status"]
        for row in final_rows
    )),
    "confidence_counts": dict(Counter(
        row["audit_confidence"]
        for row in final_rows
    )),
    "unique_prompts": len(unique_prompts),
    "malformed_excluded": len(rejected),
    "output": str(OUTPUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== D_TEST health finalized =====")
print("total rows:", len(final_rows))
print("sources:", source_counts)
print("question types:", type_counts)
print("openings:", opening_counts)
print("categories:", Counter(
    row.get("category") or "UNKNOWN"
    for row in final_rows
))
print("validation:", Counter(
    row["validation_status"]
    for row in final_rows
))
print("confidence:", Counter(
    row["audit_confidence"]
    for row in final_rows
))
print("unique prompts:", len(unique_prompts))
print("malformed excluded:", len(rejected))
print("output:", OUTPUT_PATH)
print("summary:", SUMMARY_PATH)
