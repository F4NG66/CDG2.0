#!/usr/bin/env python

import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/path/to/rrae_steering_work_v2")

INPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_NONHEALTH_EGIDA_DEEPSEEK_AUDIT.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_NONHEALTH_EGIDA_FINAL_100.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_NONHEALTH_EGIDA_FINAL_100_summary.json"
)

TARGET_TOTAL = 100

# Scaled version of the original 120-example topic distribution.
TOPIC_QUOTAS = {
    "fraud": 13,
    "drugs": 13,
    "guns": 10,
    "smuggling": 8,
    "p_info": 8,
    "robbery": 8,
    "bioweapons": 6,
    "cybercrime": 6,
    "trafficking": 5,
    "vandalism": 4,
    "fake_news": 3,
    "violence": 3,
    "dangerous_activities": 3,
    "arson": 2,
    "animal_crimes": 2,
    "piracy": 2,
    "terrorism": 1,
    "sex_crimes": 1,
    "bioterrorism": 1,
    "harassment": 1,
}

if sum(TOPIC_QUOTAS.values()) != TARGET_TOTAL:
    raise RuntimeError(
        f"Topic quotas sum to {sum(TOPIC_QUOTAS.values())}, "
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
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


rows = load_jsonl(INPUT_PATH)

accepted = [
    row
    for row in rows
    if row.get("audit_classification") == "ACTIONABLE_HARMFUL"
]

rejected = [
    row
    for row in rows
    if row.get("audit_classification") != "ACTIONABLE_HARMFUL"
]

print("audited rows:", len(rows))
print("accepted rows:", len(accepted))
print("excluded rows:", len(rejected))

for row in rejected:
    print(
        "excluded:",
        row["temporary_id"],
        "|", row.get("audit_classification"),
        "|", row["prompt"],
    )

if len(accepted) < TARGET_TOTAL:
    raise RuntimeError(
        f"Only {len(accepted)} accepted rows; need {TARGET_TOTAL}"
    )

# Deterministic shuffle.
random.Random(20260711).shuffle(accepted)

selected = []
selected_ids = set()
source_counts = Counter()
topic_counts = Counter()

for topic, quota in TOPIC_QUOTAS.items():
    candidates = [
        row
        for row in accepted
        if row["assigned_topic"] == topic
        and row["temporary_id"] not in selected_ids
    ]

    # Prefer sources currently used less often.
    candidates.sort(
        key=lambda row: (
            source_counts[row["source"]],
            row["temporary_id"],
        )
    )

    if len(candidates) < quota:
        raise RuntimeError(
            f"Topic {topic} has only {len(candidates)} accepted rows, "
            f"but quota is {quota}"
        )

    for row in candidates[:quota]:
        selected.append(row)
        selected_ids.add(row["temporary_id"])
        source_counts[row["source"]] += 1
        topic_counts[topic] += 1

if len(selected) != TARGET_TOTAL:
    raise RuntimeError(
        f"Selected {len(selected)} rows, expected {TARGET_TOTAL}"
    )

final_rows = []

for index, row in enumerate(selected):
    final_rows.append({
        "id": f"A_TEST_{400 + index:04d}",
        "pair_id": f"TEST_PAIR_{400 + index:04d}",
        "bucket": "A",
        "label": "harmful",
        "domain": "non_health_control",
        "source_dataset": "Egida",
        "source_split": "test",
        "source": row["source"],
        "source_uuid": row["source_uuid"],
        "style": "BaseInstruction",
        "assigned_topic": row["assigned_topic"],
        "active_topics": row["active_topics"],
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
    "domain_counts": dict(Counter(
        row["domain"] for row in final_rows
    )),
    "label_counts": dict(Counter(
        row["label"] for row in final_rows
    )),
    "source_counts": dict(source_counts),
    "topic_counts": dict(topic_counts),
    "validation_counts": dict(Counter(
        row["validation_status"] for row in final_rows
    )),
    "confidence_counts": dict(Counter(
        row["audit_confidence"] for row in final_rows
    )),
    "unique_prompts": len(unique_prompts),
    "excluded_nonharmful_rows": len(rejected),
    "output": str(OUTPUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n===== A_TEST non-health finalized =====")
print("total rows:", len(final_rows))
print("sources:", source_counts)
print("topics:", topic_counts)
print("validation:", Counter(
    row["validation_status"] for row in final_rows
))
print("confidence:", Counter(
    row["audit_confidence"] for row in final_rows
))
print("unique prompts:", len(unique_prompts))
print("output:", OUTPUT_PATH)
print("summary:", SUMMARY_PATH)
