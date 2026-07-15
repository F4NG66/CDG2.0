#!/usr/bin/env python

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from rapidfuzz.fuzz import ratio


ROOT = Path("/path/to/rrae_steering_work_v2")

OUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_NONHEALTH_EGIDA_CANDIDATE_120.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_NONHEALTH_EGIDA_CANDIDATE_120_summary.json"
)

BLOCKLIST_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/training_blocklist/"
      "RRAE_V2_TRAIN_BLOCKLIST_N1996.jsonl"
)

EXCLUDED_SOURCES = {
    "AdvBench",
    "HarmBench",
}

# Candidate pool target.
TARGET_TOTAL = 120

# Topic quotas deliberately sum to 120.
TOPIC_QUOTAS = {
    "fraud": 16,
    "drugs": 16,
    "guns": 12,
    "smuggling": 10,
    "p_info": 10,
    "robbery": 9,
    "bioweapons": 7,
    "cybercrime": 7,
    "trafficking": 6,
    "vandalism": 5,
    "fake_news": 4,
    "violence": 4,
    "arson": 3,
    "dangerous_activities": 3,
    "animal_crimes": 2,
    "piracy": 2,
    "terrorism": 1,
    "sex_crimes": 1,
    "bioterrorism": 1,
    "harassment": 1,
}

# Prevent one source from dominating the pool.
SOURCE_CAPS = {
    "DoNotAnswer": 24,
    "StrongREJECT": 24,
    "BSS": 22,
    "Misuse": 18,
    "SimpleSafetyTests": 16,
    "TDCRedTeaming": 10,
    "MaliciousInstructions": 10,
}


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "HPAI-BSC/Egida",
    "Egida",
    split="test",
)

block_rows = load_jsonl(BLOCKLIST_PATH)

blocked_texts = []

for row in block_rows:
    for key in (
        "text",
        "prompt",
        "clean_prompt",
        "question",
        "original_text",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            blocked_texts.append(value.strip())

blocked_norm = {normalize(text) for text in blocked_texts}

eligible = []
seen_questions = set()

for row in ds:
    if row["style"] != "BaseInstruction":
        continue

    if row["source"] in EXCLUDED_SOURCES:
        continue

    topics = row["topics"] or {}

    if topics.get("health", False):
        continue

    question = (row["question"] or "").strip()

    if not question:
        continue

    norm = normalize(question)

    if norm in seen_questions:
        continue

    seen_questions.add(norm)

    if norm in blocked_norm:
        continue

    active_topics = [
        topic
        for topic, active in topics.items()
        if active and topic != "health"
    ]

    if not active_topics:
        continue

    eligible.append({
        "uuid": row["uuid"],
        "source": row["source"],
        "style": row["style"],
        "question": question,
        "active_topics": active_topics,
    })

# Stable randomization for reproducibility.
random.Random(20260711).shuffle(eligible)

# Prefer rare topics first when assigning examples that have multiple labels.
topic_priority = sorted(
    TOPIC_QUOTAS,
    key=lambda topic: TOPIC_QUOTAS[topic],
)

selected = []
selected_norms = set()
topic_counts = Counter()
source_counts = Counter()

for topic in topic_priority:
    quota = TOPIC_QUOTAS[topic]

    candidates = [
        row
        for row in eligible
        if topic in row["active_topics"]
        and normalize(row["question"]) not in selected_norms
    ]

    # Prefer sources currently used less often.
    candidates.sort(
        key=lambda row: (
            source_counts[row["source"]],
            len(row["question"]),
            row["uuid"],
        )
    )

    for row in candidates:
        if topic_counts[topic] >= quota:
            break

        source = row["source"]

        if source_counts[source] >= SOURCE_CAPS.get(source, 999):
            continue

        out_row = {
            "temporary_id": f"A_TEST_NONHEALTH_CAND_{len(selected):04d}",
            "domain": "non_health_control",
            "label": "harmful",
            "bucket": "A",
            "source_dataset": "Egida",
            "source_split": "test",
            "source": source,
            "source_uuid": row["uuid"],
            "style": "BaseInstruction",
            "assigned_topic": topic,
            "active_topics": row["active_topics"],
            "prompt": row["question"],
            "validation_status": "pending_deepseek_audit",
        }

        selected.append(out_row)
        selected_norms.add(normalize(row["question"]))
        topic_counts[topic] += 1
        source_counts[source] += 1

# Fallback only if a topic quota could not be filled.
if len(selected) < TARGET_TOTAL:
    remaining = [
        row
        for row in eligible
        if normalize(row["question"]) not in selected_norms
    ]

    remaining.sort(
        key=lambda row: (
            source_counts[row["source"]],
            len(row["question"]),
            row["uuid"],
        )
    )

    for row in remaining:
        if len(selected) >= TARGET_TOTAL:
            break

        source = row["source"]

        if source_counts[source] >= SOURCE_CAPS.get(source, 999):
            continue

        available_topics = [
            topic
            for topic in row["active_topics"]
            if topic in TOPIC_QUOTAS
        ]

        if not available_topics:
            continue

        assigned_topic = min(
            available_topics,
            key=lambda topic: topic_counts[topic],
        )

        selected.append({
            "temporary_id": f"A_TEST_NONHEALTH_CAND_{len(selected):04d}",
            "domain": "non_health_control",
            "label": "harmful",
            "bucket": "A",
            "source_dataset": "Egida",
            "source_split": "test",
            "source": source,
            "source_uuid": row["uuid"],
            "style": "BaseInstruction",
            "assigned_topic": assigned_topic,
            "active_topics": row["active_topics"],
            "prompt": row["question"],
            "validation_status": "pending_deepseek_audit",
        })

        selected_norms.add(normalize(row["question"]))
        topic_counts[assigned_topic] += 1
        source_counts[source] += 1

if len(selected) != TARGET_TOTAL:
    raise RuntimeError(
        f"Expected {TARGET_TOTAL} selected rows, found {len(selected)}"
    )

# Final near-overlap check against training texts.
near_overlaps = []

for row in selected:
    candidate = normalize(row["prompt"])

    for blocked in blocked_norm:
        score = ratio(candidate, blocked)

        if score >= 90:
            near_overlaps.append({
                "candidate": row["prompt"],
                "blocked": blocked,
                "score": score,
            })
            break

if near_overlaps:
    raise RuntimeError(
        f"Found {len(near_overlaps)} near-overlaps with training data"
    )

with OUT_PATH.open("w", encoding="utf-8") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "total": len(selected),
    "source_counts": dict(source_counts),
    "assigned_topic_counts": dict(topic_counts),
    "unique_prompts": len({
        normalize(row["prompt"])
        for row in selected
    }),
    "exact_training_overlap": 0,
    "near_training_overlap": 0,
    "output": str(OUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== A_TEST non-health candidate pool =====")
print("total:", len(selected))
print("sources:", source_counts)
print("assigned topics:", topic_counts)
print("unique prompts:", summary["unique_prompts"])
print("exact training overlap:", 0)
print("near training overlap:", 0)
print("output:", OUT_PATH)
print("summary:", SUMMARY_PATH)
