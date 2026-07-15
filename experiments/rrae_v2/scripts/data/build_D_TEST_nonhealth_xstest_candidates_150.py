#!/usr/bin/env python

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio


ROOT = Path("/path/to/rrae_steering_work_v2")

INPUT_PATH = (
    ROOT
    / "external_datasets/TEST_HELDOUT_V1/xstest/"
      "xstest_prompts.csv"
)

BLOCKLIST_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/training_blocklist/"
      "RRAE_V2_TRAIN_BLOCKLIST_N1996.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_D_benign_clean/"
      "D_TEST_NONHEALTH_XSTEST_CANDIDATE_150.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "D_TEST_NONHEALTH_XSTEST_CANDIDATE_150_summary.json"
)

SAFE_TYPES = [
    "homonyms",
    "figurative_language",
    "safe_targets",
    "safe_contexts",
    "definitions",
    "nons_group_real_discr",
    "real_group_nons_discr",
    "historical_events",
    "privacy_public",
    "privacy_fictional",
]

PER_TYPE = 15
TARGET_TOTAL = len(SAFE_TYPES) * PER_TYPE


def normalize(text):
    text = str(text).lower().strip()
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


df = pd.read_csv(INPUT_PATH)

safe = df[
    df["label"].astype(str).str.lower().eq("safe")
].copy()

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

blocked_norm = {
    normalize(text)
    for text in blocked_texts
}

selected = []
selected_norms = set()
type_counts = Counter()
opening_counts = Counter()

for type_name in SAFE_TYPES:
    group = safe[safe["type"] == type_name].copy()

    # Stable ordering by original XSTest ID.
    group = group.sort_values("id")

    type_selected = 0

    for _, row in group.iterrows():
        prompt = str(row["prompt"]).strip()
        norm = normalize(prompt)

        if not prompt or norm in selected_norms:
            continue

        if norm in blocked_norm:
            continue

        first_word = prompt.split()[0].lower().rstrip("?,")

        selected.append({
            "temporary_id": (
                f"D_TEST_NONHEALTH_CAND_{len(selected):04d}"
            ),
            "bucket": "D",
            "label": "benign",
            "domain": "non_health_control",
            "source_dataset": "XSTest",
            "source_split": "safe",
            "source_id": int(row["id"]),
            "xstest_type": type_name,
            "focus": (
                None
                if pd.isna(row["focus"])
                else str(row["focus"])
            ),
            "note": (
                None
                if pd.isna(row["note"])
                else str(row["note"])
            ),
            "opening_type": first_word,
            "prompt": prompt,
            "validation_status": "pending_deepseek_audit",
        })

        selected_norms.add(norm)
        type_counts[type_name] += 1
        opening_counts[first_word] += 1
        type_selected += 1

        if type_selected >= PER_TYPE:
            break

    if type_selected != PER_TYPE:
        raise RuntimeError(
            f"{type_name}: selected {type_selected}, "
            f"expected {PER_TYPE}"
        )

if len(selected) != TARGET_TOTAL:
    raise RuntimeError(
        f"Selected {len(selected)}, expected {TARGET_TOTAL}"
    )

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

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "total_rows": len(selected),
    "type_counts": dict(type_counts),
    "opening_counts": dict(opening_counts),
    "unique_prompts": len(selected_norms),
    "exact_training_overlap": 0,
    "near_training_overlap": 0,
    "output": str(OUTPUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== D_TEST non-health candidate pool =====")
print("total:", len(selected))
print("types:", type_counts)
print("openings:", opening_counts)
print("unique prompts:", len(selected_norms))
print("exact training overlap:", 0)
print("near training overlap:", 0)

print("\n===== SAMPLE BY TYPE =====")
for type_name in SAFE_TYPES:
    print(f"\n--- {type_name} ---")

    examples = [
        row for row in selected
        if row["xstest_type"] == type_name
    ][:3]

    for row in examples:
        print(row["prompt"])

print("\noutput:", OUTPUT_PATH)
print("summary:", SUMMARY_PATH)
