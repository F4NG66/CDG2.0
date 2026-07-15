#!/usr/bin/env python

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from rapidfuzz import fuzz, process


ROOT = Path("/path/to/rrae_steering_work_v2")

BLOCKLIST = (
    ROOT
    / "data/TEST_HELDOUT_V1/training_blocklist/"
      "RRAE_V2_TRAIN_BLOCKLIST_N1996.jsonl"
)

OUTPUT = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_CANDIDATE_400.jsonl"
)

SUMMARY = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_HEALTH_CARES_summary.json"
)

TARGET = 400
NEAR_THRESHOLD = 95.0


def normalize(text):
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"<\|mdm_mask\|>|<mask:\d+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def digest(text):
    return hashlib.sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()


def stable_key(row):
    material = (
        f"{row['harmful_level']}|"
        f"{row['generation_model']}|"
        f"{row['source_id']}|"
        f"{row['prompt']}"
    )
    return hashlib.sha256(material.encode()).hexdigest()


block_rows = [
    json.loads(line)
    for line in BLOCKLIST.open(encoding="utf-8")
    if line.strip()
]

blocked_hashes = {row["sha256"] for row in block_rows}
blocked_texts = [row["normalized_text"] for row in block_rows]

dataset = load_dataset("HFXM/CARES-18K")["test"]

eligible = []
seen = set()
stats = Counter()

for row in dataset:
    stats["input_rows"] += 1

    if row["harmful_level"] not in {2, 3}:
        stats["rejected_level"] += 1
        continue

    text = str(row["base_prompt"]).strip()

    if not text:
        stats["rejected_empty"] += 1
        continue

    key = digest(text)

    if key in seen:
        stats["rejected_duplicate"] += 1
        continue

    if key in blocked_hashes:
        stats["rejected_exact_overlap"] += 1
        continue

    normalized = normalize(text)

    near = process.extractOne(
        normalized,
        blocked_texts,
        scorer=fuzz.WRatio,
        score_cutoff=NEAR_THRESHOLD,
    )

    if near is not None:
        stats["rejected_near_overlap"] += 1
        continue

    seen.add(key)

    eligible.append({
        "source_dataset": "CARES-18K",
        "source_split": "test",
        "source_id": str(row["principle_index"]),
        "generation_model": row["generation_model"],
        "harmful_level": int(row["harmful_level"]),
        "original_method": row["method"],
        "bucket": "A",
        "label": "harmful",
        "domain": "health_related",
        "prompt": text,
        "clean_prompt": text,
        "normalized_sha256": key,
        "selection_status": "candidate_pending_manual_audit",
    })

groups = defaultdict(list)

for row in eligible:
    key = (
        row["harmful_level"],
        row["generation_model"],
    )
    groups[key].append(row)

for key in groups:
    groups[key].sort(key=stable_key)

group_keys = sorted(groups)
selected = []
offsets = {key: 0 for key in group_keys}

while len(selected) < TARGET:
    added = False

    for key in group_keys:
        index = offsets[key]

        if index >= len(groups[key]):
            continue

        selected.append(groups[key][index])
        offsets[key] += 1
        added = True

        if len(selected) >= TARGET:
            break

    if not added:
        break

if len(selected) < TARGET:
    raise RuntimeError(
        f"Only {len(selected)} eligible examples; need {TARGET}"
    )

for index, row in enumerate(selected):
    row["id"] = f"A_TEST_{index:04d}"
    row["pair_id"] = f"TEST_PAIR_{index:04d}"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "selected_total": len(selected),
    "levels": dict(Counter(r["harmful_level"] for r in selected)),
    "generation_models": dict(
        Counter(r["generation_model"] for r in selected)
    ),
    "original_methods": dict(
        Counter(r["original_method"] for r in selected)
    ),
    "pipeline_stats": dict(stats),
    "output": str(OUTPUT),
}

SUMMARY.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("selected total:", len(selected))
print("levels:", Counter(r["harmful_level"] for r in selected))
print(
    "generation models:",
    Counter(r["generation_model"] for r in selected),
)
print(
    "original methods:",
    Counter(r["original_method"] for r in selected),
)
print("pipeline stats:", stats)
print("output:", OUTPUT)
