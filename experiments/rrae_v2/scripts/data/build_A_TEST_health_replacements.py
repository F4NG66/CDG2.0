#!/usr/bin/env python

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from rapidfuzz import fuzz, process


ROOT = Path("/path/to/rrae_steering_work_v2")

CURRENT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_CANDIDATE_400.jsonl"
)

DECISIONS_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_HEALTH_CARES_MANUAL_DECISIONS.json"
)

BLOCKLIST_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/training_blocklist/"
      "RRAE_V2_TRAIN_BLOCKLIST_N1996.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_REPLACEMENTS_9.jsonl"
)

NEAR_THRESHOLD = 95.0

# Exact replacement requirements by generation model and harmful level.
NEEDED = {
    ("gemini", 2): 3,
    ("gemini", 3): 1,
    ("claude", 2): 1,
    ("claude", 3): 2,
    ("gpt", 2): 1,
    ("deepseek", 2): 1,
}


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
    text = (
        f"{row['generation_model']}|"
        f"{row['harmful_level']}|"
        f"{row['principle_index']}|"
        f"{row['prompt']}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


current_rows = load_jsonl(CURRENT_PATH)
decisions = json.loads(
    DECISIONS_PATH.read_text(encoding="utf-8")
)
block_rows = load_jsonl(BLOCKLIST_PATH)

removed_ids = set(decisions["manual_remove"])

kept_rows = [
    row for row in current_rows
    if row["id"] not in removed_ids
]

used_hashes = {
    digest(row["prompt"])
    for row in current_rows
}

kept_texts = [
    normalize(row["prompt"])
    for row in kept_rows
]

blocked_hashes = {
    row["sha256"]
    for row in block_rows
}

blocked_texts = [
    row["normalized_text"]
    for row in block_rows
]

dataset = load_dataset("HFXM/CARES-18K")["test"]

eligible = []
seen = set()
stats = Counter()

for row in dataset:
    stats["input_rows"] += 1

    level = int(row["harmful_level"])
    model = str(row["generation_model"]).lower().strip()

    if (model, level) not in NEEDED:
        continue

    text = str(row["base_prompt"]).strip()

    if not text:
        stats["empty"] += 1
        continue

    key = digest(text)

    # Do not reuse anything from the original 400, including removed rows.
    if key in used_hashes:
        stats["already_used_original_400"] += 1
        continue

    if key in seen:
        stats["duplicate_candidate"] += 1
        continue

    if key in blocked_hashes:
        stats["exact_training_overlap"] += 1
        continue

    normalized = normalize(text)

    near_training = process.extractOne(
        normalized,
        blocked_texts,
        scorer=fuzz.WRatio,
        score_cutoff=NEAR_THRESHOLD,
    )

    if near_training is not None:
        stats["near_training_overlap"] += 1
        continue

    near_kept = process.extractOne(
        normalized,
        kept_texts,
        scorer=fuzz.WRatio,
        score_cutoff=NEAR_THRESHOLD,
    )

    if near_kept is not None:
        stats["near_current_test_overlap"] += 1
        continue

    seen.add(key)

    eligible.append({
        "principle_index": row["principle_index"],
        "generation_model": model,
        "harmful_level": level,
        "original_method": row["method"],
        "prompt": text,
        "clean_prompt": text,
        "normalized_sha256": key,
    })

eligible.sort(key=stable_key)

selected = []
counts = Counter()

for row in eligible:
    group = (
        row["generation_model"],
        row["harmful_level"],
    )

    if counts[group] >= NEEDED[group]:
        continue

    selected.append(row)
    counts[group] += 1

    if all(counts[key] >= value for key, value in NEEDED.items()):
        break

missing = {
    str(key): required - counts[key]
    for key, required in NEEDED.items()
    if counts[key] < required
}

if missing:
    raise RuntimeError(
        f"Could not satisfy replacement distribution: {missing}"
    )

for index, row in enumerate(selected):
    row.update({
        "id": f"A_TEST_REPL_{index:02d}",
        "pair_id": f"TEST_REPLACEMENT_PAIR_{index:02d}",
        "source_dataset": "CARES-18K",
        "source_split": "test",
        "source_id": str(row.pop("principle_index")),
        "bucket": "A",
        "label": "harmful",
        "domain": "health_related",
        "selection_status": "replacement_pending_deepseek_audit",
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("selected replacements:", len(selected))
print(
    "model × level:",
    Counter(
        (row["generation_model"], row["harmful_level"])
        for row in selected
    ),
)
print(
    "methods:",
    Counter(row["original_method"] for row in selected),
)
print("pipeline stats:", stats)
print("output:", OUTPUT_PATH)

print("\n===== replacement prompts =====")
for row in selected:
    print(
        f"{row['id']} | "
        f"level={row['harmful_level']} | "
        f"model={row['generation_model']} | "
        f"method={row['original_method']} | "
        f"{row['prompt']}"
    )
