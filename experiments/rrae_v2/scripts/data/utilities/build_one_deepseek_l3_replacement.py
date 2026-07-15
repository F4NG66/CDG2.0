#!/usr/bin/env python

import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from rapidfuzz import fuzz, process


ROOT = Path("/path/to/rrae_steering_work_v2")

ORIGINAL = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_CANDIDATE_400.jsonl"
)

REPLACEMENTS = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_REPLACEMENTS_9.jsonl"
)

BLOCKLIST = (
    ROOT
    / "data/TEST_HELDOUT_V1/training_blocklist/"
      "RRAE_V2_TRAIN_BLOCKLIST_N1996.jsonl"
)

OUTPUT = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_DEEPSEEK_L3_REPLACEMENT_1.jsonl"
)

NEAR_THRESHOLD = 95.0


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


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


original_rows = load_jsonl(ORIGINAL)
replacement_rows = load_jsonl(REPLACEMENTS)
block_rows = load_jsonl(BLOCKLIST)

used_rows = original_rows + replacement_rows

used_hashes = {
    digest(row["prompt"])
    for row in used_rows
}

used_texts = [
    normalize(row["prompt"])
    for row in used_rows
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

candidates = []

for row in dataset:
    if int(row["harmful_level"]) != 3:
        continue

    if str(row["generation_model"]).lower().strip() != "deepseek":
        continue

    text = str(row["base_prompt"]).strip()

    if not text:
        continue

    key = digest(text)

    if key in used_hashes:
        continue

    if key in blocked_hashes:
        continue

    normalized = normalize(text)

    near_training = process.extractOne(
        normalized,
        blocked_texts,
        scorer=fuzz.WRatio,
        score_cutoff=NEAR_THRESHOLD,
    )

    if near_training is not None:
        continue

    near_test = process.extractOne(
        normalized,
        used_texts,
        scorer=fuzz.WRatio,
        score_cutoff=NEAR_THRESHOLD,
    )

    if near_test is not None:
        continue

    candidates.append({
        "principle_index": row["principle_index"],
        "generation_model": "deepseek",
        "harmful_level": 3,
        "original_method": row["method"],
        "prompt": text,
        "clean_prompt": text,
        "normalized_sha256": key,
    })

candidates.sort(
    key=lambda row: hashlib.sha256(
        (
            f"{row['principle_index']}|"
            f"{row['original_method']}|"
            f"{row['prompt']}"
        ).encode("utf-8")
    ).hexdigest()
)

if not candidates:
    raise RuntimeError("No eligible Gemini level-2 replacement found.")

selected = candidates[0]

selected.update({
    "id": "A_TEST_REPL_10",
    "pair_id": "TEST_REPLACEMENT_PAIR_10",
    "source_dataset": "CARES-18K",
    "source_split": "test",
    "source_id": str(selected.pop("principle_index")),
    "bucket": "A",
    "label": "harmful",
    "domain": "health_related",
    "selection_status": "replacement_pending_manual_audit",
})

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    f.write(json.dumps(selected, ensure_ascii=False) + "\n")

print("replacement id:", selected["id"])
print("level:", selected["harmful_level"])
print("model:", selected["generation_model"])
print("method:", selected["original_method"])
print("prompt:", selected["prompt"])
print("output:", OUTPUT)
