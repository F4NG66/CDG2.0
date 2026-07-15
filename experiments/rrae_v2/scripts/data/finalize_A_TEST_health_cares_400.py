#!/usr/bin/env python

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process


ROOT = Path("/path/to/rrae_steering_work_v2")

ORIGINAL_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_CANDIDATE_400.jsonl"
)

AUDIT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_HEALTH_CARES_DEEPSEEK_AUDIT.jsonl"
)

DECISIONS_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_HEALTH_CARES_MANUAL_DECISIONS.json"
)

REPLACEMENTS_9_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_REPLACEMENTS_9.jsonl"
)

REPLACEMENT_1_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_GEMINI_L2_REPLACEMENT_1.jsonl"
)

REPLACEMENT_DEEPSEEK_L3_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_DEEPSEEK_L3_REPLACEMENT_1.jsonl"
)

REPLACEMENT_CLAUDE_L3_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CLAUDE_L3_REPLACEMENT_1.jsonl"
)

BLOCKLIST_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/training_blocklist/"
      "RRAE_V2_TRAIN_BLOCKLIST_N1996.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_HEALTH_CARES_FINAL_400.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_HEALTH_CARES_FINAL_400_summary.json"
)

NEAR_THRESHOLD = 95.0


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"<\|mdm_mask\|>|<mask:\d+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def digest(text: str) -> str:
    return hashlib.sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()


for path in [
    ORIGINAL_PATH,
    AUDIT_PATH,
    DECISIONS_PATH,
    REPLACEMENTS_9_PATH,
    REPLACEMENT_1_PATH,
    REPLACEMENT_DEEPSEEK_L3_PATH,
    REPLACEMENT_CLAUDE_L3_PATH,
    BLOCKLIST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)

original_rows = load_jsonl(ORIGINAL_PATH)
audit_rows = load_jsonl(AUDIT_PATH)
decisions = json.loads(
    DECISIONS_PATH.read_text(encoding="utf-8")
)
replacement_rows = load_jsonl(REPLACEMENTS_9_PATH)
replacement_1_rows = load_jsonl(REPLACEMENT_1_PATH)
replacement_deepseek_l3_rows = load_jsonl(
    REPLACEMENT_DEEPSEEK_L3_PATH
)
replacement_claude_l3_rows = load_jsonl(
    REPLACEMENT_CLAUDE_L3_PATH
)
block_rows = load_jsonl(BLOCKLIST_PATH)

audit_by_id = {
    row["id"]: row
    for row in audit_rows
}

manual_keep = set(
    decisions["manual_keep_as_actionable_harmful"]
)
manual_remove = set(
    decisions["manual_remove"]
)

accepted_original = []

for row in original_rows:
    row_id = row["id"]
    audit = audit_by_id.get(row_id)

    if audit is None:
        raise RuntimeError(
            f"Missing DeepSeek audit for {row_id}"
        )

    deepseek_accept = (
        audit.get("audit_label") == "ACTIONABLE_HARMFUL"
        and float(audit.get("audit_confidence") or 0) >= 0.80
    )

    if row_id in manual_remove:
        continue

    if deepseek_accept or row_id in manual_keep:
        new_row = dict(row)
        new_row["validation_status"] = (
            "accepted_by_deepseek"
            if deepseek_accept
            else "accepted_by_manual_review"
        )
        new_row["deepseek_audit_label"] = audit.get("audit_label")
        new_row["deepseek_audit_confidence"] = audit.get(
            "audit_confidence"
        )
        new_row["deepseek_audit_reason"] = audit.get(
            "audit_reason"
        )
        accepted_original.append(new_row)

if len(accepted_original) != 391:
    raise RuntimeError(
        f"Expected 391 accepted original rows, "
        f"found {len(accepted_original)}"
    )

accepted_replacements = []

for row in replacement_rows:
    if row["id"] in {
        "A_TEST_REPL_00",
        "A_TEST_REPL_06",
        "A_TEST_REPL_07",
    }:
        continue

    new_row = dict(row)
    new_row["validation_status"] = (
        "accepted_by_manual_review_replacement"
    )
    accepted_replacements.append(new_row)

if len(accepted_replacements) != 6:
    raise RuntimeError(
        f"Expected 6 accepted rows from replacement file, "
        f"found {len(accepted_replacements)}"
    )

if len(replacement_1_rows) != 1:
    raise RuntimeError(
        f"Expected exactly 1 final replacement, "
        f"found {len(replacement_1_rows)}"
    )

replacement_09 = dict(replacement_1_rows[0])
replacement_09["validation_status"] = (
    "accepted_by_manual_review_replacement"
)
accepted_replacements.append(replacement_09)

if len(replacement_deepseek_l3_rows) != 1:
    raise RuntimeError(
        f"Expected exactly 1 DeepSeek level-3 replacement, "
        f"found {len(replacement_deepseek_l3_rows)}"
    )

replacement_10 = dict(replacement_deepseek_l3_rows[0])
replacement_10["validation_status"] = (
    "accepted_by_manual_review_replacement"
)
accepted_replacements.append(replacement_10)

if len(replacement_claude_l3_rows) != 1:
    raise RuntimeError(
        f"Expected exactly 1 Claude level-3 replacement, "
        f"found {len(replacement_claude_l3_rows)}"
    )

replacement_11 = dict(replacement_claude_l3_rows[0])
replacement_11["validation_status"] = (
    "accepted_by_manual_review_replacement"
)
accepted_replacements.append(replacement_11)

if len(accepted_replacements) != 9:
    raise RuntimeError(
        f"Expected 9 total replacements, "
        f"found {len(accepted_replacements)}"
    )

final_rows = accepted_original + accepted_replacements

if len(final_rows) != 400:
    raise RuntimeError(
        f"Expected 400 final rows, found {len(final_rows)}"
    )

# Reassign clean, sequential IDs and pair IDs.
for index, row in enumerate(final_rows):
    row["original_candidate_id"] = row["id"]
    row["id"] = f"A_TEST_{index:04d}"
    row["pair_id"] = f"TEST_PAIR_{index:04d}"
    row["bucket"] = "A"
    row["label"] = "harmful"
    row["domain"] = "health_related"
    row["source_dataset"] = "CARES-18K"
    row["source_split"] = "test"
    row["prompt"] = str(row["prompt"]).strip()
    row["clean_prompt"] = row["prompt"]
    row["normalized_sha256"] = digest(row["prompt"])
    row["selection_status"] = "final_validated"

ids = [row["id"] for row in final_rows]
pair_ids = [row["pair_id"] for row in final_rows]
hashes = [row["normalized_sha256"] for row in final_rows]

if len(set(ids)) != 400:
    raise RuntimeError("Duplicate final IDs detected")

if len(set(pair_ids)) != 400:
    raise RuntimeError("Duplicate pair IDs detected")

if len(set(hashes)) != 400:
    duplicates = [
        item
        for item, count in Counter(hashes).items()
        if count > 1
    ]
    raise RuntimeError(
        f"Duplicate normalized prompts detected: {duplicates[:10]}"
    )

levels = Counter(
    row["harmful_level"]
    for row in final_rows
)

models = Counter(
    row["generation_model"]
    for row in final_rows
)

methods = Counter(
    row["original_method"]
    for row in final_rows
)

if levels != Counter({2: 200, 3: 200}):
    raise RuntimeError(
        f"Unexpected harmful-level balance: {levels}"
    )

expected_models = Counter({
    "claude": 100,
    "deepseek": 100,
    "gemini": 100,
    "gpt": 100,
})

if models != expected_models:
    raise RuntimeError(
        f"Unexpected generation-model balance: {models}"
    )

blocked_hashes = {
    row["sha256"]
    for row in block_rows
}

exact_overlap_ids = [
    row["id"]
    for row in final_rows
    if row["normalized_sha256"] in blocked_hashes
]

if exact_overlap_ids:
    raise RuntimeError(
        f"Exact training overlap found: {exact_overlap_ids}"
    )

blocked_texts = [
    row["normalized_text"]
    for row in block_rows
]

near_overlaps = []

for row in final_rows:
    normalized = normalize(row["prompt"])

    match = process.extractOne(
        normalized,
        blocked_texts,
        scorer=fuzz.WRatio,
        score_cutoff=NEAR_THRESHOLD,
    )

    if match is not None:
        matched_text, score, matched_index = match
        near_overlaps.append({
            "id": row["id"],
            "score": score,
            "prompt": row["prompt"],
            "matched_training_text": matched_text,
            "matched_training_metadata": block_rows[
                matched_index
            ],
        })

if near_overlaps:
    raise RuntimeError(
        f"Near training overlaps found: {len(near_overlaps)}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in final_rows:
        f.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

summary = {
    "status": "FINAL_VALIDATED",
    "total_rows": len(final_rows),
    "bucket_counts": dict(
        Counter(row["bucket"] for row in final_rows)
    ),
    "label_counts": dict(
        Counter(row["label"] for row in final_rows)
    ),
    "domain_counts": dict(
        Counter(row["domain"] for row in final_rows)
    ),
    "harmful_level_counts": dict(levels),
    "generation_model_counts": dict(models),
    "original_method_counts": dict(methods),
    "validation_status_counts": dict(
        Counter(
            row["validation_status"]
            for row in final_rows
        )
    ),
    "unique_ids": len(set(ids)),
    "unique_pair_ids": len(set(pair_ids)),
    "unique_normalized_prompts": len(set(hashes)),
    "exact_training_overlap_count": len(
        exact_overlap_ids
    ),
    "near_training_overlap_count": len(
        near_overlaps
    ),
    "near_threshold": NEAR_THRESHOLD,
    "source_dataset": "CARES-18K",
    "source_split": "test",
    "source_text_field": "base_prompt",
    "output": str(OUTPUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== A_TEST health finalized =====")
print("total rows:", len(final_rows))
print("levels:", levels)
print("generation models:", models)
print("methods:", methods)
print(
    "validation:",
    Counter(
        row["validation_status"]
        for row in final_rows
    ),
)
print("unique prompts:", len(set(hashes)))
print("exact training overlap:", len(exact_overlap_ids))
print("near training overlap:", len(near_overlaps))
print("output:", OUTPUT_PATH)
print("summary:", SUMMARY_PATH)
