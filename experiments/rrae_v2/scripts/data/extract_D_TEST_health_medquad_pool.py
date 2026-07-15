#!/usr/bin/env python

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from rapidfuzz.fuzz import ratio


ROOT = Path("/path/to/rrae_steering_work_v2")

MEDQUAD_ROOT = (
    ROOT
    / "external_datasets/TEST_HELDOUT_V1/MedQuAD"
)

BLOCKLIST_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/training_blocklist/"
      "RRAE_V2_TRAIN_BLOCKLIST_N1996.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_D_benign_clean/"
      "D_TEST_HEALTH_MEDQUAD_FULL_POOL.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "D_TEST_HEALTH_MEDQUAD_FULL_POOL_summary.json"
)


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    return text


def clean_question(text):
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\?", "?", text)
    text = re.sub(r"\?+\s*\?+", "?", text)
    return text


def load_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def get_first_text(root, tag):
    elem = root.find(f".//{tag}")

    if elem is None:
        return None

    text = (elem.text or "").strip()
    return text or None


xml_files = sorted(MEDQUAD_ROOT.rglob("*.xml"))

if not xml_files:
    raise RuntimeError("No MedQuAD XML files found")

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

seen_questions = set()
pool = []

parse_errors = 0
empty_questions = 0
short_questions = 0
exact_overlap_removed = 0

for xml_path in xml_files:
    try:
        tree = ET.parse(xml_path)
        root_elem = tree.getroot()
    except Exception:
        parse_errors += 1
        continue

    focus = get_first_text(root_elem, "Focus")
    category = get_first_text(root_elem, "Category")

    relative_path = xml_path.relative_to(MEDQUAD_ROOT)
    source_folder = relative_path.parts[0]

    for question_elem in root_elem.findall(".//Question"):
        question = clean_question(question_elem.text)

        if not question:
            empty_questions += 1
            continue

        # Remove fragments or unusably short prompts.
        if len(question.split()) < 4:
            short_questions += 1
            continue

        question_norm = normalize(question)

        if question_norm in seen_questions:
            continue

        seen_questions.add(question_norm)

        if question_norm in blocked_norm:
            exact_overlap_removed += 1
            continue

        pool.append({
            "temporary_id": f"D_TEST_HEALTH_POOL_{len(pool):05d}",
            "bucket": "D",
            "label": "benign",
            "domain": "health_related",
            "source_dataset": "MedQuAD",
            "source_file": str(relative_path),
            "source_folder": source_folder,
            "focus": focus,
            "category": category,
            "prompt": question,
            "validation_status": "pending_selection",
        })

if not pool:
    raise RuntimeError("No eligible MedQuAD questions extracted")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in pool:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

category_counts = Counter(
    row["category"] or "UNKNOWN"
    for row in pool
)

folder_counts = Counter(
    row["source_folder"]
    for row in pool
)

summary = {
    "xml_files": len(xml_files),
    "parse_errors": parse_errors,
    "eligible_unique_questions": len(pool),
    "empty_questions_removed": empty_questions,
    "short_questions_removed": short_questions,
    "exact_training_overlap_removed": exact_overlap_removed,
    "category_counts": dict(category_counts),
    "source_folder_counts": dict(folder_counts),
    "output": str(OUTPUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== MedQuAD health pool =====")
print("XML files:", len(xml_files))
print("parse errors:", parse_errors)
print("eligible unique questions:", len(pool))
print("empty removed:", empty_questions)
print("short removed:", short_questions)
print("exact training overlap removed:", exact_overlap_removed)

print("\n===== CATEGORY COUNTS =====")
for category, count in category_counts.most_common():
    print(f"{category}: {count}")

print("\n===== SOURCE FOLDER COUNTS =====")
for folder, count in folder_counts.most_common():
    print(f"{folder}: {count}")

print("\n===== SAMPLE QUESTIONS =====")
for row in pool[:20]:
    print(
        f'{row["category"]} | {row["focus"]} | '
        f'{row["prompt"]}'
    )

print("\noutput:", OUTPUT_PATH)
print("summary:", SUMMARY_PATH)
