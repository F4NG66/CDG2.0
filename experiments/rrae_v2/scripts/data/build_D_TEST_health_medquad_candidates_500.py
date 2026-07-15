#!/usr/bin/env python

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/path/to/rrae_steering_work_v2")

INPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_D_benign_clean/"
      "D_TEST_HEALTH_MEDQUAD_FULL_POOL.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_D_benign_clean/"
      "D_TEST_HEALTH_MEDQUAD_CANDIDATE_500.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "D_TEST_HEALTH_MEDQUAD_CANDIDATE_500_summary.json"
)

TARGET_TOTAL = 500

# Prevent the two largest MedQuAD folders from dominating.
SOURCE_QUOTAS = {
    "10_MPlus_ADAM_QA": 75,
    "11_MPlusDrugs_QA": 65,
    "2_GARD_QA": 55,
    "3_GHR_QA": 55,
    "6_NINDS_QA": 45,
    "12_MPlusHerbsSupplements_QA": 40,
    "4_MPlus_Health_Topics_QA": 40,
    "5_NIDDK_QA": 35,
    "1_CancerGov_QA": 30,
    "8_NHLBI_QA_XML": 25,
    "9_CDC_QA": 20,
    "7_SeniorHealth_QA": 15,
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


def question_type(text):
    q = normalize(text)

    # Classify using the request pattern near the beginning,
    # not words appearing inside the medical condition name.
    patterns = [
        ("symptoms", r"^what are the symptoms"),
        ("treatment", r"^(what are the treatments|how is .* treated|how to treat)"),
        ("causes", r"^what causes"),
        ("diagnosis", r"^(how to diagnose|how is .* diagnosed|what tests|what is .* test)"),
        ("prevention", r"^(how to prevent|can .* be prevented)"),
        ("risk", r"^(who is at risk|what are the risk factors)"),
        ("drug_usage", r"^(how should .* be used|how do i take|what is .* used for)"),
        ("side_effects", r"^(what are the side effects|what warnings)"),
        ("prognosis", r"^(what is the outlook|what is the prognosis)"),
        ("support", r"^(where to find support|where can i find support)"),
        ("definition", r"^(what is|what are|what is \(are\)|do you have information about)"),
    ]

    for label, pattern in patterns:
        if re.search(pattern, q):
            return label

    return "other"


def opening_type(text):
    q = normalize(text)

    if q.startswith("what "):
        return "what"
    if q.startswith("how "):
        return "how"
    if q.startswith("do "):
        return "do"
    if q.startswith("can "):
        return "can"
    if q.startswith("who "):
        return "who"
    if q.startswith("where "):
        return "where"
    if q.startswith("when "):
        return "when"
    if q.startswith("why "):
        return "why"

    return "other"


def is_low_quality(row):
    prompt = row["prompt"].strip()
    focus = (row.get("focus") or "").strip().lower()
    prompt_norm = normalize(prompt)

    # Obvious malformed guide-title templates.
    if focus.startswith("a guide to") and re.match(
        r"^what is \(are\) a guide to",
        prompt_norm,
    ):
        return True

    # Duplicate punctuation or unfinished fragments.
    if "??" in prompt:
        return True

    if prompt.endswith(":"):
        return True

    # Remove questions that are too short or too long.
    words = prompt.split()

    if len(words) < 7 or len(words) > 45:
        return True

    # Remove terse template-like questions that provide little context.
    if re.match(
        r"^(what to do for|what is \(are\))\s+[^?]{1,35}\?$",
        prompt,
        flags=re.IGNORECASE,
    ):
        return True

    return False


rows = load_jsonl(INPUT_PATH)

eligible = []

for row in rows:
    if is_low_quality(row):
        continue

    candidate = dict(row)
    candidate["question_type"] = question_type(row["prompt"])
    candidate["opening_type"] = opening_type(row["prompt"])
    eligible.append(candidate)

by_source = defaultdict(list)

for row in eligible:
    by_source[row["source_folder"]].append(row)

rng = random.Random(20260712)

selected = []
selected_prompts = set()
type_counts = Counter()
opening_counts = Counter()
source_counts = Counter()

# Maximum counts across the 500 candidates.
OPENING_CAPS = {
    "what": 240,
    "how": 90,
    "do": 75,
    "can": 40,
    "who": 25,
    "where": 15,
    "when": 10,
    "why": 10,
    "other": 50,
}

for source, quota in SOURCE_QUOTAS.items():
    remaining = by_source[source][:]
    rng.shuffle(remaining)

    source_selected = 0

    while source_selected < quota:
        valid = [
            row
            for row in remaining
            if normalize(row["prompt"]) not in selected_prompts
        ]

        if not valid:
            raise RuntimeError(
                f"Source {source}: selected {source_selected}, "
                f"expected {quota}"
            )

        # Recalculate balance after every selected example.
        row = min(
            valid,
            key=lambda candidate: (
                opening_counts[candidate["opening_type"]],
                type_counts[candidate["question_type"]],
                abs(len(candidate["prompt"].split()) - 14),
                candidate["temporary_id"],
            ),
        )

        norm = normalize(row["prompt"])

        out_row = {
            "temporary_id": (
                f"D_TEST_HEALTH_CAND_{len(selected):04d}"
            ),
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
            "validation_status": "pending_deepseek_audit",
        }

        selected.append(out_row)
        selected_prompts.add(norm)
        source_counts[source] += 1
        type_counts[row["question_type"]] += 1
        opening_counts[row["opening_type"]] += 1
        source_selected += 1

        remaining.remove(row)

if len(selected) != TARGET_TOTAL:
    raise RuntimeError(
        f"Selected {len(selected)} rows, expected {TARGET_TOTAL}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "total_rows": len(selected),
    "source_counts": dict(source_counts),
    "question_type_counts": dict(type_counts),
    "opening_counts": dict(opening_counts),
    "category_counts": dict(Counter(
        row.get("category") or "UNKNOWN"
        for row in selected
    )),
    "unique_prompts": len(selected_prompts),
    "output": str(OUTPUT_PATH),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== D_TEST health candidate pool =====")
print("total:", len(selected))
print("sources:", source_counts)
print("question types:", type_counts)
print("openings:", opening_counts)
print("categories:", Counter(
    row.get("category") or "UNKNOWN"
    for row in selected
))
print("unique prompts:", len(selected_prompts))

print("\n===== SAMPLE =====")
for row in selected[:20]:
    print(
        f'{row["source_folder"]} | '
        f'{row["question_type"]} | '
        f'{row["prompt"]}'
    )

print("\noutput:", OUTPUT_PATH)
print("summary:", SUMMARY_PATH)
