#!/usr/bin/env python

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process


ROOT = Path("/path/to/rrae_steering_work_v2")

SALAD_PATH = (
    ROOT
    / "external_datasets/TEST_HELDOUT_V1/"
      "SALAD_OFFICIAL_HF/salad_official_all.jsonl"
)

BLOCKLIST_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/training_blocklist/"
      "RRAE_V2_TRAIN_BLOCKLIST_N1996.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/group_A_harmful_clean/"
      "A_TEST_CANDIDATE_harmful_clean_500.jsonl"
)

SUMMARY_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/source_audits/"
      "A_TEST_CANDIDATE_summary.json"
)

REJECTIONS_PATH = (
    ROOT
    / "data/TEST_HELDOUT_V1/overlap_reports/"
      "A_TEST_SALAD_overlap_rejections.jsonl"
)

HEALTH_TARGET = 400
NONHEALTH_TARGET = 100
NEAR_THRESHOLD = 95.0

HEALTH_CATEGORIES = {
    "O20: Medical Advice",
    "O39: Biological and Chemical Harms",
}

# O19 is mixed. Only include prompts with clear health / unsafe-use framing.
O19_HEALTH_PATTERN = re.compile(
    r"\b("
    r"overdose|dose|dosage|fatal|kill me|safe to mix|mix .* alcohol|"
    r"prescription|medication|medicine|opioid|fentanyl|xanax|"
    r"benzodiazepine|benzos|oxycontin|codeine|heroin|stimulant|"
    r"sedative|painkiller|withdrawal|respiratory depression|"
    r"heart attack|addiction|intoxication|inject|snort|inhale|"
    r"aerosol|cough syrup|get high|large dose|drug interaction|"
    r"signs of an overdose|toxic|poison"
    r")\b",
    re.IGNORECASE,
)

MEDICAL_TERMS = re.compile(
    r"\b("
    r"medical|medicine|medication|doctor|patient|disease|infection|"
    r"cancer|drug|opioid|fentanyl|overdose|prescription|hospital|"
    r"health|diagnosis|treatment|symptom|virus|bacteria|chemical|"
    r"biological|poison|toxin"
    r")\b",
    re.IGNORECASE,
)

SOURCE_PRIORITY = {
    "HH-red-team": 0,
    "HH-harmless": 1,
    "Do-Not-Answer": 2,
    "ToxicChat": 3,
    "Do Anything Now": 4,
    "GPTFuzzer": 5,
    "Multilingual": 6,
    "GPT-Gen": 7,
}


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"<\|mdm_mask\|>", " ", text)
    text = re.sub(r"<mask:\d+>", " ", text)
    text = re.sub(r"<<\s*tpl\s*>>|<</\s*tpl\s*>>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def digest(text: str) -> str:
    return hashlib.sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()


def stable_key(row: dict) -> str:
    material = (
        f"{row.get('source')}|"
        f"{row.get('qid')}|"
        f"{row.get('question')}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def classify_domain(row: dict):
    category = str(row.get("3-category", ""))
    question = str(row.get("question", ""))

    if category in HEALTH_CATEGORIES:
        return "health_related"

    if (
        category
        == "O19: Illegal Drugs and Regulated/Controlled Substances"
        and O19_HEALTH_PATTERN.search(question)
    ):
        return "health_related"

    # Non-health controls must not look medical.
    if (
        category not in {
            "O19: Illegal Drugs and Regulated/Controlled Substances",
            "O20: Medical Advice",
            "O39: Biological and Chemical Harms",
            "O54: Drug-related Crimes",
        }
        and not MEDICAL_TERMS.search(question)
    ):
        return "non_health_control"

    return None


def round_robin_select(rows, target):
    """
    Prefer source diversity rather than taking all rows
    from one source consecutively.
    """
    groups = defaultdict(list)

    for row in rows:
        groups[row["source"]].append(row)

    for source in groups:
        groups[source].sort(key=stable_key)

    ordered_sources = sorted(
        groups,
        key=lambda source: (
            SOURCE_PRIORITY.get(source, 100),
            source,
        ),
    )

    selected = []
    offsets = {source: 0 for source in ordered_sources}

    while len(selected) < target:
        added = False

        for source in ordered_sources:
            index = offsets[source]

            if index >= len(groups[source]):
                continue

            selected.append(groups[source][index])
            offsets[source] += 1
            added = True

            if len(selected) >= target:
                break

        if not added:
            break

    return selected


def main():
    for path in [SALAD_PATH, BLOCKLIST_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REJECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    salad_rows = load_jsonl(SALAD_PATH)
    block_rows = load_jsonl(BLOCKLIST_PATH)

    blocked_hashes = {
        row["sha256"]
        for row in block_rows
    }

    blocked_texts = [
        row["normalized_text"]
        for row in block_rows
    ]

    accepted_by_domain = defaultdict(list)
    rejections = []
    seen_candidate_hashes = set()

    stats = Counter()

    for row in salad_rows:
        stats["input_rows"] += 1

        source = str(row.get("source", "")).strip()
        question = str(row.get("question", "")).strip()

        if source.lower() == "advbench":
            stats["rejected_source_advbench"] += 1
            continue

        if not question:
            stats["rejected_empty"] += 1
            continue

        domain = classify_domain(row)

        if domain is None:
            stats["rejected_domain_not_selected"] += 1
            continue

        normalized = normalize(question)
        question_hash = digest(question)

        if question_hash in seen_candidate_hashes:
            stats["rejected_internal_duplicate"] += 1
            continue

        if question_hash in blocked_hashes:
            stats["rejected_exact_training_overlap"] += 1

            rejections.append({
                "reason": "exact_training_overlap",
                "qid": row.get("qid"),
                "source": source,
                "question": question,
            })
            continue

        near_match = None

        # Avoid unstable fuzzy decisions for very short strings.
        if len(normalized) >= 35:
            near_match = process.extractOne(
                normalized,
                blocked_texts,
                scorer=fuzz.WRatio,
                score_cutoff=NEAR_THRESHOLD,
            )

        if near_match is not None:
            matched_text, score, matched_index = near_match

            stats["rejected_near_training_overlap"] += 1

            rejections.append({
                "reason": "near_training_overlap",
                "qid": row.get("qid"),
                "source": source,
                "question": question,
                "similarity": score,
                "matched_training_text": matched_text,
                "matched_training_metadata": block_rows[matched_index],
            })
            continue

        seen_candidate_hashes.add(question_hash)

        candidate = {
            "source_dataset": "SALAD-Bench",
            "source_config": "base_set",
            "source": source,
            "source_id": str(row.get("qid")),
            "bucket": "A",
            "pair_id": None,
            "label": "harmful",
            "domain": domain,
            "prompt": question,
            "clean_prompt": question,
            "category_level_1": row.get("1-category"),
            "category_level_2": row.get("2-category"),
            "category_level_3": row.get("3-category"),
            "hf_split": row.get("_hf_split"),
            "hf_index": row.get("_hf_index"),
            "normalized_sha256": question_hash,
            "selection_status": "candidate_pending_manual_audit",
        }

        accepted_by_domain[domain].append(candidate)
        stats[f"eligible_{domain}"] += 1

    health_selected = round_robin_select(
        accepted_by_domain["health_related"],
        HEALTH_TARGET,
    )

    nonhealth_selected = round_robin_select(
        accepted_by_domain["non_health_control"],
        NONHEALTH_TARGET,
    )

    if len(health_selected) < HEALTH_TARGET:
        raise RuntimeError(
            f"Only {len(health_selected)} eligible health examples; "
            f"need {HEALTH_TARGET}."
        )

    if len(nonhealth_selected) < NONHEALTH_TARGET:
        raise RuntimeError(
            f"Only {len(nonhealth_selected)} eligible non-health examples; "
            f"need {NONHEALTH_TARGET}."
        )

    final_rows = health_selected + nonhealth_selected

    for index, row in enumerate(final_rows):
        row["id"] = f"A_TEST_{index:04d}"
        row["pair_id"] = f"TEST_PAIR_{index:04d}"

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with REJECTIONS_PATH.open("w", encoding="utf-8") as f:
        for row in rejections:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "status": "candidate_pending_manual_audit",
        "input_dataset": str(SALAD_PATH),
        "training_blocklist": str(BLOCKLIST_PATH),
        "near_duplicate_threshold": NEAR_THRESHOLD,
        "target_counts": {
            "health_related": HEALTH_TARGET,
            "non_health_control": NONHEALTH_TARGET,
            "total": HEALTH_TARGET + NONHEALTH_TARGET,
        },
        "selected_counts": {
            "health_related": len(health_selected),
            "non_health_control": len(nonhealth_selected),
            "total": len(final_rows),
        },
        "selected_sources": dict(
            Counter(row["source"] for row in final_rows)
        ),
        "selected_sources_by_domain": {
            domain: dict(
                Counter(
                    row["source"]
                    for row in final_rows
                    if row["domain"] == domain
                )
            )
            for domain in [
                "health_related",
                "non_health_control",
            ]
        },
        "selected_categories_level_3": dict(
            Counter(
                row["category_level_3"]
                for row in final_rows
            )
        ),
        "pipeline_stats": dict(stats),
        "output": str(OUTPUT_PATH),
        "rejections_output": str(REJECTIONS_PATH),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("===== A_TEST candidate built =====")
    print("selected total:", len(final_rows))
    print(
        "domains:",
        Counter(row["domain"] for row in final_rows),
    )
    print(
        "sources:",
        Counter(row["source"] for row in final_rows),
    )
    print(
        "level-3 categories:",
        Counter(
            row["category_level_3"]
            for row in final_rows
        ),
    )
    print("output:", OUTPUT_PATH)
    print("summary:", SUMMARY_PATH)
    print("rejections:", REJECTIONS_PATH)


if __name__ == "__main__":
    main()
