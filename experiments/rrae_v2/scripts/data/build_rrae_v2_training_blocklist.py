#!/usr/bin/env python

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


INPUT = Path(
    "/path/to/rrae_steering_work_v2/"
    "data/abcd/ABCD_clean_v2_DIJA_Qwen_A_B_C_D.jsonl"
)

OUTPUT = Path(
    "/path/to/rrae_steering_work_v2/"
    "data/phase3_heldout/rrae_v2_training_blocklist.jsonl"
)


def strip_injection_markup(text: str) -> str:
    text = str(text or "")

    text = re.sub(
        r"<\|mdm_mask\|>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<mask:\d+>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<<\s*tpl\s*>>|<</\s*tpl\s*>>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def normalize(text: str) -> str:
    text = strip_injection_markup(text)
    text = text.lower()
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in INPUT.open(encoding="utf-8")
        if line.strip()
    ]

    seen = set()
    records = []

    for row in rows:
        for field in ["prompt", "clean_prompt"]:
            raw = row.get(field)

            if not isinstance(raw, str) or not raw.strip():
                continue

            normalized = normalize(raw)

            if not normalized:
                continue

            digest = sha256(normalized)

            if digest in seen:
                continue

            seen.add(digest)

            records.append({
                "sha256": digest,
                "normalized_text": normalized,
                "raw_text": raw,
                "source_bucket": row.get("bucket"),
                "source_id": row.get("id"),
                "source_field": field,
                "source_dataset": row.get("source_dataset"),
                "domain": row.get("domain"),
                "label": row.get("label"),
                "variant": row.get("variant"),
            })

    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("input rows:", len(rows))
    print("unique blocked texts:", len(records))
    print(
        "blocked by source:",
        Counter(r["source_dataset"] for r in records),
    )
    print(
        "blocked by bucket:",
        Counter(r["source_bucket"] for r in records),
    )
    print("output:", OUTPUT)


if __name__ == "__main__":
    main()
