#!/usr/bin/env python

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


OUT_ROOT = Path(
    "/path/to/rrae_steering_work_v2/"
    "external_datasets/TEST_HELDOUT_V1/SALAD_OFFICIAL_HF"
)

OUT_JSONL = OUT_ROOT / "salad_official_all.jsonl"
OUT_SCHEMA = OUT_ROOT / "salad_schema_summary.json"


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading official OpenSafetyLab/Salad-Data...")
    dataset_dict = load_dataset(
        "OpenSafetyLab/Salad-Data",
        "base_set",
        trust_remote_code=False,
    )

    print("splits:", list(dataset_dict.keys()))

    all_rows = []
    split_sizes = {}

    for split_name, dataset in dataset_dict.items():
        split_sizes[split_name] = len(dataset)

        for index, row in enumerate(dataset):
            record = dict(row)
            record["_hf_split"] = split_name
            record["_hf_index"] = index
            all_rows.append(record)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    keys = Counter()

    for row in all_rows:
        keys.update(row.keys())

    summary = {
        "dataset": "OpenSafetyLab/Salad-Data",
        "split_sizes": split_sizes,
        "total_rows": len(all_rows),
        "field_counts": dict(keys),
        "first_record_keys": sorted(all_rows[0].keys()) if all_rows else [],
    }

    OUT_SCHEMA.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("total rows:", len(all_rows))
    print("split sizes:", split_sizes)
    print("fields:", sorted(keys))
    print("jsonl:", OUT_JSONL)
    print("schema:", OUT_SCHEMA)


if __name__ == "__main__":
    main()
