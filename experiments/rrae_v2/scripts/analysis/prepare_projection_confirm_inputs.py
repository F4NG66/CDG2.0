#!/usr/bin/env python

import json
from pathlib import Path
from collections import Counter


WORK_ROOT = Path("/path/to/rrae_steering_work_v2")

GEN_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/projection_removal_confirm"
)

OUT_ROOT = (
    WORK_ROOT
    / "data/phase3_judge_inputs/projection_removal_confirm"
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_layer(layer):
    path = (
        GEN_ROOT
        / f"L{layer}_r24_raw_projection_R0_N100"
        / f"results__custom_template_mask__rank24__L{layer}.jsonl"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return read_jsonl(path)


def main():
    rows11 = load_layer(11)
    rows16 = load_layer(16)

    baseline11 = [
        row for row in rows11
        if row.get("mode") == "baseline"
    ]

    baseline16 = [
        row for row in rows16
        if row.get("mode") == "baseline"
    ]

    map11 = {
        (row["group"], row["case_id"]): row.get("output", "")
        for row in baseline11
    }

    map16 = {
        (row["group"], row["case_id"]): row.get("output", "")
        for row in baseline16
    }

    if map11 != map16:
        raise RuntimeError(
            "L11 and L16 baseline outputs are not identical."
        )

    steered11 = [
        row for row in rows11
        if row.get("mode") == "steered"
    ]

    steered16 = [
        row for row in rows16
        if row.get("mode") == "steered"
    ]

    baseline_path = OUT_ROOT / "baseline_canonical_N200.jsonl"
    l11_path = OUT_ROOT / "L11_steered_N600.jsonl"
    l16_path = OUT_ROOT / "L16_steered_N600.jsonl"

    write_jsonl(baseline_path, baseline11)
    write_jsonl(l11_path, steered11)
    write_jsonl(l16_path, steered16)

    print("baseline:", baseline_path, len(baseline11))
    print("L11 steered:", l11_path, len(steered11))
    print("L16 steered:", l16_path, len(steered16))

    print(
        "baseline groups:",
        Counter(row["group"] for row in baseline11),
    )

    print(
        "L11:",
        Counter(
            (row["group"], float(row["alpha"]))
            for row in steered11
        ),
    )

    print(
        "L16:",
        Counter(
            (row["group"], float(row["alpha"]))
            for row in steered16
        ),
    )


if __name__ == "__main__":
    main()
