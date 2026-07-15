#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

DIJA_REFINE_DIR = Path(
    "/path/to/DIJA/run_harmbench/refine_prompt"
)

sys.path.insert(0, str(DIJA_REFINE_DIR))

from utils import Refiner  # noqa: E402


def load_jsonl(path: Path):
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                )

    return rows


def atomic_write_json(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--prompt-template-path", required=True)
    parser.add_argument("--attack-prompt", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)

    args = parser.parse_args()

    attack_prompt = Path(args.attack_prompt)
    output_jsonl = Path(args.output_jsonl)
    output_json = Path(args.output_json)

    df = pd.read_csv(attack_prompt)

    required_columns = {
        "BehaviorID",
        "FunctionalCategory",
        "SemanticCategory",
        "Behavior",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["BehaviorID"].duplicated().any():
        duplicates = (
            df.loc[
                df["BehaviorID"].duplicated(keep=False),
                "BehaviorID",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate BehaviorID values found: {duplicates[:10]}"
        )

    completed_rows = load_jsonl(output_jsonl)

    completed_by_id = {}

    for row in completed_rows:
        behavior_id = str(row["BehaviorID"])

        if behavior_id in completed_by_id:
            raise RuntimeError(
                f"Duplicate completed BehaviorID: {behavior_id}"
            )

        completed_by_id[behavior_id] = row

    expected_ids = set(df["BehaviorID"].astype(str))

    unexpected_ids = (
        set(completed_by_id)
        - expected_ids
    )

    if unexpected_ids:
        raise RuntimeError(
            "Resume file contains unexpected BehaviorIDs: "
            f"{sorted(unexpected_ids)[:10]}"
        )

    print("input rows:", len(df))
    print("already completed:", len(completed_by_id))
    print("remaining:", len(df) - len(completed_by_id))

    refiner = Refiner(
        hf_model_path=args.hf_model_path,
        api_model_name="",
        prompt_template_path=args.prompt_template_path,
        attack_prompt=str(attack_prompt),
        output_json=str(output_json),
        base_url="",
        api_key="",
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("a", encoding="utf-8") as out_file:
        with torch.no_grad():
            for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="DIJA resume refinement",
            ):
                behavior_id = str(row["BehaviorID"])

                if behavior_id in completed_by_id:
                    continue

                vanilla_behavior = str(row["Behavior"])

                prompt = refiner.apply_prompt_template(
                    vanilla_behavior,
                    refiner.template_str,
                )

                response = refiner.qwen_generate(
                    prompt,
                    args.max_new_tokens,
                )

                result = {
                    "BehaviorID": behavior_id,
                    "FunctionalCategory": row[
                        "FunctionalCategory"
                    ],
                    "SemanticCategory": row[
                        "SemanticCategory"
                    ],
                    "Behavior": vanilla_behavior,
                    "Refined_behavior": response,
                }

                out_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                out_file.flush()
                os.fsync(out_file.fileno())

                completed_by_id[behavior_id] = result

    ordered_rows = []

    for behavior_id in df["BehaviorID"].astype(str):
        if behavior_id not in completed_by_id:
            raise RuntimeError(
                f"Missing completed BehaviorID: {behavior_id}"
            )

        ordered_rows.append(
            completed_by_id[behavior_id]
        )

    atomic_write_json(
        ordered_rows,
        output_json,
    )

    print("===== DIJA refinement complete =====")
    print("rows:", len(ordered_rows))
    print("JSONL:", output_jsonl)
    print("JSON:", output_json)


if __name__ == "__main__":
    main()
