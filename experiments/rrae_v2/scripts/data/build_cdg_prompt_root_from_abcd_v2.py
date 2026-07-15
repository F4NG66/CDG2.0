#!/usr/bin/env python

import argparse
import json
import re
from collections import Counter
from pathlib import Path


MASK_SPAN_RE = re.compile(r"<mask:(\d+)>")


VARIANT_DIRS = {
    "A": "A_harmful_clean",
    "B": "B_harmful_injected",
    "C": "C_benign_injected",
    "D": "D_benign_clean",
}


def expand_mask_spans(text: str, mask_token: str) -> str:
    """
    Convert:
        <mask:5>
    into:
        <|mdm_mask|><|mdm_mask|>... repeated 5 times.
    """

    def repl(match: re.Match) -> str:
        count = int(match.group(1))
        if count < 0:
            raise ValueError(f"Negative mask count: {count}")
        return mask_token * count

    return MASK_SPAN_RE.sub(repl, text)


def wrap_template(text: str) -> str:
    """
    The CDG runner recognizes the template region using <<TPL>>...<</TPL>>.
    Avoid wrapping twice if the DIJA prompt already contains markers.
    """
    if "<<TPL>>" in text and "<</TPL>>" in text:
        return text
    return f"<<TPL>>{text}<</TPL>>"


def convert_row(row: dict, mask_token: str, original_split: str) -> dict:
    bucket = str(row["bucket"]).upper()
    case_id = str(row["id"])
    prompt = str(row["prompt"])
    clean_prompt = str(row.get("clean_prompt") or prompt)
    domain = str(row.get("domain", "unknown"))
    label = str(row.get("label", "")).lower()
    variant = str(row.get("variant", "")).lower()

    if bucket not in VARIANT_DIRS:
        raise ValueError(f"Unexpected bucket: {bucket}")

    is_injected = bucket in {"B", "C"}
    is_harmful = bucket in {"A", "B"}

    behavior = clean_prompt

    if is_injected:
        user_content = expand_mask_spans(prompt, mask_token)
        user_content = wrap_template(user_content)
        attack_method = "DIJA"
    else:
        user_content = "{behavior}"
        attack_method = "none"

    content_type = "harmful" if is_harmful else "neutral"

    result = {
        "id": case_id,
        "source_case_id": (
            row.get("pair_id")
            or row.get("paired_A_id")
            or row.get("paired_D_id")
            or case_id
        ),
        "behavior": behavior,
        "user_content": user_content,
        "content_type": content_type,
        "has_template": is_injected,
        "original_split": original_split,
        "category": domain,
        "risk_level": "unknown",
        "bucket": bucket,
        "label_v2": label,
        "variant_v2": variant,
        "domain": domain,
        "source_dataset": row.get("source_dataset"),
        "original_v2_id": case_id,
        "clean_prompt": clean_prompt,
    }

    if is_injected:
        result["attack_method"] = attack_method

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--mask-token", default="<|mdm_mask|>")
    ap.add_argument(
        "--original-split",
        default="phase3_v2_in_sample",
    )
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    output_root = Path(args.output_root)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rows = []
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)

    counts = Counter(str(row["bucket"]).upper() for row in rows)

    expected = {"A": 500, "B": 500, "C": 500, "D": 500}
    if dict(counts) != expected:
        raise RuntimeError(
            f"Unexpected bucket counts: {dict(counts)}; expected {expected}"
        )

    converted = {bucket: [] for bucket in VARIANT_DIRS}

    for row in rows:
        bucket = str(row["bucket"]).upper()
        converted[bucket].append(convert_row(row, args.mask_token, args.original_split))

    for bucket, dirname in VARIANT_DIRS.items():
        out_dir = output_root / dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        output_path = out_dir / "cases.json"
        output_path.write_text(
            json.dumps(converted[bucket], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"{bucket}: wrote {len(converted[bucket])} cases -> {output_path}"
        )

    injected_without_masks = []

    for bucket in ["B", "C"]:
        for row in converted[bucket]:
            if args.mask_token not in row["user_content"]:
                injected_without_masks.append(row["id"])

    if injected_without_masks:
        raise RuntimeError(
            "Injected cases without expanded masks: "
            + ", ".join(injected_without_masks[:20])
        )

    print()
    print("CDG prompt-root build passed.")
    print("output_root:", output_root)
    print("counts:", dict(counts))


if __name__ == "__main__":
    main()
