#!/usr/bin/env python3
"""Verify the canonical ABCD v2.2 dataset. Read-only; never rewrites data.

Usage:
    python verify_dataset.py [/path/to/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl]
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

NAME = "ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl"
DEFAULT_PATH = Path(__file__).resolve().parent / NAME
EXPECTED_SHA256 = "4392afca6d8f6b788a848d0417c92807ce01aeb077a67d6424bf12d69138eaaf"
EXPECTED_ROWS = 2000
EXPECTED_SCHEMA = {
    "id", "bucket", "label", "variant", "injection_type", "domain", "prompt",
    "paired_A_id", "paired_D_id", "clean_prompt", "source_dataset",
    "needs_manual_audit",
}
BUCKETS = ("A", "B", "C", "D")
EXPECTED_DOMAINS = {"health_related": 1600, "non_health_control": 400}


def verify(path):
    errors = []
    if not path.is_file():
        return [f"file not found: {path}"]

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        errors.append(f"sha256 mismatch: got {sha}, expected {EXPECTED_SHA256}")

    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON ({e})")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {lineno}: not a JSON object")
                continue
            rows.append(row)

    if len(rows) != EXPECTED_ROWS:
        errors.append(f"row count {len(rows)} != {EXPECTED_ROWS}")

    bad_schema = [r.get("id") for r in rows if set(r) != EXPECTED_SCHEMA]
    if bad_schema:
        errors.append(f"{len(bad_schema)} rows with wrong schema, e.g. {bad_schema[:3]}")

    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate row ids")

    buckets = Counter(r.get("bucket") for r in rows)
    for b in BUCKETS:
        if buckets.get(b, 0) != 500:
            errors.append(f"bucket {b} = {buckets.get(b, 0)}, expected 500")
    if set(buckets) - set(BUCKETS):
        errors.append(f"unexpected buckets: {sorted(map(str, set(buckets) - set(BUCKETS)))}")

    domains = Counter(r.get("domain") for r in rows)
    if dict(domains) != EXPECTED_DOMAINS:
        errors.append(f"domain counts {dict(domains)} != {EXPECTED_DOMAINS}")

    if any(r.get("needs_manual_audit") is not False for r in rows):
        errors.append("needs_manual_audit is not false for all rows")

    groups = defaultdict(list)
    for r in rows:
        rid = str(r.get("id", ""))
        bucket, _, num = rid.partition("_")
        if bucket != r.get("bucket"):
            errors.append(f"{rid}: id prefix does not match bucket {r.get('bucket')}")
        if r.get("paired_A_id") != f"A_{num}" or r.get("paired_D_id") != f"D_{num}":
            errors.append(f"{rid}: paired_A_id/paired_D_id inconsistent")
        groups[num].append(r.get("bucket"))
    if len(groups) != 500:
        errors.append(f"{len(groups)} groups, expected 500")
    incomplete = [g for g, bs in groups.items() if sorted(bs) != list(BUCKETS)]
    if incomplete:
        errors.append(f"{len(incomplete)} groups without exactly A/B/C/D, e.g. {incomplete[:3]}")

    return errors


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    errors = verify(path)
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        print("DATASET_VERIFICATION=FAIL")
        return 1
    print(f"file: {path}")
    print(f"sha256: {EXPECTED_SHA256}")
    print("rows=2000 A=500 B=500 C=500 D=500 health=1600 non_health=400 groups=500")
    print("DATASET_VERIFICATION=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
