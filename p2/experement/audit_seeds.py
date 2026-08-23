#!/usr/bin/env python3
"""
audit_seeds.py — Audit, validate, dedupe and stratify the CDG seed set.

Reads a directory of seed files (one object per file, a JSON array, JSONL, or
several concatenated objects per file — all handled), then reports schema
problems, duplicates, near-duplicates and the per-topic distribution, and emits
a single cleaned, stratified JSONL plus a machine-readable audit report.

Stdlib only (difflib for near-dup), so it runs on the cluster with no installs.

Usage:
    python audit_seeds.py --in ./seeds --out ./curated
    python audit_seeds.py --in ./seeds --out ./curated --dup-threshold 0.92
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REQUIRED_FIELDS = ("id", "topic", "prompt", "harmful_variant")
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """Lowercase + collapse whitespace, for comparison only (not stored)."""
    return _WS.sub(" ", text.strip().lower())


def load_records(path: Path) -> list[dict]:
    """Robust loader: single object, JSON array, JSONL, or concatenated objects."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Fast path: a single array or object.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    # Fall back: whitespace-separated / concatenated JSON objects (also covers JSONL).
    records, decoder, idx, n = [], json.JSONDecoder(), 0, len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name}: could not parse near offset {idx}: {e}") from e
        if isinstance(obj, dict):
            records.append(obj)
        idx = end
    return records


def validate(rec: dict, source: str) -> list[str]:
    """Return a list of problem strings for one record (empty == clean)."""
    problems = []
    for f in REQUIRED_FIELDS:
        if f not in rec:
            problems.append(f"missing field '{f}'")
        elif not isinstance(rec[f], str):
            problems.append(f"field '{f}' is not a string ({type(rec[f]).__name__})")
        elif not rec[f].strip():
            problems.append(f"field '{f}' is empty")
    # Mislabel smell-test: harmful_variant too close to the benign prompt.
    if all(f in rec and isinstance(rec[f], str) for f in ("prompt", "harmful_variant")):
        r = SequenceMatcher(None, norm(rec["prompt"]), norm(rec["harmful_variant"])).ratio()
        if r >= 0.85:
            problems.append(f"harmful_variant ~ prompt (ratio {r:.2f}) — possible mislabel")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit/clean/stratify CDG seed files.")
    ap.add_argument("--in", dest="indir", required=True, help="directory of seed files")
    ap.add_argument("--out", dest="outdir", required=True, help="output directory")
    ap.add_argument("--glob", default="*.json", help="file glob (default: *.json)")
    ap.add_argument("--dup-threshold", type=float, default=0.92,
                    help="harmful_variant near-dup similarity cutoff (default 0.92)")
    args = ap.parse_args()

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(indir.glob(args.glob))
    if not files:
        print(f"No files matching {args.glob} in {indir}", file=sys.stderr)
        return 1

    records: list[dict] = []        # all parsed records, with provenance attached
    parse_errors: list[str] = []
    for fp in files:
        try:
            for rec in load_records(fp):
                rec.setdefault("_source", fp.name)
                records.append(rec)
        except ValueError as e:
            parse_errors.append(str(e))

    # --- Validation ---------------------------------------------------------
    invalid: list[dict] = []
    valid: list[dict] = []
    for rec in records:
        probs = validate(rec, rec.get("_source", "?"))
        if probs:
            invalid.append({"id": rec.get("id", "?"), "source": rec["_source"], "problems": probs})
        else:
            valid.append(rec)

    # --- Duplicate ids ------------------------------------------------------
    id_counts = Counter(r["id"] for r in valid)
    dup_ids = {i: c for i, c in id_counts.items() if c > 1}

    # --- Exact + near-duplicate harmful_variant -----------------------------
    by_norm: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_norm[norm(r["harmful_variant"])].append(r)
    exact_dups = {k: [r["id"] for r in v] for k, v in by_norm.items() if len(v) > 1}

    # Keep first occurrence of each exact-duplicate harmful_variant.
    seen_norm, deduped = set(), []
    for r in valid:
        key = norm(r["harmful_variant"])
        if key in seen_norm:
            continue
        seen_norm.add(key)
        deduped.append(r)

    # Near-dup pairs (O(n^2), fine for ~100). Compare only distinct items.
    near_pairs = []
    for i in range(len(deduped)):
        a = deduped[i]
        for j in range(i + 1, len(deduped)):
            b = deduped[j]
            ratio = SequenceMatcher(None, norm(a["harmful_variant"]),
                                    norm(b["harmful_variant"])).ratio()
            if ratio >= args.dup_threshold:
                near_pairs.append({"a": a["id"], "b": b["id"], "ratio": round(ratio, 3)})

    # --- Stratification -----------------------------------------------------
    topic_counts = Counter(r["topic"] for r in deduped)
    singletons = [t for t, c in topic_counts.items() if c == 1]

    # --- Write cleaned, stratified JSONL (sorted by topic then id) ----------
    clean_path = outdir / "seeds_curated.jsonl"
    with clean_path.open("w", encoding="utf-8") as f:
        for r in sorted(deduped, key=lambda x: (x["topic"], x["id"])):
            out = {k: r[k] for k in REQUIRED_FIELDS}  # drop _source, keep schema parity
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    report = {
        "files_scanned": len(files),
        "records_parsed": len(records),
        "valid": len(valid),
        "invalid": len(invalid),
        "after_dedup": len(deduped),
        "parse_errors": parse_errors,
        "invalid_records": invalid,
        "duplicate_ids": dup_ids,
        "exact_dup_harmful_variant": exact_dups,
        "near_dup_pairs": sorted(near_pairs, key=lambda p: -p["ratio"]),
        "topic_counts": dict(topic_counts.most_common()),
        "singleton_topics": singletons,
        "output_jsonl": str(clean_path),
    }
    (outdir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Console summary ----------------------------------------------------
    print(f"Scanned {len(files)} files, parsed {len(records)} records.")
    print(f"  valid: {len(valid)}   invalid: {len(invalid)}   after dedup: {len(deduped)}")
    if parse_errors:
        print(f"  PARSE ERRORS ({len(parse_errors)}):")
        for e in parse_errors:
            print(f"    - {e}")
    if invalid:
        print(f"  INVALID ({len(invalid)}):")
        for v in invalid:
            print(f"    - {v['id']} [{v['source']}]: {'; '.join(v['problems'])}")
    if dup_ids:
        print(f"  DUPLICATE IDs: {dup_ids}")
    if exact_dups:
        print(f"  EXACT-DUP harmful_variant clusters: {len(exact_dups)}")
        for ids in exact_dups.values():
            print(f"    - {ids}")
    if near_pairs:
        print(f"  NEAR-DUP pairs (>= {args.dup_threshold}): {len(near_pairs)}")
        for p in sorted(near_pairs, key=lambda x: -x['ratio'])[:20]:
            print(f"    - {p['a']} ~ {p['b']}  ({p['ratio']})")
    print("  TOPIC distribution:")
    for t, c in topic_counts.most_common():
        flag = "  <-- singleton" if c == 1 else ""
        print(f"    {c:>3}  {t}{flag}")
    if singletons:
        print(f"  NOTE: {len(singletons)} singleton topic(s) — too small for within-stratum lift stats.")
    print(f"\nWrote {clean_path}")
    print(f"Wrote {outdir / 'audit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())