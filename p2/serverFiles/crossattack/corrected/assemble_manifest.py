#!/usr/bin/env python
"""crossattack/corrected/assemble_manifest.py

Build the attack2 CORRECTED transfer root manifest that load_records() will read:
  - clean A2 arm  : REUSED verbatim from the existing crossattack states
                    (rows copied from crossattack/outputs/manifest.jsonl, their
                    'path' already points at the existing A2 .pt) -- NOT regenerated
                    (the clean arm is a bare {behavior}, scaffold-independent).
  - injected B2   : the NEW attack2_qwen-scaffold states just captured
                    (rows from crossattack/corrected/outputs/manifest.jsonl).

Writes ONLY crossattack/corrected/transfer_attack2/manifest.jsonl (a manifest-only
dir; load_records loads each .pt via its 'path' field, resolved from repo root).
Verifies every referenced .pt exists and that A2/B2 base-ids line up 1:1.
"""
import json, os, collections

ROOT = "/home/ore99/serverFiles"
OLD_MAN = os.path.join(ROOT, "crossattack/outputs/manifest.jsonl")
NEW_MAN = os.path.join(ROOT, "crossattack/corrected/outputs/manifest.jsonl")
OUT_DIR = os.path.join(ROOT, "crossattack/corrected/transfer_attack2")
OUT = os.path.join(OUT_DIR, "manifest.jsonl")


def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    old = read(OLD_MAN)
    new = read(NEW_MAN)

    a2 = [r for r in old if r.get("variant") == "A2_harmful_clean"]
    # dedup B2 by case_id (keep last) in case capture was resumed
    b2_by = {}
    for r in new:
        if r.get("variant") == "B2_harmful_injected":
            b2_by[r["case_id"]] = r
    b2 = [b2_by[k] for k in sorted(b2_by)]

    assert len(a2) == 100, f"expected 100 A2 clean, got {len(a2)}"
    assert len(b2) == 100, f"expected 100 B2 injected, got {len(b2)}"

    # verify pairing + files on disk
    a_ids = {r["case_id"][1:] for r in a2}
    b_ids = {r["case_id"][1:] for r in b2}
    assert a_ids == b_ids, f"base-id mismatch: A-only={a_ids-b_ids} B-only={b_ids-a_ids}"
    missing = [r["path"] for r in (a2 + b2)
               if not os.path.exists(os.path.join(ROOT, r["path"]))]
    assert not missing, f"missing .pt files: {missing[:5]}"

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in a2 + b2:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gl = collections.Counter(r["variant"][:1] for r in a2 + b2)
    print(f"[assemble] wrote {len(a2)+len(b2)} rows -> {OUT}")
    print(f"[assemble] groups: {dict(gl)}  (A=clean reused, B=injected new)")
    print(f"[assemble] all {len(a2)+len(b2)} referenced .pt exist; A2<->B2 base ids 1:1")


if __name__ == "__main__":
    main()
