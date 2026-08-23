#!/usr/bin/env python3
"""clock_attack/build_dija_scaffolds.py — strengthen the DIJA baseline.

The blank-canvas DIJA fallback (bare mask canvas, no context) makes LLaDA fill with
EOS -> empty baseline. To give the injection real dynamic range we attach LEADING
CONTEXT to each injected mask (the published DIJA "Step 1: [mask]  Step 2: [mask]"
worksheet skeleton). We author NO attack content: the generic step skeleton comes
from dija_refiner.stub_scaffold, and ablate_scaffold(..., "minimal") strips its debug
sentinels, leaving `{behavior}\nStep 1: <mask:14>\n...`. The masks stay empty; only
the model fills them. is_valid_scaffold() (imported) validates each one.

Harmful behaviors come from attack2/source_A/cases.json (byte-identical, never edited).

    python clock_attack/build_dija_scaffolds.py   # -> clock_attack/dija_scaffolds.jsonl
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
if EXP not in sys.path:
    sys.path.insert(0, EXP)

from dija_refiner import stub_scaffold, ablate_scaffold, is_valid_scaffold, parse_masks  # noqa: E402

HARMFUL_SEEDS = "/home/ore99/serverFiles/attack2/source_A/cases.json"
OUT = os.path.join(HERE, "dija_scaffolds.jsonl")


def main():
    obj = json.load(open(HARMFUL_SEEDS))
    rows, n_invalid = [], 0
    for c in obj:
        beh = c.get("behavior")
        if not beh:
            continue
        # reuse dija_refiner's generic step skeleton, then strip trailing sentinels
        scaffold = ablate_scaffold(stub_scaffold(beh), "minimal")
        ok, reason = is_valid_scaffold(scaffold)
        if not ok:
            n_invalid += 1
        assert beh in scaffold, "behavior must be carried through byte-identical"
        rows.append({
            "id": c["id"],
            "behavior": beh,                       # untouched
            "harmful_variant": scaffold,           # {behavior} + leading-context <mask:N>
            "n_masks": len(parse_masks(scaffold)),
            "scaffold_valid": ok,
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[scaffold] wrote {len(rows)} DIJA scaffolds -> {OUT}")
    print(f"[scaffold] masks/case = {rows[0]['n_masks']}; invalid = {n_invalid}")
    print("[scaffold] example (behavior redacted to placeholder):")
    ex = rows[0]["harmful_variant"].replace(rows[0]["behavior"], "{behavior}")
    print("    " + ex.replace("\n", "\n    "))


if __name__ == "__main__":
    main()
