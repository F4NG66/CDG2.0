#!/usr/bin/env python3
"""build_attack2_prompts.py -- apply the EXISTING attack2 (Q&A/dialogue) scaffold
to the fresh 100 harmful behaviors in attack2/source_A/, emitting a runner-ready
prompt-root with a single B-group (harmful_injected) subfolder.

We REUSE the transform verbatim: `build_b2_user_content(idx)` is imported from
crossattack/build/build_attack2.py (NOT copied, NOT edited). We do NOT run that
file's main() -- that one reads the DIJA prompt-roots and asserts against them.
Here the source is attack2/source_A/cases.json instead, and we assert the behavior
string is carried through byte-identical from source -> emitted case.

Output (single B subfolder so the runner's group map -> group "B"):
    attack2/prompts/attack2_b/B2_harmful_injected/cases.jsonl

Run (CPU, no model):
    /home/ore99/env_llada/bin/python3 attack2/build_attack2_prompts.py
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../serverFiles/attack2
ROOT = os.path.dirname(HERE)                               # .../serverFiles
BUILDER_DIR = os.path.join(ROOT, "crossattack", "build")   # holds build_attack2.py

# Import the SAME transform the crossattack B2 family was built with.
sys.path.insert(0, BUILDER_DIR)
from build_attack2 import build_b2_user_content            # noqa: E402  (reused, unedited)

SRC = os.path.join(HERE, "source_A", "cases.json")
OUT = os.path.join(HERE, "prompts", "attack2_b", "B2_harmful_injected", "cases.jsonl")


def main():
    with open(SRC) as f:
        src = json.load(f)
    assert isinstance(src, list) and len(src) == 100, \
        f"expected 100 source behaviors, got {type(src)} / {len(src)}"

    rows = []
    for idx, case in enumerate(src):
        beh = case["behavior"]
        src_id = case["id"]
        assert case.get("user_content", "{behavior}") == "{behavior}", \
            f"source {src_id} is not clean/no-template form"

        user_content = build_b2_user_content(idx)          # <-- reused transform
        # the runner fills this placeholder with `behavior`; it MUST survive intact
        assert "{behavior}" in user_content, f"placeholder lost at idx {idx}"

        rows.append({
            "id": f"B{idx:03d}",                # group letter B -> runner group "B"
            "group": "B2",
            "behavior": beh,                    # carried byte-identical (asserted below)
            "user_content": user_content,       # {behavior} + Q&A scaffold after it
            "content_type": "harmful",
            "has_template": True,
            "attack_method": "attack2",
            "source_id": src_id,                # provenance back to source_A
            "source_index": idx,
        })
        # identity control: behavior string must equal the source, byte for byte
        assert rows[-1]["behavior"] == src[idx]["behavior"], \
            f"behavior mutated at idx {idx}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(rows)} B2 cases -> {OUT}")
    print("[ok] all 100 behavior strings carried byte-identical from source_A (asserted)")
    print(f"[ok] prompt-root for the runner: {os.path.dirname(os.path.dirname(OUT))}")


if __name__ == "__main__":
    main()
