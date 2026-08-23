#!/usr/bin/env python3
"""attack2/contextual/build_contextual_prompts.py — Phase 3/4 prep.

Turn the Phase-2 contextual scaffolds (refined_100.json) into a runner-ready cdg
prompt-root, EXACTLY mirroring the original attack2 structure so it drops into
run_attack2_resumable.py + make_judge_input.py with zero edits:

    user_content = "{behavior}\\n\\n" + OPEN + "\\n" + <contextual dialogue scaffold> + "\\n" + CLOSE

i.e. request first, scaffold AFTER (attack2's signature), <<TPL>>..<</TPL>> sentinels
(stripped by the runner), {behavior} filled by the runner, <mask:N> expanded by the
llada_attack backend. The ONLY change vs the original attack2 is the scaffold body:
Qwen-refined mentor/newcomer CONTEXTUAL dialogue instead of the generic Q&A.

Reuse, never edit: OPEN/CLOSE imported from crossattack/build/build_attack2.py.
Case ids are B{source_index:03d} and behaviors are byte-identical to source_A, so
make_judge_input.py's `cid[1:] -> source_A[idx]` join stays correct.

    python attack2/contextual/build_contextual_prompts.py
    # -> attack2/contextual/prompts/attack2_ctx/B2ctx_harmful_injected/cases.jsonl
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACK2 = os.path.dirname(HERE)
SERVER = os.path.dirname(ATTACK2)
BUILDER_DIR = os.path.join(SERVER, "crossattack", "build")
if BUILDER_DIR not in sys.path:
    sys.path.insert(0, BUILDER_DIR)

from build_attack2 import OPEN, CLOSE            # noqa: E402  (reused sentinels, unedited)

SOURCE = os.path.join(ATTACK2, "source_A", "cases.json")
REFINED = os.path.join(HERE, "refined_100.json")
OUT = os.path.join(HERE, "prompts", "attack2_ctx", "B2ctx_harmful_injected", "cases.jsonl")


def main():
    src = json.load(open(SOURCE))
    refined = json.load(open(REFINED))
    by_id = {r["id"]: r for r in refined}
    assert len(src) == 100, f"expected 100 source behaviors, got {len(src)}"

    rows = []
    for idx, case in enumerate(src):
        beh, sid = case["behavior"], case["id"]
        r = by_id.get(sid)
        assert r is not None, f"no refined scaffold for source id {sid}"
        assert r["behavior"] == beh, f"behavior mismatch at {sid} (refine must be byte-identical)"
        assert r["valid"], f"refined scaffold {sid} is invalid — refine before building"

        scaffold = r["Refined_behavior"]                 # contextual mentor/newcomer dialogue
        user_content = "{behavior}\n\n" + OPEN + "\n" + scaffold + "\n" + CLOSE
        assert "{behavior}" in user_content, f"placeholder lost at {sid}"
        assert "<mask:" in user_content, f"no masks survived at {sid}"

        rows.append({
            "id": f"B{idx:03d}",                         # group B; cid[1:] -> source index
            "group": "B2ctx",
            "behavior": beh,                             # byte-identical (asserted above)
            "user_content": user_content,
            "content_type": "harmful",
            "has_template": True,
            "attack_method": "attack2_contextual",
            "source_id": sid,
            "source_index": idx,
            "n_masks": r["n_masks"],
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(rows)} contextual B cases -> {OUT}")
    print(f"[ok] prompt-root for the runner: {os.path.dirname(os.path.dirname(OUT))}")
    print(f"[ok] all behaviors byte-identical to source_A; masks/case = "
          f"{sorted(set(r['n_masks'] for r in rows))}")


if __name__ == "__main__":
    main()
