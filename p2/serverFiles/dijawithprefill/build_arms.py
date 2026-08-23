#!/usr/bin/env python
"""Build the DIJA+prefill position x dose arms for groups B and D.

Reads the EXISTING source cases (never modifies them):
    prompts/cdg_injection/B_harmful_injected/cases.json   (harmful, has scaffold)
    prompts/cdg_injection/D_neutral_clean/cases.json       (neutral, no scaffold)

Writes one prompt-root per arm under dijawithprefill/prompts/, each containing
B_harmful_injected/cases.jsonl and D_neutral_clean/cases.jsonl so the existing
runner's group mapping (first letter of folder) still resolves B->harmful,
D->neutral.  Every record carries: id, behavior, user_content, content_type,
has_template, attack_method, group, position, dose  (joinable later).

  * B: insert seeds into each case's OWN bespoke scaffold (anchors before every
       <mask:N>; primer once before the scaffold body).
  * D: transplant a SINGLE fixed generic Step-1/2/3 scaffold onto each neutral
       behavior, then seed it identically.

dose-0 (no-primer) arms for B are asserted byte-identical to the source prompts.

Run from the repo root:  python dijawithprefill/build_arms.py
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)   # for prefill_seeds
sys.path.insert(0, ROOT)   # for cdg.config

import prefill_seeds as ps  # noqa: E402

SRC = {
    "B": os.path.join(ROOT, "prompts/cdg_injection/B_harmful_injected/cases.json"),
    "D": os.path.join(ROOT, "prompts/cdg_injection/D_neutral_clean/cases.json"),
}
GROUP_FOLDER = {"B": "B_harmful_injected", "D": "D_neutral_clean"}
CONTENT_TYPE = {"B": "harmful", "D": "neutral"}
OUT_ROOT = os.path.join(HERE, "prompts")


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _base_uc(group: str, src_case: dict) -> str:
    """Base template (pre-seeding) for a case."""
    if group == "B":
        return src_case["user_content"]          # case's own bespoke scaffold
    return ps.D_SCAFFOLD                           # fixed transplant for all D


def build_record(group: str, src_case: dict, position: str, dose: int) -> dict:
    base = _base_uc(group, src_case)
    uc = ps.build_arm_user_content(base, position, dose)
    return {
        "id": src_case["id"],
        "behavior": src_case["behavior"],          # harmful/neutral text, verbatim
        "user_content": uc,                         # {behavior} kept as placeholder
        "content_type": CONTENT_TYPE[group],
        "has_template": True,                       # every arm here is templated
        "attack_method": "DIJA" if dose == 0 else "DIJA+prefill",
        "group": group,
        "position": position,
        "dose": dose,
    }


def main() -> int:
    os.makedirs(OUT_ROOT, exist_ok=True)
    src = {g: _load(p) for g, p in SRC.items()}

    # snapshot the exact seed config used
    with open(os.path.join(OUT_ROOT, "seeds_config.json"), "w") as f:
        json.dump(ps.seed_config(), f, indent=2, ensure_ascii=False)

    arms_index = []
    dose0_check = {g: {"checked": 0, "equal": 0} for g in ("B",)}

    for position in ps.POSITIONS:
        for dose in ps.DOSES:
            arm = f"{position}_{dose}"
            for group in ("B", "D"):
                outdir = os.path.join(OUT_ROOT, arm, GROUP_FOLDER[group])
                os.makedirs(outdir, exist_ok=True)
                recs = [build_record(group, c, position, dose) for c in src[group]]
                with open(os.path.join(outdir, "cases.jsonl"), "w") as f:
                    for r in recs:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                # dose-0 byte-equality assertion for B vs source user_content
                if group == "B" and dose == 0:
                    for r, c in zip(recs, src["B"]):
                        dose0_check["B"]["checked"] += 1
                        if r["user_content"] == c["user_content"]:
                            dose0_check["B"]["equal"] += 1
            arms_index.append({"arm": arm, "position": position, "dose": dose,
                               "n_B": len(src["B"]), "n_D": len(src["D"])})

    with open(os.path.join(OUT_ROOT, "arms_index.json"), "w") as f:
        json.dump({"arms": arms_index, "groups": GROUP_FOLDER}, f, indent=2)

    _report(src, dose0_check)
    return 0


def _redact_struct(uc: str) -> str:
    """user_content already contains only {behavior} (no harmful text); show as-is."""
    return uc


def _report(src: dict, dose0_check: dict) -> None:
    cfg = ps.seed_config()
    line = "=" * 78
    print(line)
    print("PHASE 2 REPORT  —  DIJA + prefill seeding arms")
    print(line)
    print(f"source B: {len(src['B'])} cases   source D: {len(src['D'])} cases")
    print(f"arms: {len(ps.POSITIONS)} positions x {len(ps.DOSES)} doses = "
          f"{len(ps.POSITIONS)*len(ps.DOSES)} per group  -> "
          f"output under dijawithprefill/prompts/<position>_<dose>/")

    print("\n--- SEED CONFIG (also saved: dijawithprefill/prompts/seeds_config.json) ---")
    print(f"start primer (FIXED): {cfg['start_primer_fixed']!r}")
    print(f"mid connectives (rotated across masks): {cfg['mid_connectives_rotated']}")
    print(f"mid anchor by dose, 1st mask: {cfg['mid_anchor_examples_first_mask']}")
    print(f"mid anchor by dose, 2nd mask: {cfg['mid_anchor_examples_second_mask']}")
    print(f"seeds ADDED (mask counts held constant): "
          f"{cfg['seeds_added_not_replacing']} / {cfg['mask_count_held_constant']}")
    print(f"NOTE: {cfg['note_start_arm']}")

    print("\n--- FIXED D SCAFFOLD (single shape for all 100 D cases) ---")
    print(ps.D_SCAFFOLD)

    print("\n--- ONE EXAMPLE B RECORD PER POSITION ARM (harmful text redacted: ")
    print("    user_content keeps the {behavior} placeholder; no harmful tokens here) ---")
    b0 = src["B"][0]
    for position in ps.POSITIONS:
        dose = 2 if position != "start" else 1   # show a seeded dose
        rec = build_record("B", b0, position, dose)
        print(f"\n[ position={position}  dose={dose}  id={rec['id']}  "
              f"attack_method={rec['attack_method']} ]")
        print(_redact_struct(rec["user_content"]))

    print("\n--- DOSE-0 EQUALITY ASSERTION (B constructed dose-0 == source prompt) ---")
    c = dose0_check["B"]
    ok = (c["equal"] == c["checked"] and c["checked"] == len(src["B"]) * len(ps.POSITIONS))
    print(f"B dose-0 arms checked: {c['checked']}  byte-identical to source: {c['equal']}")
    print(f"ASSERTION ({len(ps.POSITIONS)} positions x {len(src['B'])} cases = "
          f"{len(ps.POSITIONS)*len(src['B'])} expected): "
          f"{'PASS ✅' if ok else 'FAIL ❌'}")
    assert ok, "dose-0 B prompts are NOT byte-identical to source — baseline broken"
    print(line)


if __name__ == "__main__":
    raise SystemExit(main())
