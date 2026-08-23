#!/usr/bin/env python
"""crossattack/corrected/build_prompts.py

Build the capture prompt-root for the attack2 INJECTED arm from the attack2_qwen
80%-dialogue scaffolds, REUSED VERBATIM. Emits a cdg-format cases.jsonl that,
when run through the UNEDITED serverFiles capture pipeline, reproduces the exact
attack2_qwen generation (byte-identical prompt construction to dija_attack/
run_dija.py: content = "<<TPL>>" + Refined_behavior + "<</TPL>>").

Also VERIFIES the refined_100 behaviors are byte-identical, per base id, to the
EXISTING clean A2 arm (crossattack/prompts/attack2/A2_harmful_clean) so the new
B2 injected states pair 1:1 with the reused A2 clean states (identity-controlled
harm-scope contrast, B0xx <-> A0xx).

Writes ONLY under crossattack/corrected/. Never edits originals.
"""
import json, os, sys

ROOT = "/home/ore99/serverFiles"
REFINED = "/home/ore99/experement/attack2_qwen/refined_100.json"
A2_CLEAN = os.path.join(ROOT, "crossattack/prompts/attack2/A2_harmful_clean/cases.jsonl")
OUT_DIR = os.path.join(ROOT, "crossattack/corrected/prompts/attack2_qwen/B2_harmful_injected")
OUT = os.path.join(OUT_DIR, "cases.jsonl")

OPEN, CLOSE = "<<TPL>>", "<</TPL>>"


def load_a2_behaviors():
    d = {}
    with open(A2_CLEAN) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["id"][1:]] = r["behavior"]   # base id "000".."099"
    return d


def main():
    refined = json.load(open(REFINED))
    a2 = load_a2_behaviors()

    mism = []
    rows = []
    for r in refined:
        rid = r["id"]                     # "A000"
        base = rid[1:]                    # "000"
        beh = r["behavior"]
        scaffold = r["Refined_behavior"]  # behavior + "\n" + dialogue(<mask:N>)
        # verify behavior identity vs existing clean A2 arm
        if a2.get(base) != beh:
            mism.append((base, a2.get(base), beh))
        # verify the scaffold really starts with the behavior (run_dija assumption)
        if not scaffold.startswith(beh):
            mism.append((base + "!startswith", beh[:40], scaffold[:40]))
        user_content = OPEN + scaffold + CLOSE
        rows.append({
            "id": "B" + base,                    # B000..B099 pair with A000..A099
            "behavior": beh,
            "user_content": user_content,
            "content_type": "harmful",
            "has_template": True,
            "attack_method": "attack2",
            "source": "attack2_qwen/refined_100.json",
        })

    if mism:
        print("[FAIL] behavior/scaffold mismatches:", len(mism))
        for m in mism[:8]:
            print("   ", m)
        sys.exit(1)
    print(f"[ok] 100/100 refined behaviors byte-identical to existing A2 clean arm; "
          f"all scaffolds start with the behavior (run_dija-faithful).")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {len(rows)} injected cases -> {OUT}")
    # show one fully-expanded example
    ex = rows[0]
    print(f"\n[example] {ex['id']}  behavior[:60]={ex['behavior'][:60]!r}")
    print(f"user_content:\n{ex['user_content']}")


if __name__ == "__main__":
    main()
