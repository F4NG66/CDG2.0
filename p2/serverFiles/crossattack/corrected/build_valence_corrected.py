#!/usr/bin/env python
"""Build valence-judge input for the NEW corrected attack2 B2 states (response_text
from the corrected capture manifest, behavior from the corrected prompt-root).
traj_id = "attack2c:<case_id>". Writes crossattack/corrected/valence_input_b2.jsonl."""
import json, os

ROOT = "/home/ore99/serverFiles"
MAN = os.path.join(ROOT, "crossattack/corrected/outputs/manifest.jsonl")
PROMPTS = os.path.join(ROOT, "crossattack/corrected/prompts/attack2_qwen/B2_harmful_injected/cases.jsonl")
OUT = os.path.join(ROOT, "crossattack/corrected/valence_input_b2.jsonl")


def main():
    beh = {}
    with open(PROMPTS) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                beh[r["id"]] = r["behavior"]

    resp = {}
    with open(MAN) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("variant") == "B2_harmful_injected":
                    resp[r["case_id"]] = r.get("response_text", "")   # last wins

    rows = []
    for cid in sorted(resp):
        rows.append({"traj_id": f"attack2c:{cid}", "id": cid,
                     "family": "attack2_corrected",
                     "behavior": beh[cid], "response": resp[cid]})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[valence-input] {len(rows)} B2 rows -> {OUT}")
    empty = sum(1 for r in rows if not r["response"].strip())
    print(f"[valence-input] empty responses: {empty}")


if __name__ == "__main__":
    main()
