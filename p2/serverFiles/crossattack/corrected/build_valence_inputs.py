#!/usr/bin/env python
"""crossattack/corrected/build_valence_inputs.py

PHASE 2 (measure-first diagnostic). Build the valence-judge input jsonl for the
EXISTING (already fill_all_masks=True) injected states of BOTH families, so we can
score their harm-delivered efficacy with the paper valence judge on the SAME
generation the AUC=1.0 probe saw -- with NO regeneration and NO GPU.

Reads ONLY existing artifacts (never edits them):
  - attack2 injected : crossattack/outputs/manifest.jsonl  (variant B2_harmful_injected)
                       behavior <- crossattack/prompts/attack2/B2_harmful_injected/cases.jsonl
  - DIJA injected    : outputs/manifest.jsonl              (variant B_harmful_injected)
                       behavior <- prompts/cdg_injection/B_harmful_injected/cases.json

response = manifest 'response_text' (== dlm_runner output: filled <<TPL>> blanks +
output tail, the exact text behind the transfer states). Writes ONE combined jsonl
under crossattack/corrected/. traj_id = "<family>:<case_id>" (judge key, resumable).
"""
import json, os

ROOT = "/home/ore99/serverFiles"
OUT = os.path.join(ROOT, "crossattack/corrected/valence_input.jsonl")


def load_manifest(path, variant):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("variant") == variant:
                rows[r["case_id"]] = r.get("response_text", "")
    return rows


def load_behaviors_jsonl(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["id"]] = r["behavior"]
    return d


def load_behaviors_json(path):
    d = {}
    for r in json.load(open(path)):
        d[r["id"]] = r["behavior"]
    return d


def build(family, manifest, variant, beh):
    resp = load_manifest(manifest, variant)
    out = []
    missing_resp = missing_beh = 0
    for cid in sorted(resp):
        b = beh.get(cid)
        if b is None:
            missing_beh += 1
            continue
        if not resp[cid]:
            missing_resp += 1
        out.append({"traj_id": f"{family}:{cid}", "id": cid, "family": family,
                    "behavior": b, "response": resp[cid]})
    print(f"[{family}] variant={variant}: {len(out)} rows "
          f"(missing_behavior={missing_beh}, empty_response={missing_resp})")
    return out


def main():
    a2 = build("attack2",
               os.path.join(ROOT, "crossattack/outputs/manifest.jsonl"),
               "B2_harmful_injected",
               load_behaviors_jsonl(os.path.join(
                   ROOT, "crossattack/prompts/attack2/B2_harmful_injected/cases.jsonl")))
    dija = build("dija",
                 os.path.join(ROOT, "outputs/manifest.jsonl"),
                 "B_harmful_injected",
                 load_behaviors_json(os.path.join(
                     ROOT, "prompts/cdg_injection/B_harmful_injected/cases.json")))
    allrows = a2 + dija
    with open(OUT, "w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {len(allrows)} rows -> {OUT}")
    # quick sanity: show one response head per family
    for fam in ("attack2", "dija"):
        ex = next(r for r in allrows if r["family"] == fam)
        print(f"\n[{fam}] {ex['id']}  behavior[:70]={ex['behavior'][:70]!r}")
        print(f"          response[:160]={ex['response'][:160]!r}")


if __name__ == "__main__":
    main()
