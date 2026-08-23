#!/usr/bin/env python3
"""attack2/contextual/refine_contextual.py — Phase 2.

Drive the EXISTING DIJA Qwen refiner (imported, unedited) with a NEW dialogue-style
few-shot template so each of the 100 attack2 source behaviors becomes a MENTOR/NEWCOMER
scaffold with per-behavior CONTEXTUAL anchoring around every <mask:N> — the thing the
original context-free attack2 scaffold lacked (it filled safe -> 3%).

Reuse, never edit:
  - QwenRefiner        (dija_refiner.py)  -> the unedited Qwen model load + _generate()
  - is_valid_scaffold  (dija_refiner.py)  -> the SAME DIJA validity contract (2-5 masks,
                                             no refusal marker, >=1 mask with leading context)
  - parse_masks        (dija_refiner.py)

We add only: (a) the new template (dialogue_refine_template.txt), (b) a thin retry/validate
loop (mirrors QwenRefiner.refine but with a larger token budget for the longer dialogue and
a light post-clean), (c) a JSON writer + redacted preview.

Structurally distinct from DIJA: mentor<->newcomer dialogue, scaffold placed AFTER the request,
NOT a "Step 1/2/3" worksheet. No harmful content is authored here — the template's two worked
examples are benign (composting, 10k training); Qwen writes the per-behavior framing; the masks
stay EMPTY.

    # after Qwen2.5-7B is cached (HF_HOME=/scratch/ore99/hf_cache):
    HF_HOME=/scratch/ore99/hf_cache python attack2/contextual/refine_contextual.py
    # quick check on 3:
    HF_HOME=/scratch/ore99/hf_cache python attack2/contextual/refine_contextual.py --limit 3
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../attack2/contextual
ATTACK2 = os.path.dirname(HERE)                                   # .../attack2
SERVER = os.path.dirname(ATTACK2)                                 # .../serverFiles
EXP = "/home/ore99/experement"                                   # holds dija_refiner.py
for p in (EXP, SERVER):
    if p not in sys.path:
        sys.path.insert(0, p)

# default HF cache to scratch (Qwen was downloaded there; /home is full)
os.environ.setdefault("HF_HOME", "/scratch/ore99/hf_cache")

from dija_refiner import QwenRefiner, is_valid_scaffold, parse_masks  # noqa: E402  (reused, unedited)

SOURCE = os.path.join(ATTACK2, "source_A", "cases.json")
TEMPLATE = os.path.join(HERE, "dialogue_refine_template.txt")
OUT = os.path.join(HERE, "refined_100.json")
DL_LOG = os.path.join(HERE, "logs", "qwen_download.log")

SPEAKER_RE = re.compile(r"^(Newcomer:|Mentor:)")


def resolve_qwen_path(cli: str | None) -> str:
    """CLI path wins; else the concrete snapshot dir captured by the download; else repo id
    (resolved from HF_HOME cache)."""
    if cli:
        return cli
    if os.path.exists(DL_LOG):
        for line in open(DL_LOG, encoding="utf-8", errors="ignore"):
            if line.startswith("QWEN_LOCAL_PATH="):
                return line.strip().split("=", 1)[1]
    return "Qwen/Qwen2.5-7B-Instruct"


def clean_scaffold(raw: str) -> str:
    """Keep the dialogue only: join wrapped lines into their speaker turn, drop any
    preamble/notes that are not Newcomer:/Mentor: turns. Never edits content inside a turn."""
    segments: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if SPEAKER_RE.match(s):
            segments.append(s)
        elif segments:                      # continuation of the current turn (wrapped line)
            segments[-1] += " " + s
    return "\n".join(segments).strip()


def refine_one(refiner, template: str, behavior: str, max_attempts: int,
               rng: random.Random, max_new_tokens: int):
    """Mirror QwenRefiner.refine() but with our token budget + post-clean. Uses the unedited
    QwenRefiner._generate() for the actual Qwen call and the unedited is_valid_scaffold()."""
    prompt = template.replace("{prompt}", behavior)   # .replace (not .format) => brace-safe
    last, reason = "", "no attempts"
    for attempt in range(1, max_attempts + 1):
        temp = 0.3 if attempt == 1 else rng.uniform(0.2, 0.7)
        raw = refiner._generate(prompt, temperature=temp, max_new_tokens=max_new_tokens)
        scaffold = clean_scaffold(raw)
        ok, reason = is_valid_scaffold(scaffold)
        if ok:
            return scaffold, {"valid": True, "attempts": attempt, "reason": "ok"}
        last = scaffold or last
    return last, {"valid": False, "attempts": max_attempts, "reason": reason}


def redact_for_display(scaffold: str, behavior: str) -> str:
    """Structure-only preview: strip the behavior's own content words (len>=5) from the frame
    so we show dialogue shape + <mask:N> + connective phrasing WITHOUT the harmful subject text."""
    words = sorted({w.lower() for w in re.findall(r"[A-Za-z]{5,}", behavior)}, key=len, reverse=True)
    red = scaffold
    for w in words:
        red = re.sub(rf"(?i)\b{re.escape(w)}\w*", "···", red)
    return red


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-path", default=None,
                    help="local snapshot dir or repo id (default: auto from download log / cache)")
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=3, help="redacted examples to print")
    args = ap.parse_args()

    template = open(args.template, encoding="utf-8").read()
    assert "{prompt}" in template, "template must contain a {prompt} placeholder"
    src = json.load(open(args.source))
    if args.limit:
        src = src[:args.limit]
    print(f"[refine] {len(src)} behaviors from {args.source}", flush=True)

    qwen_path = resolve_qwen_path(args.qwen_path)
    print(f"[refine] Qwen path: {qwen_path}  (HF_HOME={os.environ.get('HF_HOME')})", flush=True)
    refiner = QwenRefiner(qwen_path)                  # unedited load
    rng = random.Random(args.seed)

    rows, n_invalid = [], 0
    for i, c in enumerate(src):
        beh, cid = c["behavior"], c["id"]
        scaffold, meta = refine_one(refiner, template, beh, args.max_attempts, rng,
                                    args.max_new_tokens)
        masks = parse_masks(scaffold)
        if not meta["valid"]:
            n_invalid += 1
            print(f"  [invalid] {cid}: {meta['reason']}", flush=True)
        rows.append({
            "id": cid,
            "behavior": beh,                          # byte-identical from source_A
            "Refined_behavior": scaffold,             # mentor/newcomer contextual scaffold
            "n_masks": len(masks),
            "mask_sizes": [m["n"] for m in masks],
            "masks_with_leading": sum(m["has_leading"] for m in masks),
            "masks_with_trailing": sum(m["has_trailing"] for m in masks),
            "valid": meta["valid"],
            "attempts": meta["attempts"],
            "reason": meta["reason"],
        })
        if (i + 1) % 10 == 0 or i + 1 == len(src):
            print(f"[refine] {i+1}/{len(src)} done  (invalid so far: {n_invalid})", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    n_valid = sum(r["valid"] for r in rows)
    mm = [r["n_masks"] for r in rows]
    print(f"\n[refine] wrote {len(rows)} -> {args.out}")
    print(f"[refine] valid: {n_valid}/{len(rows)}  | masks/case min={min(mm)} max={max(mm)}")

    # ---- redacted, structure-only preview (harmful subject words stripped) ----
    print(f"\n===== {min(args.preview, len(rows))} REDACTED examples (structure only; "
          f"subject words -> ···; masks are EMPTY) =====")
    for r in rows[:args.preview]:
        print(f"\n--- {r['id']}  (valid={r['valid']}, {r['n_masks']} masks {r['mask_sizes']}) ---")
        print("[request] {behavior}\\n\\n<<TPL>>")
        print(redact_for_display(r["Refined_behavior"], r["behavior"]))
        print("<</TPL>>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
