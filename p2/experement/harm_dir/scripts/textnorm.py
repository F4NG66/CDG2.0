#!/usr/bin/env python3
"""Shared response-text normalisation.

Applied IDENTICALLY to harm responses, natural safe responses and generated safe
responses. Anything applied to one side only would become the signal.

Two steps:
  1. strip the echoed user behavior that the decoded sequence re-emits
  2. strip the scaffold-exit boilerplate: the '<' end-of-template marker and any
     trailing canned refusal ("I'm sorry, but I can't assist with that.")

Step 2 matters: the tail is present on 82% of harmful responses but 86% of
natural safe ones, and generated text carries it inconsistently, so its
presence/absence is weakly discriminative on its own. Removing it everywhere
costs no harm signal - the tail is identical boilerplate on both sides.
"""
import re

# canned refusal / apology tail, optionally preceded by the '<' template-exit mark
_TAIL = re.compile(
    r"\s*<?\s*(?:I\s*'?\s*m\s+sorry|I\s+cannot\s+assist|I\s+can\s*'?\s*t\s+assist|"
    r"I\s+am\s+sorry|Sorry\s*,)\b.*\Z",
    re.IGNORECASE | re.DOTALL)

# a bare trailing '<' with nothing after it
_ANGLE = re.compile(r"\s*<\s*\Z")

# leading "Sure, here is the procedure:"-style compliance primer is part of the
# prefill seed and appears on BOTH sides, so it is left alone deliberately.


def strip_echo(response_text: str, behavior: str) -> str:
    t = (response_text or "").strip()
    b = (behavior or "").strip()
    if b and t.startswith(b):
        t = t[len(b):]
    return t.strip()


def strip_tail(text: str) -> str:
    t = (text or "").strip()
    t = _TAIL.sub("", t)
    t = _ANGLE.sub("", t)
    return t.strip()


def normalize(response_text: str, behavior: str = "") -> str:
    return strip_tail(strip_echo(response_text, behavior))


def quality(text: str) -> dict:
    toks = re.findall(r"[A-Za-z]+|[0-9]+", text)
    words = [t.lower() for t in toks if not t.isdigit()]
    digit_frac = (sum(1 for t in toks if t.isdigit()) / len(toks)) if toks else 1.0
    uniq = (len(set(words)) / len(words)) if words else 0.0
    tri = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    trirep = (1 - len(set(tri)) / len(tri)) if tri else 0.0
    return {"words": len(text.split()), "digit_frac": round(digit_frac, 4),
            "uniq_word_ratio": round(uniq, 4), "trigram_rep": round(trirep, 4)}
