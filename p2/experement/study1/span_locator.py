#!/usr/bin/env python3
"""study1/span_locator.py — locate the harmful reply span in a final generation.

Study 1 needs, per harmful case, the CHAR (and optionally TOKEN) offsets of the
harmful portion of the REPLY, so that k_commit can be computed over exactly those
reply positions (not the whole canvas).

Ground truth we have: NONE. The judge files store only a yes/no HarmBench label
(medical_scored_llama.jsonl) or a graded rubric with free-text `reasoning`
(scored_harm/…) — neither stores span offsets, and the reasoning rarely quotes the
response verbatim. So this is an explicit, documented HEURISTIC, not extraction of
a stored span. It is honest about what it can and cannot do; callers must eyeball
the output on their data (hand_labels.jsonl is unfilled).

Definition used: the harmful span = the substantive body of a complying response,
i.e. everything AFTER a leading affirmative opener ("Sure, here is …:") and BEFORE
a trailing disclaimer / moralizing caveat tail ("Please consult a physician …").
The opener and the caveat tail are located but excluded from the span (they are
returned in `meta` for inspection). If the text is a refusal / benign / non-harmful
per the label, the span is empty.

Public API:
    locate_harm_span(final_text, harm_label, judge_row=None, tokenizer=None)
        -> {"char_spans": [(s,e), ...],
            "token_spans": [(ts,te), ...] | None,
            "meta": {...}}
"""
from __future__ import annotations

import re
from typing import Optional

# --- leading affirmative opener: an announcement that delivers no content itself.
# Matches "Sure, here is a treatment plan:", "Here's the answer:", "Certainly!",
# "The correct answer is", "Below is …:" etc. Kept deliberately narrow so we never
# eat real content. Anchored at start; consumes up to (and including) a trailing
# ':' when the opener is an announcement, else just the affirmative interjection.
_OPENER_RE = re.compile(
    r"^\s*"
    r"(?:sure|certainly|of course|absolutely|yes)?[\s,!.:-]*"
    r"(?:here\s+(?:is|are|'s)|below\s+is|the\s+(?:correct\s+)?answer(?:\s+is)?|"
    r"i(?:'ll| will| can)\s+(?:help|provide|give|explain)[^\n:]*)"
    r"[^\n:]*:?",
    re.IGNORECASE,
)

# A bare interjection opener with no "here is" ("Sure!", "Certainly.")
_INTERJECTION_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|absolutely|yes|okay|ok)[\s,!.:-]+",
    re.IGNORECASE,
)

# Trailing caveat / disclaimer / moralizing tail — NOT part of the harm delivery.
# We trim the LAST contiguous run of such sentences/lines from the span end.
_CAVEAT_MARKERS = (
    "please note", "please consult", "please be aware", "please remember",
    "consult a", "consult with", "consult your", "seek professional",
    "seek medical", "it is important to consult", "it's important to consult",
    "this advice is general", "this information is general",
    "this is not a substitute", "not a substitute for",
    "disclaimer", "keep in mind that", "note that this",
    "always consult", "important:", "warning:", "caution:",
    "however, it is important", "however, it's important",
    "i must emphasize", "ethical considerations", "this is unethical",
    "is unethical", "violates the principles", "against guidelines",
)

# Markdown separators / decorations we strip from span edges.
_EDGE_JUNK_RE = re.compile(r"^[\s\-\*_#>`\"']+|[\s\-\*_#>`\"']+$")

# A leading echoed request: DIJA responses restate the user's harmful question
# verbatim ("How can I ...?\n") before the worksheet. The echo is not harm delivery.
_REQUEST_ECHO_RE = re.compile(
    r"^\s*(?:how|what|where|when|why|can|could|should|tell\s+me|give\s+me|i\s+(?:want|need|bought)|explain)"
    r"[^?]{0,400}\?\s*",
    re.IGNORECASE,
)

# Immediate word/phrase repetition ("professional professional", "the the the") is a
# degeneration signal — the harm span should not run into the degenerate tail.
_DUP_RUN_RE = re.compile(r"\b(\w{4,})\b(?:\s+\1\b)+", re.IGNORECASE)

# Default cap: the injected worksheet that carries the delivered harm is short; a
# harm span longer than this is almost always sweeping in rambling continuation.
_MAX_SPAN_CHARS = 900


def _is_harmful(harm_label) -> bool:
    """Map a HarmBench yes/no or rubric harmful flag to a bool."""
    if harm_label is None:
        return False
    if isinstance(harm_label, bool):
        return harm_label
    s = str(harm_label).strip().lower()
    return s in ("yes", "1", "true", "harmful", "delivered")


def _strip_opener(text: str) -> int:
    """Return the char index where the substantive body begins (after any opener)."""
    m = _OPENER_RE.match(text)
    if m and m.end() < len(text):
        return m.end()
    m2 = _INTERJECTION_RE.match(text)
    if m2 and m2.end() < len(text):
        return m2.end()
    return 0


def _strip_echoed_request(text: str, body_start: int, judge_row) -> int:
    """Skip a leading restatement of the harmful request (not harm delivery).

    Generic: a leading interrogative sentence ending in '?'. If the row carries the
    original `behavior`, also strip a fuzzy verbatim echo up to the first newline.
    """
    seg = text[body_start:]
    m = _REQUEST_ECHO_RE.match(seg)
    if m and m.end() < len(seg):
        return body_start + m.end()
    beh = (judge_row or {}).get("behavior") if judge_row else None
    if beh:
        head = beh.strip()[:18].lower()
        if head and seg[:80].lower().startswith(head[:12]):
            nl = seg.find("\n")
            if 0 <= nl < 400:
                return body_start + nl + 1
    return body_start


def _degeneration_cut(text: str, s: int, e: int) -> int:
    """Pull the span end back to before the first degenerate repetition run."""
    seg = text[s:e]
    m = _DUP_RUN_RE.search(seg)
    if m and m.start() > 40:          # keep at least a minimal worksheet
        return s + m.start()
    return e


def _find_caveat_tail(text: str, body_start: int) -> int:
    """Return the char index where a trailing caveat tail begins (or len(text)).

    We look for the EARLIEST caveat marker that starts a sentence/line in the
    latter half of the body — everything from there on is treated as caveat tail
    and excluded from the harmful span. Only trims if the marker appears past the
    first third of the body (so an early 'important:' heading is not mistaken).
    """
    low = text.lower()
    body_len = max(1, len(text) - body_start)
    guard = body_start + body_len // 3          # don't trim within the first third
    best = len(text)
    for marker in _CAVEAT_MARKERS:
        idx = low.find(marker, guard)
        while idx != -1:
            # marker must begin a sentence/line (preceded by newline/./: or start)
            prev = text[idx - 1] if idx > 0 else "\n"
            if prev in "\n.:;!?-*â€¢ " or idx == 0:
                best = min(best, idx)
                break
            idx = low.find(marker, idx + 1)
    return best


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    """Shrink [s,e) inward past edge junk / whitespace / markdown rules."""
    seg = text[s:e]
    lead = _EDGE_JUNK_RE.match(seg)
    if lead:
        s += lead.end()
    seg = text[s:e]
    m = re.search(r"[\s\-\*_#>`\"']+$", seg)
    if m:
        e = s + m.start()
    return s, e


def locate_harm_span(final_text: str, harm_label, judge_row: Optional[dict] = None,
                     tokenizer=None, max_span_chars: int = _MAX_SPAN_CHARS) -> dict:
    """Locate the harmful reply span. See module docstring for the definition.

    Args:
      final_text : the decoded REPLY text (model continuation; prompt excluded).
      harm_label : HarmBench 'yes'/'no', rubric harmful 0/1, or bool. Non-harmful
                   -> empty span (nothing was delivered to locate).
      judge_row  : optional full judge record (unused for offsets today; reserved
                   for future rationale-quote matching). Kept in the signature so
                   the API is stable when we add rationale-guided refinement.
      tokenizer  : optional HF tokenizer. If it exposes offset mapping we also
                   return token spans; otherwise token_spans is None.

    Returns dict{char_spans, token_spans, meta}.
    """
    meta = {"strategy": "opener+caveat-trim heuristic", "harmful": _is_harmful(harm_label)}
    if not final_text or not final_text.strip():
        meta["note"] = "empty text"
        return {"char_spans": [], "token_spans": None, "meta": meta}
    if not _is_harmful(harm_label):
        meta["note"] = "label not harmful/delivered -> no span"
        return {"char_spans": [], "token_spans": None, "meta": meta}

    body_start = _strip_opener(final_text)
    body_start = _strip_echoed_request(final_text, body_start, judge_row)
    caveat_start = _find_caveat_tail(final_text, body_start)
    s, e = _trim(final_text, body_start, caveat_start)
    e = _degeneration_cut(final_text, s, e)                 # stop before degenerate tail
    if max_span_chars and (e - s) > max_span_chars:         # anti-balloon cap
        e = s + max_span_chars
        meta["capped"] = max_span_chars

    # Degenerate: opener ate everything, or nothing substantive left -> fall back
    # to the whole trimmed text so we never silently drop a harmful body.
    if e - s < 3:
        s, e = _trim(final_text, 0, len(final_text))
        meta["fallback"] = "opener/caveat trim collapsed span; using full text"

    meta["opener"] = final_text[:body_start]
    meta["echo_stripped"] = body_start
    meta["caveat_tail"] = final_text[caveat_start:].strip()[:200]
    meta["body_len"] = e - s
    meta["total_len"] = len(final_text)

    # crude malformed / low-content flag: harmful responses in this dataset are
    # often fragmented ('.\nStep 1:\n1..'). Flag so callers can down-weight.
    frag = final_text.count("..") + final_text.count("(empty)") + final_text.count("\n.")
    meta["fragmented_flag"] = frag >= 3

    token_spans = None
    if tokenizer is not None:
        token_spans = _char_to_token(final_text, [(s, e)], tokenizer)

    return {"char_spans": [(s, e)], "token_spans": token_spans, "meta": meta}


def _char_to_token(text: str, char_spans, tokenizer):
    """Best-effort char->token span mapping using a fast tokenizer's offsets.

    Returns None if the tokenizer cannot produce an offset mapping (LLaDA's
    tokenizer is not guaranteed to be a 'fast' tokenizer). At capture time we
    have final_ids directly, so token spans are more robustly derived there; this
    helper is a convenience for offline use.
    """
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except Exception:
        return None
    out = []
    for (cs, ce) in char_spans:
        toks = [i for i, (a, b) in enumerate(offsets) if b > cs and a < ce]
        if toks:
            out.append((toks[0], toks[-1] + 1))
    return out or None


# ============================ validation harness ============================
def _validate():
    import json
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    EXP = os.path.dirname(HERE)
    path = os.path.join(EXP, "scored", "medical_scored_llama.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    harmful_arm = [r for r in rows if r.get("arm") == "harmful"]
    delivered = [r for r in harmful_arm if r.get("judge_label") == "yes"]
    refused = [r for r in harmful_arm if r.get("judge_label") == "no"]

    print(f"loaded {len(rows)} rows | harmful arm={len(harmful_arm)} "
          f"| HarmBench yes={len(delivered)} no={len(refused)}\n")
    print("=" * 100)
    print("VALIDATION — 8 harmful-arm rows (HarmBench judge = delivered). "
          "Eyeball located span vs full text.")
    print("=" * 100)

    sample = delivered[:8] if len(delivered) >= 8 else (delivered + harmful_arm)[:8]
    for r in sample:
        text = r["response"]
        res = locate_harm_span(text, r.get("judge_label"), r)
        (s, e) = res["char_spans"][0] if res["char_spans"] else (0, 0)
        print(f"\n--- {r['id']} ({r.get('topic')}) HarmBench={r.get('judge_label')} "
              f"fragmented={res['meta'].get('fragmented_flag')} ---")
        print(f"  PROMPT : {r['prompt_text'][:90]!r}")
        print(f"  FULL   : {text!r}")
        print(f"  OPENER : {res['meta'].get('opener','')!r}")
        print(f"  SPAN   : {text[s:e]!r}")
        if res["meta"].get("caveat_tail"):
            print(f"  CAVEAT-: {res['meta']['caveat_tail']!r}")
        if res["meta"].get("fallback"):
            print(f"  NOTE   : {res['meta']['fallback']}")

    # also show it correctly returns [] on a couple of refusals
    print("\n" + "=" * 100)
    print("SANITY — refusals / non-delivered should yield EMPTY span:")
    for r in refused[:3]:
        res = locate_harm_span(r["response"], r.get("judge_label"), r)
        print(f"  {r['id']}: span={res['char_spans']}  full={r['response'][:70]!r}")


if __name__ == "__main__":
    _validate()
