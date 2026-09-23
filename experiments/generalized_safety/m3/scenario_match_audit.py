"""
Mandatory fail-closed scenario-match audit.

Purpose: catch exactly the class of bug found in M3 BUILD/DEV's RENeLLM
construction (generalized_safety_v2_renellm_v31_full_execution_v3), where the
text actually wrapped into a RENeLLM attack scenario came from a different
canonical source file than the text later shown to the judge as
`base_request` — same pair-id numbering, different underlying content.

This module extracts the embedded scenario text from a constructed RENeLLM
attack prompt and compares it against the base_request the construction
CLAIMS it attacked. Any attack the pipeline actually intends to use for
generation/judging must pass this audit before proceeding — do not silently
skip an unresolvable item and continue past a hard mismatch on a resolvable
one.

Required at M3 contract pipeline step 8 ("only now: construct attacks for the
final pool and extract hidden states") before generation is run on the final
pool: call `audit_or_raise(rows)` — one row per constructed attack, each with
`scenario_id`, `attacked_prompt` (the literal text sent to the model), and
`claimed_base_request` (the text that will be shown to the judge / used as
this item's semantic identity). It raises on the first resolvable mismatch;
it never silently drops or "fixes" a bad row.

Known scenario templates in this project (extend SCENARIO_PATTERNS if a new
RENeLLM scenario type is ever added — do not add a bare `None` fallback that
lets an unknown scenario silently count as "unresolved" without being logged).
"""

import re
from collections import Counter

SCENARIO_PATTERNS = {
    "code_completion": re.compile(r"\{(.*?)\}\s*function", re.S),
    "table_filling": re.compile(r"\{\|c\|c\|\}\s*\{(.*?)\}\s*\\\\", re.S),
    "text_continuation": re.compile(r'course title:\s*"(.*?)"', re.S),
}

MATCH_THRESHOLD = 0.3  # word-level Jaccard; same threshold used to root-cause the M3 bug


def extract_embedded(scenario_id, attacked_prompt):
    pattern = SCENARIO_PATTERNS.get(scenario_id)
    if pattern is None:
        return None
    m = pattern.search(attacked_prompt)
    return m.group(1).strip() if m else None


def jaccard(a, b):
    sa, sb = set((a or "").lower().split()), set((b or "").lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def audit(rows):
    """rows: iterable of dicts with scenario_id, attacked_prompt, claimed_base_request,
    and an identifier field (e.g. pair_id or group_id) for error messages.
    Returns a summary dict. Never raises — use audit_or_raise for the fail-closed gate."""
    counts = Counter()
    mismatches = []
    for r in rows:
        emb = extract_embedded(r["scenario_id"], r["attacked_prompt"])
        if emb is None:
            counts["unresolved"] += 1
            continue
        score = jaccard(emb, r["claimed_base_request"])
        if score > MATCH_THRESHOLD:
            counts["match"] += 1
        else:
            counts["mismatch"] += 1
            ident = r.get("id")
            if ident is None:
                ident = r.get("pair_id")
            if ident is None:
                ident = r.get("group_id")
            mismatches.append({
                "id": ident,
                "scenario_id": r["scenario_id"],
                "embedded_text": emb,
                "claimed_base_request": r["claimed_base_request"],
                "jaccard": round(score, 4),
            })
    return {
        "total": len(list(rows)) if not isinstance(rows, list) else len(rows),
        "match": counts["match"],
        "mismatch": counts["mismatch"],
        "unresolved": counts["unresolved"],
        "mismatches": mismatches,
        "PASS": counts["mismatch"] == 0,
    }


def audit_or_raise(rows, context=""):
    """Fail-closed gate. Call this before running generation/judging on any
    freshly-constructed RENeLLM attack batch. Raises RuntimeError on the
    first batch containing any resolvable mismatch — construction must be
    fixed and re-run, not patched around."""
    rows = list(rows)
    result = audit(rows)
    if not result["PASS"]:
        raise RuntimeError(
            f"SCENARIO_MATCH_AUDIT_FAILED{(' [' + context + ']') if context else ''}: "
            f"{result['mismatch']} of {len(rows)} resolvable rows attacked a different "
            f"request than their claimed base_request. First mismatch: "
            f"{result['mismatches'][0]}. Do not proceed — fix the construction source "
            f"binding and re-run from step 8, not from here.")
    return result
