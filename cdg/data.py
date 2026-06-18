from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Group convention (folder name prefix -> condition).  This is the 2x2:
#   content in {harmful, neutral}  x  template in {present, absent}
#
#   A_harmful_clean      harmful  , no template   -> model should REFUSE
#   B_harmful_injected   harmful  , + template    -> model COMPLIES (attack works)
#   C_neutral_injected   neutral  , + template    -> injection structure, benign fill
#   D_neutral_clean      neutral  , no template   -> normal answer (baseline)
# ---------------------------------------------------------------------------
GROUP_TABLE = {
    "A": dict(content_type="harmful", has_template=False),
    "B": dict(content_type="harmful", has_template=True),
    "C": dict(content_type="neutral", has_template=True),
    "D": dict(content_type="neutral", has_template=False),
}


@dataclass
class PromptCase:
    variant: str                       # group / folder name
    case_id: str
    # NEW injection path -------------------------------------------------
    behavior: str = ""                 # the question / behavior text
    user_content: str = "{behavior}"   # raw template, may hold {behavior},
                                       # <<TPL>>...<</TPL>> and <mask:N>
    content_type: str = "harmful"      # "harmful" | "neutral"
    has_template: bool = False
    attack_method: str = "none"
    # OLD chat path (back-compat) ---------------------------------------
    messages: Optional[list] = None
    is_neutral: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def group_letter(self) -> str:
        return self.variant[:1].upper()


def _infer_group(variant: str, d: dict) -> dict:
    g = variant[:1].upper()
    base = dict(GROUP_TABLE.get(g, {}))
    if "content_type" in d:
        base["content_type"] = d["content_type"]
    if "has_template" in d:
        base["has_template"] = bool(d["has_template"])
    base.setdefault("content_type", "harmful")
    base.setdefault("has_template", False)
    return base


def _user_content(d: dict) -> str:
    if "user_content" in d:
        return d["user_content"]
    # default assembly: clean prompt = just the behavior
    return "{behavior}"


def _meta(d: dict) -> dict:
    skip = {"id", "behavior", "user_content", "messages", "prompt",
            "content_type", "has_template", "attack_method"}
    return {k: v for k, v in d.items() if k not in skip}


def _case_from_dict(d: dict, variant: str, stem: str, idx: int) -> PromptCase:
    cid = str(d.get("id", f"{stem}_{idx}"))
    grp = _infer_group(variant, d)
    behavior = d.get("behavior", d.get("prompt", ""))
    uc = _user_content(d)
    # old chat-style record still works
    messages = d.get("messages")
    return PromptCase(
        variant=variant,
        case_id=cid,
        behavior=behavior,
        user_content=uc,
        content_type=grp["content_type"],
        has_template=grp["has_template"],
        attack_method=d.get("attack_method", "DIJA" if grp["has_template"] else "none"),
        messages=messages,
        is_neutral=(grp["content_type"] == "neutral"),
        meta=_meta(d),
    )


def _load_file(path: str, variant: str) -> list[PromptCase]:
    stem = os.path.splitext(os.path.basename(path))[0]
    if path.endswith(".jsonl"):
        out = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                out.append(_case_from_dict(json.loads(line), variant, stem, i))
        return out
    if path.endswith(".json"):
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return [_case_from_dict(obj, variant, stem, 0)]
        if isinstance(obj, list):
            # Flatten one level of accidental nesting: [[{...}], {...}] → [{...}, {...}]
            flat = []
            for item in obj:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            return [_case_from_dict(d, variant, stem, i)
                    for i, d in enumerate(flat) if isinstance(d, dict)]
    if path.endswith(".txt"):
        with open(path) as f:
            txt = f.read().strip()
        return [_case_from_dict({"behavior": txt}, variant, stem, 0)]
    return []


def load_cdg_root(prompt_root: str) -> list[PromptCase]:
    """Load the 4-group injection corpus.  Each subdir = one group (A/B/C/D)."""
    if not os.path.isdir(prompt_root):
        raise FileNotFoundError(f"prompt root not found: {prompt_root}")
    cases: list[PromptCase] = []
    for variant in sorted(os.listdir(prompt_root)):
        vdir = os.path.join(prompt_root, variant)
        if not os.path.isdir(vdir):
            continue
        for fn in sorted(os.listdir(vdir)):
            cases.extend(_load_file(os.path.join(vdir, fn), variant))
    if not cases:
        raise RuntimeError(f"{prompt_root}: no cases found")
    return cases


# --- backward compatible loader (old neutral-name based) -------------------
def load_prompt_root(prompt_root: str, neutral_name: str = "neutral") -> list[PromptCase]:
    cases = load_cdg_root(prompt_root)
    for c in cases:
        if c.variant == neutral_name:
            c.is_neutral = True
            c.content_type = "neutral"
    return cases
