#!/usr/bin/env python3
"""
s1_check_records.py
-------------------
Inspect Phase-1 records: how many cases, what groups, attack success rates,
and what the model actually generated.

Run: python scripts/s1_check_records.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)

from cdg.probe import load_records, group_letter
from cdg.config import get_backend_config

cfg = get_backend_config("llada_attack")
records = load_records("outputs", cfg.name)

# ── group counts ────────────────────────────────────────────────────────────
from collections import Counter
cnt = Counter(group_letter(r) for r in records)
print(f"\n{'='*60}")
print(f"  Records loaded: {len(records)}  |  Groups: {dict(sorted(cnt.items()))}")
print(f"{'='*60}\n")

# ── per-record summary ──────────────────────────────────────────────────────
for r in records:
    g   = group_letter(r)
    cid = r["case_id"]
    j   = r.get("judge") or {}
    rec = r.get("_rec") or {}

    # check what scopes / layers were recorded
    sae_store = rec.get("sae", {})
    scopes = list(sae_store.keys())
    if scopes:
        first_scope = sae_store[scopes[0]]
        fracs  = sorted(first_scope.keys())
        if fracs:
            layers = sorted(first_scope[fracs[0]].keys())
        else:
            layers = []
    else:
        fracs, layers = [], []

    resp_snippet = r.get("response_text", "")[:80].replace("\n", " ")
    judge_str    = (f"success={j['success']} ({j.get('label')})"
                    if j.get("success") is not None else "not judged")

    print(f"[{g}] {cid}")
    print(f"     judge   : {judge_str}")
    print(f"     response: {resp_snippet!r}")
    print(f"     recorded: scopes={scopes}  fracs={fracs}  layers={layers}")
    regions = rec.get("regions", r.get("regions", {}))
    print(f"     regions : {regions}")
    print()

# ── attack success rate ─────────────────────────────────────────────────────
b_records = [r for r in records if group_letter(r) == "B"]
judged    = [r for r in b_records if (r.get("judge") or {}).get("success") is not None]
if judged:
    asr = sum((r["judge"]["success"] or 0) for r in judged) / len(judged)
    print(f"Baseline ASR (group B):  {asr:.1%}  ({len(judged)} judged cases)")
else:
    print("Group B has no judge results yet.")
print()
