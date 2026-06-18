#!/usr/bin/env python3
"""p1_check.py — Data QC & Pre-flight Check. Wraps s1_check_records.py with extra validation."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from cdg.probe import load_records
from cdg.config import get_backend_config
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="outputs")
args = ap.parse_args()
cfg     = get_backend_config("llada_attack")
records = load_records(args.out_dir, cfg.name)
from collections import Counter
cnt = Counter(r["variant"][:1].upper() for r in records)
print(f"\n{'='*60}")
print(f"  Records: {len(records)}  |  Groups: {dict(sorted(cnt.items()))}")
print(f"{'='*60}")
issues = []
for r in records:
    rec = r.get("_rec") or {}
    sae = rec.get("sae", {})
    g = r["variant"][:1].upper()
    j = r.get("judge") or {}
    resp = (r.get("response_text") or "")[:60].replace("\n"," ")
    judge_str = f"success={j['success']}({j.get('label')})" if j.get("success") is not None else "not judged"
    scopes = list(sae.keys())
    if not scopes: issues.append(f"[{g}] {r['case_id']}: NO SAE DATA recorded!")
    print(f"[{g}] {r['case_id']:6s}  judge={judge_str:20s}  resp={resp!r}")
    print(f"       scopes={scopes}")
if issues:
    print("\n\033[91mWARNINGS:\033[0m")
    for w in issues: print(f"  {w}")
else:
    print("\n\033[92mAll records OK.\033[0m")
print(f"\nNext: python scripts/p2_assemble.py {args.out_dir}")
