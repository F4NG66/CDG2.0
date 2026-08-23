"""collapse_report.py - Collapse-rate report over judge-scored paired runs.

Reads one or more *_scored.jsonl files and reports, for the HARMFUL arm:
  - total harmful rows, malformed (artifact) count, scoreable count
  - collapse count and TWO rates:
      * over scoreable  (the honest attack-success rate)
      * over all harmful (includes malformed as non-collapse, for reference)
  - a benign-arm sanity check (benign should almost never "collapse"; if it does,
    the judge is miscalibrated)
  - a per-group breakdown (by `bucket` if present, else `topic`)

Pass several files to compare richness/template variants side by side. Denominators
are handled per file, so a run with many malformed rows (e.g. minimal) is not
unfairly compared by raw counts to a clean run.

    python collapse_report.py scored/standard_scored.jsonl scored/medical_scored.jsonl scored/minimal_scored.jsonl
    python collapse_report.py scored/standard_scored.jsonl --list   # also print which items collapsed
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def group_key(r: dict) -> str:
    return r.get("bucket") or r.get("topic") or "(none)"


def intent(r: dict) -> str:
    return (r.get("prompt_text", "").splitlines() or [""])[0].strip()


def summarize(rows: list[dict], label: str, show_list: bool) -> dict:
    harmful = [r for r in rows if r.get("arm") == "harmful"]
    benign = [r for r in rows if r.get("arm") == "benign"]
    malformed = [r for r in harmful if r.get("malformed")]
    scoreable = [r for r in harmful if not r.get("malformed")]
    collapse = [r for r in scoreable if r.get("successful")]
    benign_collapse = [r for r in benign if r.get("successful") and not r.get("malformed")]

    rate_scoreable = len(collapse) / len(scoreable) if scoreable else 0.0
    rate_all = len(collapse) / len(harmful) if harmful else 0.0

    print(f"\n=== {label} ===")
    print(f"  harmful rows:       {len(harmful)}")
    print(f"  malformed (artifact): {len(malformed)}")
    print(f"  scoreable:          {len(scoreable)}")
    print(f"  COLLAPSE:           {len(collapse)}")
    print(f"  collapse rate (over scoreable):   {rate_scoreable:.1%}")
    print(f"  collapse rate (over all harmful): {rate_all:.1%}")
    if benign_collapse:
        print(f"  !! benign-arm collapses: {len(benign_collapse)} "
              f"(judge may be miscalibrated; benign should ~never collapse)")
    else:
        print(f"  benign sanity: OK (0 benign collapses)")

    # per-group breakdown over scoreable
    g_total = defaultdict(int)
    g_collapse = defaultdict(int)
    for r in scoreable:
        g_total[group_key(r)] += 1
    for r in collapse:
        g_collapse[group_key(r)] += 1
    if len(g_total) > 1:
        print("  by group (collapse / scoreable):")
        for g in sorted(g_total, key=lambda k: -g_collapse[k]):
            c, t = g_collapse[g], g_total[g]
            print(f"    {c:>3}/{t:<3}  {c/t:5.0%}  {g}")

    if show_list and collapse:
        print("  collapsed items:")
        for r in collapse:
            print(f"    {r.get('id')}  sev={r.get('severity')}  {intent(r)[:70]}")

    return {"label": label, "harmful": len(harmful), "malformed": len(malformed),
            "scoreable": len(scoreable), "collapse": len(collapse),
            "rate_scoreable": rate_scoreable, "rate_all": rate_all}


def main() -> int:
    ap = argparse.ArgumentParser(description="Collapse-rate report over scored runs.")
    ap.add_argument("files", nargs="+", help="one or more *_scored.jsonl files")
    ap.add_argument("--list", action="store_true", help="print which items collapsed")
    args = ap.parse_args()

    summaries = []
    for path in args.files:
        label = Path(path).stem.replace("_scored", "")
        summaries.append(summarize(load(path), label, args.list))

    if len(summaries) > 1:
        print("\n=== comparison (rate over scoreable) ===")
        print(f"  {'variant':<22}{'collapse':>9}{'scoreable':>11}{'malformed':>11}{'rate':>8}")
        for s in summaries:
            print(f"  {s['label']:<22}{s['collapse']:>9}{s['scoreable']:>11}"
                  f"{s['malformed']:>11}{s['rate_scoreable']:>7.0%}")
        print("\n  note: compare 'rate over scoreable'. Variants with high malformed "
              "(e.g. minimal) had broken generation, not resistance — their raw "
              "collapse COUNT understates because the pool shrank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())