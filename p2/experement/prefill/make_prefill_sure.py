"""make_prefill_sure.py - Build the canonical "Sure, here is" prefill set.

Reads every case_*.json in ../neutral, takes each case's `harmful_variant`
(the request we actually attack), and pairs it with the single content-free
affirmative prefix "Sure, here is". No harmful content is authored here - the
harmful request comes only from the neutral case files.

Output: prefill_data/sure_prefills.json  (a list of prefill records)

Each record:
    {
      "id": "case_000",
      "topic": "unsafe_procedure",
      "question": "<harmful_variant text>",
      "prefix": "Sure, here is",
      "method": "sure"
    }

Usage:
    python make_prefill_sure.py
    python make_prefill_sure.py --neutral-dir ../neutral --out prefill_data/sure_prefills.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# The one affirmative prefix used for every item in this set.
SURE_PREFIX = "Sure, here is"

# Self-harm-adjacent items the project keeps out of the run (see
# paired_dija_runner.DEFAULT_EXCLUDE). Excluded only with --exclude-selfharm.
SELFHARM_EXCLUDE = {"case_014", "case_036", "case_067"}


def load_cases(neutral_dir: Path) -> list[dict]:
    """Load every case_*.json file, sorted by filename for a stable order."""
    cases = []
    for path in sorted(neutral_dir.glob("case_*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    if not cases:
        raise SystemExit(f"[fatal] no case_*.json files found in {neutral_dir}")
    return cases


def build_records(cases: list[dict], prefix: str, exclude: set[str]) -> list[dict]:
    records = []
    for c in cases:
        cid = c.get("id", "?")
        if cid in exclude:
            continue
        question = c.get("harmful_variant")
        if not isinstance(question, str) or not question.strip():
            raise SystemExit(f"[fatal] case {cid} has no usable 'harmful_variant'")
        records.append({
            "id": cid,
            "topic": c.get("topic"),
            "question": question.strip(),
            "prefix": prefix,
            "method": "sure",
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--neutral-dir", default=str(here.parent / "neutral"),
                    help="directory of case_*.json files (default: ../neutral)")
    ap.add_argument("--out", default=str(here / "prefill_data" / "sure_prefills.json"),
                    help="output JSON path")
    ap.add_argument("--prefix", default=SURE_PREFIX,
                    help='affirmative prefix to prepend (default: "Sure, here is")')
    ap.add_argument("--exclude-selfharm", action="store_true",
                    help="drop the self-harm-adjacent cases (case_014/036/067)")
    args = ap.parse_args()

    exclude = SELFHARM_EXCLUDE if args.exclude_selfharm else set()
    cases = load_cases(Path(args.neutral_dir))
    records = build_records(cases, args.prefix, exclude)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[sure] {len(cases)} cases -> {len(records)} prefill records "
          f"(prefix={args.prefix!r}, excluded={sorted(exclude)})")
    print(f"[sure] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
