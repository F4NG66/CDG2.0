#!/usr/bin/env python
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

RRAE_DATA = "/path/to/rrae_data"
if RRAE_DATA not in sys.path:
    sys.path.insert(0, RRAE_DATA)

# Exact judge used in SAE/CDG pipeline
from cdg.judge import DeepSeekJudge


DEFAULT_CONFIGS = [
    "custom_template_mask",
    "custom_template_mask_plus_output_mask",
    "custom_template_all",
    "custom_full_all",
]


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(r, judge_template):
    return (
        r.get("position_config"),
        r.get("case_id"),
        r.get("variant"),
        r.get("group"),
        str(r.get("alpha")),
        r.get("mode"),
        str(r.get("seed")),
        judge_template,
    )


def load_existing(path):
    done = set()
    rows = []

    if not path.exists():
        return done, rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done.add(tuple(r["row_key"]))

    return done, rows


def select_rows(rows, configs, groups, sample_per_bucket):
    configs = set(configs)
    groups = set(groups)

    filtered = []
    for r in rows:
        if r.get("position_config") not in configs:
            continue
        if r.get("group") not in groups:
            continue
        if r.get("error"):
            continue
        if not (r.get("output") or "").strip():
            continue
        filtered.append(r)

    if sample_per_bucket <= 0:
        return filtered

    buckets = defaultdict(list)
    for r in filtered:
        k = (
            r.get("position_config"),
            r.get("group"),
            r.get("mode"),
            str(r.get("alpha")),
        )
        buckets[k].append(r)

    selected = []
    for k, bucket in sorted(buckets.items()):
        bucket = sorted(bucket, key=lambda x: (x.get("case_id"), x.get("variant")))
        selected.extend(bucket[:sample_per_bucket])

    return selected


def summarize(judged_rows, out_dir, judge_template):
    records = []

    for r in judged_rows:
        records.append({
            "position_config": r["position_config"],
            "group": r["group"],
            "mode": r["mode"],
            "alpha": float(r["alpha"]),
            "judge_template": r["judge_template"],
            "success": r.get("success"),
            "judge_error": 1 if r.get("success") is None else 0,
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("No judged rows to summarize.")
        return

    summary = (
        df.groupby(["position_config", "group", "mode", "alpha", "judge_template"], as_index=False)
        .agg(
            n=("position_config", "count"),
            success_rate=("success", "mean"),
            judge_error_rate=("judge_error", "mean"),
        )
        .sort_values(["position_config", "group", "mode", "alpha"])
    )

    out_path = Path(out_dir) / f"summary__judge_{judge_template}.csv"
    summary.to_csv(out_path, index=False)

    print("saved summary:", out_path)
    print(summary.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--groups", nargs="+", default=["B"])

    ap.add_argument("--judge-template", default="injection")
    ap.add_argument("--judge-model", default="deepseek-v4-flash")

    ap.add_argument("--sample-per-bucket", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / f"judged__template_{args.judge_template}.jsonl"

    rows = load_rows(args.input)
    rows = select_rows(
        rows,
        configs=args.configs,
        groups=args.groups,
        sample_per_bucket=args.sample_per_bucket,
    )

    if args.max_rows > 0:
        rows = rows[:args.max_rows]

    print("rows selected:", len(rows), flush=True)
    print("judge_template:", args.judge_template, flush=True)
    print("judge_model:", args.judge_model, flush=True)
    print("configs:", args.configs, flush=True)
    print("groups:", args.groups, flush=True)

    done, judged_rows = load_existing(out_jsonl)
    print("resume done:", len(done), flush=True)

    judge = DeepSeekJudge(
        template=args.judge_template,
        model=args.judge_model,
    )

    with out_jsonl.open("a", encoding="utf-8") as out:
        for i, r in enumerate(rows, 1):
            k = row_key(r, args.judge_template)
            if k in done:
                continue

            behavior = r.get("behavior", "")
            output = r.get("output", "")

            meta = {
                "case_id": r.get("case_id"),
                "variant": r.get("variant"),
                "group": r.get("group"),
                "behavior": behavior,
                "position_config": r.get("position_config"),
                "mode": r.get("mode"),
                "alpha": r.get("alpha"),
            }

            print(
                f"[{i}/{len(rows)}] {r.get('position_config')} "
                f"{r.get('group')} {r.get('case_id')} "
                f"mode={r.get('mode')} alpha={r.get('alpha')}",
                flush=True,
            )

            # Same calling style as SAE/CDG:
            # judge.judge(messages, response, meta)
            result = judge.judge(
                [{"role": "user", "content": behavior}],
                output,
                meta,
            )

            jr = {
                "row_key": list(k),
                "position_config": r.get("position_config"),
                "case_id": r.get("case_id"),
                "variant": r.get("variant"),
                "group": r.get("group"),
                "mode": r.get("mode"),
                "alpha": r.get("alpha"),
                "seed": r.get("seed"),
                "judge_template": args.judge_template,
                "judge_model": args.judge_model,
                "generation_error": r.get("error"),
                "output_chars": len(output),
                "success": result.get("success"),
                "label": result.get("label"),
                "reason": result.get("reason"),
                "raw": result.get("raw"),
                "judge_return": result,
            }

            out.write(json.dumps(jr, ensure_ascii=False) + "\n")
            out.flush()

            done.add(k)
            judged_rows.append(jr)

    summarize(judged_rows, out_dir, args.judge_template)


if __name__ == "__main__":
    main()
