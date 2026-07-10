#!/usr/bin/env python3
import os, json, time
from collections import Counter, defaultdict

from cdg.judge import DeepSeekJudge
from cdg.data import load_cdg_root

MANIFEST = "outputs_p10_dream_outmask_L14_mid/manifest.jsonl"
PROMPT_ROOT = "prompts/cdg_injection"
OUT_JSONL = "analysis_output/judge_dream_p10_mid_B.jsonl"
OUT_SUMMARY = "analysis_output/judge_dream_p10_mid_B_summary.json"

if not os.environ.get("DEEPSEEK_API_KEY"):
    raise SystemExit("ERROR: DEEPSEEK_API_KEY is not set in this shell")

os.makedirs("analysis_output", exist_ok=True)

# Load original cases so we can recover the behavior text.
cases = load_cdg_root(PROMPT_ROOT)
case_by_id = {c.case_id: c for c in cases}

rows = [json.loads(x) for x in open(MANIFEST) if x.strip()]
targets = []
for r in rows:
    variant = r.get("variant", "")
    if not variant.startswith("B"):
        continue
    alpha = float(r.get("steer_alpha"))
    if alpha not in {0.0, 2.0, 4.0}:
        continue
    targets.append(r)

targets = sorted(targets, key=lambda r: (float(r.get("steer_alpha")), r.get("case_id", "")))

done = set()
if os.path.exists(OUT_JSONL):
    for line in open(OUT_JSONL):
        if not line.strip():
            continue
        x = json.loads(line)
        done.add((float(x["alpha"]), x["case_id"]))

print("targets:", len(targets))
print("already done:", len(done))
print("remaining:", sum((float(r.get("steer_alpha")), r.get("case_id")) not in done for r in targets))

judge = DeepSeekJudge(template="injection", model="deepseek-v4-flash")

with open(OUT_JSONL, "a") as f:
    for i, r in enumerate(targets, 1):
        alpha = float(r.get("steer_alpha"))
        case_id = r.get("case_id")
        key = (alpha, case_id)
        if key in done:
            continue

        case = case_by_id.get(case_id)
        if case is None:
            print("MISSING CASE:", case_id)
            continue

        behavior = case.behavior
        response = r.get("response_text") or ""

        meta = dict(case.meta or {})
        meta.update({
            "behavior": behavior,
            "case_id": case_id,
            "variant": r.get("variant"),
            "alpha": alpha,
            "path": r.get("path"),
        })

        try:
            j = judge.judge(
                [{"role": "user", "content": behavior}],
                response,
                meta,
            )
            out = {
                "case_id": case_id,
                "variant": r.get("variant"),
                "alpha": alpha,
                "path": r.get("path"),
                "response_len": len(response),
                "judge": j,
            }
        except Exception as e:
            out = {
                "case_id": case_id,
                "variant": r.get("variant"),
                "alpha": alpha,
                "path": r.get("path"),
                "response_len": len(response),
                "error": repr(e),
            }

        f.write(json.dumps(out, ensure_ascii=False) + "\n")
        f.flush()

        label = (out.get("judge") or {}).get("label", "ERR")
        success = (out.get("judge") or {}).get("success", "ERR")
        print(f"[{i}/{len(targets)}] alpha={alpha} {case_id} success={success} label={label}")

        time.sleep(0.2)

# Summarize.
items = [json.loads(x) for x in open(OUT_JSONL) if x.strip()]
by_alpha = defaultdict(list)
for x in items:
    if "judge" in x:
        by_alpha[float(x["alpha"])].append(x)

summary = {}
for alpha, xs in sorted(by_alpha.items()):
    success = [int((x["judge"] or {}).get("success", 0)) for x in xs]
    labels = Counter((x["judge"] or {}).get("label", "unknown") for x in xs)
    summary[str(alpha)] = {
        "n": len(xs),
        "success_count": sum(success),
        "ASR": sum(success) / len(success) if success else None,
        "label_counts": dict(labels),
        "median_response_len": sorted([x.get("response_len", 0) for x in xs])[len(xs)//2] if xs else None,
    }

base = summary.get("0.0", {}).get("ASR")
if base is not None and base > 0:
    for k, v in summary.items():
        asr = v.get("ASR")
        if asr is not None:
            v["relative_drop_vs_alpha0"] = (base - asr) / base

with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)

print("\nSUMMARY")
print(json.dumps(summary, indent=2))
print("\nSaved:")
print(" ", OUT_JSONL)
print(" ", OUT_SUMMARY)
