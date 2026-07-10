import os
import json
import torch
from collections import Counter
from cdg.judge import DeepSeekJudge

ROOT = "outputs_dream_native_history_full"
MANIFEST = f"{ROOT}/manifest.jsonl"

OUT_JSONL = "analysis_output/dream_B_existing_deepseek_judge.jsonl"
OUT_SUMMARY = "analysis_output/dream_B_existing_deepseek_summary.json"

os.makedirs("analysis_output", exist_ok=True)

print("START", flush=True)
print("has key:", bool(os.environ.get("DEEPSEEK_API_KEY")), flush=True)

rows = [json.loads(x) for x in open(MANIFEST)]
print("manifest rows:", len(rows), flush=True)

# Filter B from path first, so we do not load all 400 records
B_rows = []
for row in rows:
    path = row["path"]
    if "/B_harmful_injected/" in path or "B_harmful_injected" in path:
        B_rows.append(row)

print("B rows from manifest:", len(B_rows), flush=True)

judge = DeepSeekJudge(
    template="injection",
    model="deepseek-v4-flash",
    timeout=20,
    max_retries=1,
)

done = {}
if os.path.exists(OUT_JSONL):
    for line in open(OUT_JSONL):
        if line.strip():
            r = json.loads(line)
            done[r["case_id"]] = r

print("already judged:", len(done), flush=True)

with open(OUT_JSONL, "a") as f:
    for i, row in enumerate(B_rows, 1):
        rec = torch.load(row["path"], map_location="cpu")
        meta = rec["meta"]
        case_id = meta["case_id"]

        if case_id in done:
            print(f"[{i}/100] SKIP {case_id}", flush=True)
            continue

        behavior = meta.get("behavior", "")
        response = rec.get("response_text", "")

        print(f"[{i}/100] JUDGE {case_id} response_len={len(response.strip())}", flush=True)

        messages = [{"role": "user", "content": behavior}]
        jmeta = {
            **meta.get("case_meta", {}),
            "behavior": behavior,
            "variant": meta.get("variant"),
            "case_id": case_id,
        }

        j = judge.judge(messages, response, jmeta)

        out = {
            "case_id": case_id,
            "variant": meta["variant"],
            "path": row["path"],
            "response_len": len(response.strip()),
            "judge": j,
        }

        f.write(json.dumps(out, ensure_ascii=False) + "\n")
        f.flush()

items = [json.loads(x) for x in open(OUT_JSONL) if x.strip()]
items = [x for x in items if x["case_id"].startswith("B")]

valid = [x for x in items if x.get("judge", {}).get("success") is not None]
errors = [x for x in items if x.get("judge", {}).get("success") is None]

n = len(valid)
success = sum(int(x["judge"]["success"]) for x in valid)
labels = Counter(str(x["judge"].get("label")) for x in valid)

summary = {
    "model": "Dream-v0-Instruct-7B",
    "group": "B_harmful_injected",
    "judge_template": "injection",
    "judge_model": "deepseek-v4-flash",
    "n_total_B": len(B_rows),
    "n_judged_valid": n,
    "n_errors": len(errors),
    "success_count": success,
    "ASR": success / n if n else None,
    "label_counts": dict(labels),
}

with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n===== SUMMARY =====", flush=True)
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

print("\nSaved:", OUT_JSONL, flush=True)
print("Saved:", OUT_SUMMARY, flush=True)
