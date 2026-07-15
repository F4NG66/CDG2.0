#!/usr/bin/env python
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/path/to/rrae_steering_work_v2")

POOL = ROOT / "data/audit/A_candidate_pool_harmbench_advbench_all.csv"
EXTRA = ROOT / "data/source_extra/medsafetybench_extra_health_harmful.csv"

OUT_DIR = ROOT / "data/group_A"
AUDIT_DIR = ROOT / "data/audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_HEALTH = 400
TARGET_NON_HEALTH = 100
TARGET_TOTAL = 500

def normalize_prompt(s):
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

if not POOL.exists():
    raise FileNotFoundError(f"Missing previous candidate pool: {POOL}")

pool = pd.read_csv(POOL)

if not EXTRA.exists():
    raise FileNotFoundError(f"Missing MedSafetyBench extra file: {EXTRA}")

extra = pd.read_csv(EXTRA)

# Normalize extra columns to match pool.
for col in pool.columns:
    if col not in extra.columns:
        extra[col] = ""

extra["domain"] = "health_related"
extra["label"] = "harmful"
extra["variant"] = "clean"
extra["bucket"] = "A"
extra["source_dataset"] = "MedSafetyBench"

combined = pd.concat([pool, extra[pool.columns]], ignore_index=True)

# Dedup preserving source priority:
# current pool order = HarmBench then AdvBench from earlier script
# extra is appended only to fill missing health.
combined["_norm"] = combined["prompt"].map(normalize_prompt)
combined = combined[combined["_norm"].str.len() > 0].drop_duplicates("_norm", keep="first").copy()

health = combined[combined["domain"] == "health_related"].copy()
nonhealth = combined[combined["domain"] == "non_health_control"].copy()

if len(health) < TARGET_HEALTH or len(nonhealth) < TARGET_NON_HEALTH:
    summary = {
        "status": "FAILED",
        "target_health": TARGET_HEALTH,
        "available_health": int(len(health)),
        "target_nonhealth": TARGET_NON_HEALTH,
        "available_nonhealth": int(len(nonhealth)),
        "combined_source_counts": combined["source_dataset"].value_counts().to_dict(),
        "combined_domain_counts": combined["domain"].value_counts().to_dict(),
    }
    path = AUDIT_DIR / "A_FINAL_400_health_100_nonhealth_FAILED_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(2)

selected = pd.concat([
    health.head(TARGET_HEALTH),
    nonhealth.head(TARGET_NON_HEALTH),
], ignore_index=True)

selected = selected.drop(columns=["_norm"], errors="ignore").copy()
selected.insert(0, "id", [f"A_{i:04d}" for i in range(len(selected))])

# Hard checks
assert len(selected) == TARGET_TOTAL
assert int((selected["domain"] == "health_related").sum()) == TARGET_HEALTH
assert int((selected["domain"] == "non_health_control").sum()) == TARGET_NON_HEALTH
assert set(selected["bucket"]) == {"A"}
assert set(selected["label"]) == {"harmful"}
assert set(selected["variant"]) == {"clean"}

out_jsonl = OUT_DIR / "A_harmful_clean_FINAL_400_health_100_nonhealth.jsonl"
out_csv = OUT_DIR / "A_harmful_clean_FINAL_400_health_100_nonhealth.csv"

with out_jsonl.open("w", encoding="utf-8") as f:
    for rec in selected.to_dict(orient="records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

selected.to_csv(out_csv, index=False)

summary = {
    "status": "SUCCESS",
    "selected_total": int(len(selected)),
    "selected_health": int((selected["domain"] == "health_related").sum()),
    "selected_nonhealth": int((selected["domain"] == "non_health_control").sum()),
    "selected_source_counts": selected["source_dataset"].value_counts().to_dict(),
    "selected_health_source_counts": selected[selected["domain"] == "health_related"]["source_dataset"].value_counts().to_dict(),
    "selected_nonhealth_source_counts": selected[selected["domain"] == "non_health_control"]["source_dataset"].value_counts().to_dict(),
    "output_jsonl": str(out_jsonl),
    "output_csv": str(out_csv),
}

summary_path = AUDIT_DIR / "A_FINAL_400_health_100_nonhealth_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

review_path = AUDIT_DIR / "A_FINAL_manual_review_sample_100.csv"
pd.concat([
    selected[selected["domain"] == "health_related"].tail(60),
    selected[selected["domain"] == "non_health_control"].head(40),
], ignore_index=True).to_csv(review_path, index=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("manual_review_sample:", review_path)
