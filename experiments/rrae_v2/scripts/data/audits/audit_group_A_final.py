#!/usr/bin/env python
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/path/to/rrae_steering_work_v2")
CSV = ROOT / "data/group_A/A_harmful_clean_FINAL_400_health_100_nonhealth.csv"
AUDIT = ROOT / "data/audit"

AUDIT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

def norm(s):
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

df["_prompt_norm"] = df["prompt"].map(norm)

checks = {
    "rows": int(len(df)),
    "unique_ids": int(df["id"].nunique()),
    "duplicate_ids": int(df["id"].duplicated().sum()),
    "unique_prompts_norm": int(df["_prompt_norm"].nunique()),
    "duplicate_prompts_norm": int(df["_prompt_norm"].duplicated().sum()),
    "bucket_counts": df["bucket"].value_counts().to_dict(),
    "label_counts": df["label"].value_counts().to_dict(),
    "variant_counts": df["variant"].value_counts().to_dict(),
    "domain_counts": df["domain"].value_counts().to_dict(),
    "source_counts": df["source_dataset"].value_counts().to_dict(),
    "source_x_domain": pd.crosstab(df["source_dataset"], df["domain"]).to_dict(),
    "min_prompt_len": int(df["prompt"].astype(str).str.len().min()),
    "max_prompt_len": int(df["prompt"].astype(str).str.len().max()),
    "mean_prompt_len": float(df["prompt"].astype(str).str.len().mean()),
}

hard_pass = (
    checks["rows"] == 500
    and checks["duplicate_ids"] == 0
    and checks["duplicate_prompts_norm"] == 0
    and checks["bucket_counts"] == {"A": 500}
    and checks["label_counts"] == {"harmful": 500}
    and checks["variant_counts"] == {"clean": 500}
    and checks["domain_counts"].get("health_related", 0) == 400
    and checks["domain_counts"].get("non_health_control", 0) == 100
)

checks["hard_pass"] = hard_pass

out_json = AUDIT / "A_FINAL_quality_audit.json"
out_json.write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")

# Save possible suspicious rows for manual inspection.
suspicious = df[
    (df["prompt"].astype(str).str.len() < 20)
    | (df["prompt"].astype(str).str.len() > 1000)
].drop(columns=["_prompt_norm"], errors="ignore")

suspicious.to_csv(AUDIT / "A_FINAL_suspicious_length_rows.csv", index=False)

# Stratified manual sample: 40 health + 40 non-health + all MedSafetyBench if not too many.
sample_parts = [
    df[df["domain"] == "health_related"].head(40),
    df[df["domain"] == "non_health_control"].head(40),
    df[df["source_dataset"] == "MedSafetyBench"].head(80),
]
sample = pd.concat(sample_parts, ignore_index=True).drop_duplicates("id")
sample.drop(columns=["_prompt_norm"], errors="ignore").to_csv(
    AUDIT / "A_FINAL_manual_review_stratified.csv",
    index=False
)

print(json.dumps(checks, indent=2, ensure_ascii=False))
print()
print("wrote:", out_json)
print("manual review:", AUDIT / "A_FINAL_manual_review_stratified.csv")
print("suspicious length rows:", AUDIT / "A_FINAL_suspicious_length_rows.csv")

if not hard_pass:
    raise SystemExit("A quality audit failed. Do not proceed.")
