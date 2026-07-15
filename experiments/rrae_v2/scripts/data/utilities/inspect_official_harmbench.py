#!/usr/bin/env python
from pathlib import Path
import pandas as pd

ROOT = Path("/path/to/rrae_steering_work_v2")
p = ROOT / "data/source/harmbench_official_behaviors_text_all.csv"

df = pd.read_csv(p)

print("===== basic =====")
print("path:", p)
print("shape:", df.shape)
print("columns:", list(df.columns))

print("\n===== object column value counts =====")
for c in df.columns:
    if df[c].dtype == "object":
        vc = df[c].value_counts(dropna=False).head(30)
        print(f"\n--- {c} ---")
        print(vc)

print("\n===== possible category columns =====")
for c in df.columns:
    lc = c.lower()
    if "category" in lc or "type" in lc or "tag" in lc:
        print("\n", c)
        print(df[c].value_counts(dropna=False))

# health-ish keyword scan
keywords = [
    "health", "medical", "medicine", "clinical", "doctor", "patient",
    "hospital", "prescription", "dose", "dosage", "drug", "medication",
    "diagnosis", "treatment", "disease", "symptom", "vaccine",
    "poison", "toxin", "toxic", "ricin", "overdose", "insulin",
    "diabetes", "blood", "lab", "opioid", "antibiotic"
]

text_cols = [c for c in df.columns if df[c].dtype == "object"]
mask = pd.Series(False, index=df.index)

for c in text_cols:
    s = df[c].fillna("").astype(str).str.lower()
    for kw in keywords:
        mask |= s.str.contains(kw, regex=False)

print("\n===== health-ish keyword matches =====")
print("matches:", int(mask.sum()), "/", len(df))

show_cols = []
for c in ["BehaviorID", "Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString"]:
    if c in df.columns:
        show_cols.append(c)

if not show_cols:
    show_cols = text_cols[:5]

out = ROOT / "data/audit/harmbench_official_health_keyword_matches.csv"
df.loc[mask, show_cols].to_csv(out, index=False)
print("wrote:", out)

print("\n===== first 30 health-ish rows metadata =====")
print(df.loc[mask, show_cols].head(30).to_string(index=False))
