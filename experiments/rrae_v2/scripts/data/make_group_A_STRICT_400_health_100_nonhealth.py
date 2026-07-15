#!/usr/bin/env python
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/path/to/rrae_steering_work_v2")

HB = ROOT / "data/source/harmbench_official_behaviors_text_all.csv"
ADV = ROOT / "data/source/advbench_harmful_behaviors.csv"

OUT_DIR = ROOT / "data/group_A"
AUDIT_DIR = ROOT / "data/audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_HEALTH = 400
TARGET_NON_HEALTH = 100
TARGET_TOTAL = TARGET_HEALTH + TARGET_NON_HEALTH

# Strict but broad health/medical-related keywords.
# This is for DOMAIN TAGGING ONLY, not for saying the prompt is safe/unsafe.
HEALTH_KEYWORDS = [
    # direct medical
    "health", "medical", "medicine", "clinical", "clinic", "doctor", "physician",
    "nurse", "hospital", "patient", "pharmacy", "pharmacist", "prescription",
    "medical record", "medical records", "diagnosis", "diagnose", "treatment",
    "symptom", "disease", "infection", "surgery", "surgical", "therapy",

    # drugs/pharma
    "drug", "drugs", "medication", "medications", "dose", "dosage",
    "pill", "tablet", "capsule", "opioid", "opioids", "fentanyl", "morphine",
    "heroin", "codeine", "oxycodone", "cocaine", "lsd", "methamphetamine",
    "pharmaceutical", "overdose", "antibiotic", "insulin",

    # public health / misinformation
    "vaccine", "vaccination", "covid", "covid-19", "pandemic", "mask",
    "cancer", "autism", "side effects", "adverse effects",

    # bio/chemical health harm
    "poison", "poisoning", "toxin", "toxic", "ricin", "botulinum",
    "bacteria", "bacterial", "virus", "viral", "e.coli", "ecoli",
    "antibiotic resistance", "gene synthesis", "dna sequence",
    "biohazard", "biological", "pathogen",

    # devices/labs
    "pacemaker", "iv", "blood", "lab", "laboratory", "bsl-4", "bsl4",
]

COPYRIGHT_CATEGORIES = {"copyright"}

def normalize_prompt(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def is_health_related_text(*parts) -> bool:
    text = " ".join(str(p or "") for p in parts).lower()
    return any(k in text for k in HEALTH_KEYWORDS)

def add_record(records, seen, rec):
    key = normalize_prompt(rec["prompt"])
    if not key:
        return False
    if key in seen:
        return False
    seen.add(key)
    records.append(rec)
    return True

def load_harmbench():
    if not HB.exists():
        raise FileNotFoundError(f"Missing HarmBench source: {HB}")

    df = pd.read_csv(HB)
    out = []

    for _, row in df.iterrows():
        sem = str(row.get("SemanticCategory", "") or "")
        if sem in COPYRIGHT_CATEGORIES:
            continue

        prompt = str(row.get("Behavior", "") or "").strip()
        if not prompt:
            continue

        domain = "health_related" if is_health_related_text(
            row.get("Behavior", ""),
            row.get("BehaviorID", ""),
            row.get("SemanticCategory", ""),
            row.get("FunctionalCategory", ""),
            row.get("ContextString", "")
        ) else "non_health_control"

        out.append({
            "bucket": "A",
            "label": "harmful",
            "variant": "clean",
            "domain": domain,
            "prompt": prompt,
            "source_dataset": "HarmBench_official",
            "source_file": str(HB),
            "source_id": str(row.get("BehaviorID", "") or ""),
            "functional_category": str(row.get("FunctionalCategory", "") or ""),
            "semantic_category": sem,
            "context_string": "" if pd.isna(row.get("ContextString", "")) else str(row.get("ContextString", "")),
            "needs_manual_audit": True,
        })

    return out

def load_advbench():
    if not ADV.exists():
        raise FileNotFoundError(
            f"Missing AdvBench source: {ADV}\n"
            "Download it first with:\n"
            "wget -O data/source/advbench_harmful_behaviors.csv "
            "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
        )

    df = pd.read_csv(ADV)

    if "goal" in df.columns:
        prompt_col = "goal"
    elif "prompt" in df.columns:
        prompt_col = "prompt"
    elif "Behavior" in df.columns:
        prompt_col = "Behavior"
    else:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not object_cols:
            raise RuntimeError("Could not find a text column in AdvBench.")
        prompt_col = max(object_cols, key=lambda c: df[c].fillna("").astype(str).str.len().mean())

    out = []
    for i, row in df.iterrows():
        prompt = str(row.get(prompt_col, "") or "").strip()
        if not prompt:
            continue

        domain = "health_related" if is_health_related_text(prompt) else "non_health_control"

        out.append({
            "bucket": "A",
            "label": "harmful",
            "variant": "clean",
            "domain": domain,
            "prompt": prompt,
            "source_dataset": "AdvBench",
            "source_file": str(ADV),
            "source_id": f"advbench_row_{i}",
            "functional_category": "",
            "semantic_category": "",
            "context_string": "",
            "needs_manual_audit": True,
        })

    return out, prompt_col

seen = set()
all_records = []

hb_records = load_harmbench()
adv_records, adv_prompt_col = load_advbench()

# Priority order:
# 1. HarmBench health
# 2. AdvBench health
# 3. HarmBench non-health
# 4. AdvBench non-health
#
# But final output is STRICT:
# exactly 400 health_related and exactly 100 non_health_control.
health_candidates = [
    r for r in hb_records if r["domain"] == "health_related"
] + [
    r for r in adv_records if r["domain"] == "health_related"
]

nonhealth_candidates = [
    r for r in hb_records if r["domain"] == "non_health_control"
] + [
    r for r in adv_records if r["domain"] == "non_health_control"
]

selected_health = []
selected_nonhealth = []
seen_health = set()
seen_nonhealth = set()

for r in health_candidates:
    if len(selected_health) >= TARGET_HEALTH:
        break
    add_record(selected_health, seen_health, r)

for r in nonhealth_candidates:
    if len(selected_nonhealth) >= TARGET_NON_HEALTH:
        break
    add_record(selected_nonhealth, seen_nonhealth, r)

available_summary = {
    "target_health_related": TARGET_HEALTH,
    "target_non_health_control": TARGET_NON_HEALTH,
    "available_health_related_after_dedup": len(selected_health),
    "available_non_health_control_after_dedup": len(selected_nonhealth),
    "harmbench_total_noncopyright": len(hb_records),
    "harmbench_health_related": sum(r["domain"] == "health_related" for r in hb_records),
    "harmbench_non_health_control": sum(r["domain"] == "non_health_control" for r in hb_records),
    "advbench_total": len(adv_records),
    "advbench_health_related": sum(r["domain"] == "health_related" for r in adv_records),
    "advbench_non_health_control": sum(r["domain"] == "non_health_control" for r in adv_records),
    "advbench_prompt_column_used": adv_prompt_col,
}

# Save candidate pool audit even if strict build fails.
pool_df = pd.DataFrame(hb_records + adv_records)
pool_path = AUDIT_DIR / "A_candidate_pool_harmbench_advbench_all.csv"
pool_df.to_csv(pool_path, index=False)

available_summary["candidate_pool_csv"] = str(pool_path)

summary_path = AUDIT_DIR / "A_STRICT_400_health_100_nonhealth_availability_summary.json"
summary_path.write_text(json.dumps(available_summary, indent=2, ensure_ascii=False), encoding="utf-8")

if len(selected_health) < TARGET_HEALTH or len(selected_nonhealth) < TARGET_NON_HEALTH:
    print("===== STRICT BUILD FAILED: not enough examples =====")
    print(json.dumps(available_summary, indent=2, ensure_ascii=False))
    print()
    print("No final Group A file was written because the strict 400/100 requirement was not met.")
    print("Check this audit file:")
    print(summary_path)
    print("Candidate pool:")
    print(pool_path)
    raise SystemExit(2)

final_records = selected_health + selected_nonhealth

# Global dedup one more time across final records.
final = []
seen_final = set()
for r in final_records:
    add_record(final, seen_final, r)

if len(final) != TARGET_TOTAL:
    raise RuntimeError(f"Final dedup changed size to {len(final)}; expected {TARGET_TOTAL}.")

for i, r in enumerate(final):
    r["id"] = f"A_{i:04d}"

out_df = pd.DataFrame(final)

# Hard assertions.
assert len(out_df) == TARGET_TOTAL
assert int((out_df["domain"] == "health_related").sum()) == TARGET_HEALTH
assert int((out_df["domain"] == "non_health_control").sum()) == TARGET_NON_HEALTH
assert set(out_df["bucket"]) == {"A"}
assert set(out_df["label"]) == {"harmful"}
assert set(out_df["variant"]) == {"clean"}

out_jsonl = OUT_DIR / "A_harmful_clean_STRICT_400_health_100_nonhealth.jsonl"
out_csv = OUT_DIR / "A_harmful_clean_STRICT_400_health_100_nonhealth.csv"

with out_jsonl.open("w", encoding="utf-8") as f:
    for rec in out_df.to_dict(orient="records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

out_df.to_csv(out_csv, index=False)

final_summary = {
    **available_summary,
    "selected_total": int(len(out_df)),
    "selected_health_related": int((out_df["domain"] == "health_related").sum()),
    "selected_non_health_control": int((out_df["domain"] == "non_health_control").sum()),
    "selected_source_counts": out_df["source_dataset"].value_counts().to_dict(),
    "selected_health_source_counts": out_df[out_df["domain"] == "health_related"]["source_dataset"].value_counts().to_dict(),
    "selected_nonhealth_source_counts": out_df[out_df["domain"] == "non_health_control"]["source_dataset"].value_counts().to_dict(),
    "output_jsonl": str(out_jsonl),
    "output_csv": str(out_csv),
}

final_summary_path = AUDIT_DIR / "A_STRICT_400_health_100_nonhealth_FINAL_summary.json"
final_summary_path.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")

review_path = AUDIT_DIR / "A_STRICT_manual_review_sample_100.csv"
pd.concat([
    out_df[out_df["domain"] == "health_related"].head(50),
    out_df[out_df["domain"] == "non_health_control"].head(50),
], ignore_index=True).to_csv(review_path, index=False)

print("===== STRICT BUILD SUCCESS =====")
print(json.dumps(final_summary, indent=2, ensure_ascii=False))
print()
print("Manual review sample:", review_path)
