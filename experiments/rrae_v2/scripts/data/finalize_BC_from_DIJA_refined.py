#!/usr/bin/env python
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/path/to/rrae_steering_work_v2")

A_PATH = ROOT / "data/group_A/A_harmful_clean_FINAL_400_health_100_nonhealth.jsonl"
D_PATH = ROOT / "data/group_D/D_benign_clean_deepseek_chat_FULL_500.jsonl"

B_RAW = ROOT / "data/group_B/B_harmful_injected_DIJA_refined_Qwen_raw.json"
C_RAW = ROOT / "data/group_C/C_benign_injected_DIJA_refined_Qwen_raw.json"

B_JSONL = ROOT / "data/group_B/B_harmful_injected_DIJA_refined_Qwen.jsonl"
B_CSV = ROOT / "data/group_B/B_harmful_injected_DIJA_refined_Qwen.csv"

C_JSONL = ROOT / "data/group_C/C_benign_injected_DIJA_refined_Qwen.jsonl"
C_CSV = ROOT / "data/group_C/C_benign_injected_DIJA_refined_Qwen.csv"

ABCD_JSONL = ROOT / "data/abcd/ABCD_clean_v2_DIJA_Qwen_A_B_C_D.jsonl"
ABCD_CSV = ROOT / "data/abcd/ABCD_clean_v2_DIJA_Qwen_A_B_C_D.csv"

AUDIT_JSON = ROOT / "data/audit/BC_DIJA_Qwen_finalize_summary.json"

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_raw(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing raw DIJA output: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data

def mask_stats(text):
    nums = [int(x) for x in re.findall(r"<mask:(\d+)>", str(text or ""))]
    return {
        "mask_segment_count": len(nums),
        "mask_token_total_declared": sum(nums),
        "mask_lengths": nums,
    }

def clean_refined(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    return text

A = read_jsonl(A_PATH)
D = read_jsonl(D_PATH)

assert len(A) == 500, f"A rows expected 500, got {len(A)}"
assert len(D) == 500, f"D rows expected 500, got {len(D)}"

A_by_id = {r["id"]: r for r in A}
D_by_id = {r["id"]: r for r in D}
D_by_A = {r["paired_A_id"]: r for r in D}

assert len(A_by_id) == 500
assert len(D_by_id) == 500
assert len(D_by_A) == 500
assert set(A_by_id.keys()) == set(D_by_A.keys())

B_raw = load_raw(B_RAW)
C_raw = load_raw(C_RAW)

assert len(B_raw) == 500, f"B raw expected 500, got {len(B_raw)}"
assert len(C_raw) == 500, f"C raw expected 500, got {len(C_raw)}"

B = []
for i, r in enumerate(B_raw):
    a_id = r["BehaviorID"]
    a = A_by_id[a_id]
    d = D_by_A[a_id]
    prompt = clean_refined(r.get("Refined_behavior", ""))
    st = mask_stats(prompt)

    B.append({
        "id": f"B_{i:04d}",
        "paired_A_id": a_id,
        "paired_D_id": d["id"],
        "bucket": "B",
        "label": "harmful",
        "variant": "injected",
        "injection_type": "DIJA_Qwen_refined_maskN",
        "domain": a.get("domain", ""),
        "prompt": prompt,
        "clean_prompt": a["prompt"],
        "source_A_dataset": a.get("source_dataset", ""),
        "source_A_id": a.get("source_id", ""),
        "dija_behavior_id": r.get("BehaviorID", ""),
        "dija_functional_category": r.get("FunctionalCategory", ""),
        "dija_semantic_category": r.get("SemanticCategory", ""),
        "mask_segment_count": st["mask_segment_count"],
        "mask_token_total_declared": st["mask_token_total_declared"],
        "mask_lengths": st["mask_lengths"],
        "needs_manual_audit": True,
    })

C = []
for i, r in enumerate(C_raw):
    d_id = r["BehaviorID"]
    d = D_by_id[d_id]
    a = A_by_id[d["paired_A_id"]]
    prompt = clean_refined(r.get("Refined_behavior", ""))
    st = mask_stats(prompt)

    C.append({
        "id": f"C_{i:04d}",
        "paired_A_id": a["id"],
        "paired_D_id": d_id,
        "bucket": "C",
        "label": "benign",
        "variant": "injected",
        "injection_type": "DIJA_Qwen_refined_maskN",
        "domain": d.get("domain", ""),
        "prompt": prompt,
        "clean_prompt": d["prompt"],
        "source_D_generation_model": d.get("generation_model", ""),
        "source_A_dataset": d.get("source_A_dataset", ""),
        "source_A_id": d.get("source_A_id", ""),
        "dija_behavior_id": r.get("BehaviorID", ""),
        "dija_functional_category": r.get("FunctionalCategory", ""),
        "dija_semantic_category": r.get("SemanticCategory", ""),
        "mask_segment_count": st["mask_segment_count"],
        "mask_token_total_declared": st["mask_token_total_declared"],
        "mask_lengths": st["mask_lengths"],
        "needs_manual_audit": True,
    })

B_df = pd.DataFrame(B)
C_df = pd.DataFrame(C)

assert len(B_df) == 500
assert len(C_df) == 500
assert B_df["id"].nunique() == 500
assert C_df["id"].nunique() == 500
assert B_df["paired_A_id"].nunique() == 500
assert C_df["paired_A_id"].nunique() == 500
assert set(B_df["paired_A_id"]) == set(A_by_id.keys())
assert set(C_df["paired_A_id"]) == set(A_by_id.keys())

B_no_masks = int((B_df["mask_segment_count"] == 0).sum())
C_no_masks = int((C_df["mask_segment_count"] == 0).sum())

write_jsonl(B, B_JSONL)
write_jsonl(C, C_JSONL)
B_df.to_csv(B_CSV, index=False)
C_df.to_csv(C_CSV, index=False)

ABCD = []

for r in A:
    ABCD.append({
        "id": r["id"],
        "bucket": "A",
        "label": "harmful",
        "variant": "clean",
        "injection_type": "none",
        "domain": r.get("domain", ""),
        "prompt": r["prompt"],
        "paired_A_id": r["id"],
        "paired_D_id": "",
        "clean_prompt": r["prompt"],
        "source_dataset": r.get("source_dataset", ""),
        "needs_manual_audit": r.get("needs_manual_audit", True),
    })

for r in B:
    ABCD.append({
        "id": r["id"],
        "bucket": "B",
        "label": "harmful",
        "variant": "injected",
        "injection_type": r["injection_type"],
        "domain": r["domain"],
        "prompt": r["prompt"],
        "paired_A_id": r["paired_A_id"],
        "paired_D_id": r["paired_D_id"],
        "clean_prompt": r["clean_prompt"],
        "source_dataset": r.get("source_A_dataset", ""),
        "needs_manual_audit": r.get("needs_manual_audit", True),
    })

for r in C:
    ABCD.append({
        "id": r["id"],
        "bucket": "C",
        "label": "benign",
        "variant": "injected",
        "injection_type": r["injection_type"],
        "domain": r["domain"],
        "prompt": r["prompt"],
        "paired_A_id": r["paired_A_id"],
        "paired_D_id": r["paired_D_id"],
        "clean_prompt": r["clean_prompt"],
        "source_dataset": r.get("source_A_dataset", ""),
        "needs_manual_audit": r.get("needs_manual_audit", True),
    })

for r in D:
    ABCD.append({
        "id": r["id"],
        "bucket": "D",
        "label": "benign",
        "variant": "clean",
        "injection_type": "none",
        "domain": r.get("domain", ""),
        "prompt": r["prompt"],
        "paired_A_id": r["paired_A_id"],
        "paired_D_id": r["id"],
        "clean_prompt": r["prompt"],
        "source_dataset": r.get("source_A_dataset", ""),
        "needs_manual_audit": r.get("needs_manual_audit", True),
    })

ABCD_df = pd.DataFrame(ABCD)
write_jsonl(ABCD, ABCD_JSONL)
ABCD_df.to_csv(ABCD_CSV, index=False)

assert len(ABCD_df) == 2000
assert ABCD_df["id"].nunique() == 2000
assert set(ABCD_df["bucket"]) == {"A", "B", "C", "D"}

summary = {
    "B_rows": len(B),
    "C_rows": len(C),
    "ABCD_rows": len(ABCD),
    "B_no_mask_rows": B_no_masks,
    "C_no_mask_rows": C_no_masks,
    "B_mask_segment_count_counts": {
        str(k): int(v) for k, v in B_df["mask_segment_count"].value_counts().sort_index().items()
    },
    "C_mask_segment_count_counts": {
        str(k): int(v) for k, v in C_df["mask_segment_count"].value_counts().sort_index().items()
    },
    "B_mask_token_total_declared_min": int(B_df["mask_token_total_declared"].min()),
    "B_mask_token_total_declared_max": int(B_df["mask_token_total_declared"].max()),
    "C_mask_token_total_declared_min": int(C_df["mask_token_total_declared"].min()),
    "C_mask_token_total_declared_max": int(C_df["mask_token_total_declared"].max()),
    "bucket_counts": {
        str(k): int(v) for k, v in ABCD_df["bucket"].value_counts().sort_index().items()
    },
    "outputs": {
        "B_jsonl": str(B_JSONL),
        "B_csv": str(B_CSV),
        "C_jsonl": str(C_JSONL),
        "C_csv": str(C_CSV),
        "ABCD_jsonl": str(ABCD_JSONL),
        "ABCD_csv": str(ABCD_CSV),
    }
}

AUDIT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
