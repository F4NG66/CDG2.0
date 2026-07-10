import os
import json
import numpy as np
import torch

ROOT = "outputs_dream_native_history_full"
MANIFEST = os.path.join(ROOT, "manifest.jsonl")
JUDGE_JSONL = "analysis_output/dream_B_existing_deepseek_judge.jsonl"

OUT_JSON = "analysis_output/dream_success_direction_cosine.json"

os.makedirs("analysis_output", exist_ok=True)

POINTS = [
    # best behavioral-success probe points
    ("tpl_mask", 0.1, 5),
    ("out_mask", 0.05, 5),
    ("out_mask", 0.35, 5),
    ("tpl_mask", 0.35, 5),
    ("out_mask", 0.2, 5),
    ("out_mask", 0.5, 5),
    ("harm", 0.2, 5),
    ("harm", 0.05, 5),

    # best shared-injection geometry points
    ("out_mask", 0.05, 14),
    ("out_mask", 0.05, 23),
    ("harm", 0.05, 23),
    ("out_mask", 0.05, 5),
]

def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return float("nan")

    return float(np.dot(a, b) / denom)

def normalize(v):
    n = np.linalg.norm(v)

    if n == 0:
        return v

    return v / n

def get_vec(rec, scope, frac, layer):
    count = rec["counts"].get(scope, {}).get(frac, 0)

    if count == 0:
        return None

    try:
        v = rec["hidden"][scope][frac][layer]
    except KeyError:
        return None

    v = v.float()

    if not torch.isfinite(v).all():
        return None

    return v.numpy().astype(np.float32)

# Load B judge labels
B_success = {}

for line in open(JUDGE_JSONL):
    if not line.strip():
        continue

    item = json.loads(line)
    cid = item["case_id"]

    if not cid.startswith("B"):
        continue

    s = item.get("judge", {}).get("success", None)

    if s is None:
        continue

    B_success[cid] = int(s)

print("B judge labels:", len(B_success), "success:", sum(B_success.values()), "refuse:", len(B_success) - sum(B_success.values()), flush=True)

# Load records
rows = [json.loads(x) for x in open(MANIFEST)]
records = []

for row in rows:
    rec = torch.load(row["path"], map_location="cpu")
    meta = rec["meta"]
    cid = meta["case_id"]

    records.append({
        "case_id": cid,
        "group": cid[0],
        "hidden": rec["hidden"],
        "counts": rec["counts"],
    })

print("records:", len(records), flush=True)

results = []

seen = set()
for scope, frac, layer in POINTS:
    key = (scope, frac, layer)
    if key in seen:
        continue
    seen.add(key)

    group_vecs = {g: [] for g in ["A", "B", "C", "D"]}
    succ_vecs = []
    ref_vecs = []

    for rec in records:
        v = get_vec(rec, scope, frac, layer)

        if v is None:
            continue

        g = rec["group"]

        if g in group_vecs:
            group_vecs[g].append(v)

        if g == "B" and rec["case_id"] in B_success:
            if B_success[rec["case_id"]] == 1:
                succ_vecs.append(v)
            else:
                ref_vecs.append(v)

    needed = ["A", "B", "C", "D"]

    if not all(len(group_vecs[g]) > 0 for g in needed):
        print("SKIP group missing:", scope, frac, layer, {g: len(group_vecs[g]) for g in needed}, flush=True)
        continue

    if len(succ_vecs) == 0 or len(ref_vecs) == 0:
        print("SKIP success/refuse missing:", scope, frac, layer, len(succ_vecs), len(ref_vecs), flush=True)
        continue

    means = {}
    for g in needed:
        means[g] = np.stack(group_vecs[g]).mean(axis=0)

    mean_success = np.stack(succ_vecs).mean(axis=0)
    mean_refuse = np.stack(ref_vecs).mean(axis=0)

    v_BA = means["B"] - means["A"]
    v_CD = means["C"] - means["D"]
    v_BC = means["B"] - means["C"]
    v_success = mean_success - mean_refuse

    v_shared = normalize(normalize(v_BA) + normalize(v_CD))

    row = {
        "scope": scope,
        "frac": frac,
        "layer": layer,
        "n_A": len(group_vecs["A"]),
        "n_B": len(group_vecs["B"]),
        "n_C": len(group_vecs["C"]),
        "n_D": len(group_vecs["D"]),
        "n_B_success": len(succ_vecs),
        "n_B_refuse": len(ref_vecs),

        "cos_success_BA": cosine(v_success, v_BA),
        "cos_success_CD": cosine(v_success, v_CD),
        "cos_success_shared": cosine(v_success, v_shared),
        "cos_success_BC_content": cosine(v_success, v_BC),

        "cos_BA_CD": cosine(v_BA, v_CD),

        "norm_success": float(np.linalg.norm(v_success)),
        "norm_BA": float(np.linalg.norm(v_BA)),
        "norm_CD": float(np.linalg.norm(v_CD)),
        "norm_shared": float(np.linalg.norm(v_shared)),
        "norm_BC": float(np.linalg.norm(v_BC)),
    }

    results.append(row)

    print(
        f"{scope:10s} f={frac:<4} L{layer:<2} "
        f"cos(success,shared)={row['cos_success_shared']:.3f} "
        f"cos(success,BA)={row['cos_success_BA']:.3f} "
        f"cos(success,CD)={row['cos_success_CD']:.3f} "
        f"cos(success,BC)={row['cos_success_BC_content']:.3f} "
        f"cos(BA,CD)={row['cos_BA_CD']:.3f}",
        flush=True,
    )

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("\nSAVED:", OUT_JSON, flush=True)

print("\n===== SORTED BY cos(success, shared) =====", flush=True)

for r in sorted(results, key=lambda x: x["cos_success_shared"], reverse=True):
    print(
        f"{r['scope']:10s} f={r['frac']:<4} L{r['layer']:<2} "
        f"cos_success_shared={r['cos_success_shared']:.3f} "
        f"cos_success_BA={r['cos_success_BA']:.3f} "
        f"cos_success_CD={r['cos_success_CD']:.3f} "
        f"cos_success_BC={r['cos_success_BC_content']:.3f} "
        f"cos_BA_CD={r['cos_BA_CD']:.3f}",
        flush=True,
    )
