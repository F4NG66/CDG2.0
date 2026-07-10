import os
import json
import numpy as np
import torch

ROOT = "outputs_dream_native_history_full"
MANIFEST = os.path.join(ROOT, "manifest.jsonl")
OUT_JSON = "analysis_output/dream_direction_cosine.json"

os.makedirs("analysis_output", exist_ok=True)

POINTS = [
    ("harm", 0.05, 5),
    ("harm", 0.05, 14),
    ("harm", 0.05, 23),
    ("out_mask", 0.05, 5),
    ("out_mask", 0.05, 14),
    ("out_mask", 0.05, 23),
    ("out_unmask", 0.05, 5),
    ("out_unmask", 0.05, 14),
    ("out_unmask", 0.05, 23),
]

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

def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return float("nan")

    return float(np.dot(a, b) / denom)

rows = [json.loads(x) for x in open(MANIFEST)]

records = []

print("loading records:", len(rows), flush=True)

for row in rows:
    rec = torch.load(row["path"], map_location="cpu")
    meta = rec["meta"]

    records.append({
        "case_id": meta["case_id"],
        "group": meta["case_id"][0],
        "hidden": rec["hidden"],
        "counts": rec["counts"],
    })

results = []

for scope, frac, layer in POINTS:
    group_vecs = {g: [] for g in ["A", "B", "C", "D"]}

    for rec in records:
        g = rec["group"]
        v = get_vec(rec, scope, frac, layer)

        if v is not None:
            group_vecs[g].append(v)

    if not all(len(group_vecs[g]) > 0 for g in ["A", "B", "C", "D"]):
        print("SKIP", scope, frac, layer, {g: len(group_vecs[g]) for g in group_vecs}, flush=True)
        continue

    means = {}
    for g in ["A", "B", "C", "D"]:
        means[g] = np.stack(group_vecs[g]).mean(axis=0)

    v_BA = means["B"] - means["A"]
    v_CD = means["C"] - means["D"]
    v_BC = means["B"] - means["C"]

    row = {
        "scope": scope,
        "frac": frac,
        "layer": layer,
        "n_A": len(group_vecs["A"]),
        "n_B": len(group_vecs["B"]),
        "n_C": len(group_vecs["C"]),
        "n_D": len(group_vecs["D"]),
        "cos_BA_CD": cosine(v_BA, v_CD),
        "cos_BA_BC": cosine(v_BA, v_BC),
        "cos_CD_BC": cosine(v_CD, v_BC),
        "norm_BA": float(np.linalg.norm(v_BA)),
        "norm_CD": float(np.linalg.norm(v_CD)),
        "norm_BC": float(np.linalg.norm(v_BC)),
    }

    results.append(row)

    print(
        f"{scope:10s} f={frac:<4} L{layer:<2} "
        f"cos(BA,CD)={row['cos_BA_CD']:.3f} "
        f"cos(BA,BC)={row['cos_BA_BC']:.3f} "
        f"cos(CD,BC)={row['cos_CD_BC']:.3f}",
        flush=True,
    )

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("\nSAVED:", OUT_JSON, flush=True)

print("\n===== BEST SHARED INJECTION DIRECTION =====", flush=True)
for r in sorted(results, key=lambda x: x["cos_BA_CD"], reverse=True):
    print(
        f"{r['scope']:10s} f={r['frac']:<4} L{r['layer']:<2} "
        f"cos_BA_CD={r['cos_BA_CD']:.3f}",
        flush=True,
    )
