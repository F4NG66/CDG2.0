import os
import json
import torch
import numpy as np

ROOT = "outputs_dream_native_history_full"
MANIFEST = f"{ROOT}/manifest.jsonl"

OUT_DIR = "analysis_output/p10_dream_outmask_directions"
os.makedirs(OUT_DIR, exist_ok=True)

SCOPE = "out_mask"
FRAC = 0.05
LAYER = 14

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

    return v

def normalize(v):
    return v / (v.norm() + 1e-8)

rows = [json.loads(x) for x in open(MANIFEST)]
vecs = {g: [] for g in ["A", "B", "C", "D"]}

for row in rows:
    rec = torch.load(row["path"], map_location="cpu")
    cid = rec["meta"]["case_id"]
    g = cid[0]

    v = get_vec(rec, SCOPE, FRAC, LAYER)
    if v is not None:
        vecs[g].append(v)

print("counts:", {g: len(vecs[g]) for g in vecs}, flush=True)

means = {}
for g in ["A", "B", "C", "D"]:
    means[g] = torch.stack(vecs[g]).mean(dim=0)

v_BA = means["B"] - means["A"]
v_CD = means["C"] - means["D"]

u_BA = normalize(v_BA)
u_CD = normalize(v_CD)

M = torch.stack([u_BA, u_CD], dim=0).numpy()
_, S, Vh = np.linalg.svd(M, full_matrices=False)

shared = torch.from_numpy(Vh[0]).float()
if torch.dot(shared, u_BA) < 0:
    shared = -shared

cos_BA_CD = float(torch.dot(u_BA, u_CD).item())
cos_shared_BA = float(torch.dot(shared, u_BA).item())
cos_shared_CD = float(torch.dot(shared, u_CD).item())
var_explained = float(S[0] ** 2 / (S ** 2).sum())

print("cos_BA_CD:", cos_BA_CD, flush=True)
print("cos_shared_BA:", cos_shared_BA, flush=True)
print("cos_shared_CD:", cos_shared_CD, flush=True)
print("var_explained:", var_explained, flush=True)
print("norm_BA:", float(v_BA.norm()), flush=True)
print("norm_CD:", float(v_CD.norm()), flush=True)
print("norm_shared:", float(shared.norm()), flush=True)

payloads = {
    "BA": v_BA,
    "CD": v_CD,
    "shared": shared,
}

for name, vec in payloads.items():
    path = f"{OUT_DIR}/dir_{name}_scope{SCOPE}_f{FRAC:.2f}_L{LAYER}.pt"
    torch.save({
        "vec": vec,
        "layer": LAYER,
        "scope": SCOPE,
        "frac": FRAC,
        "pair": name,
        "cos_BA_CD": cos_BA_CD,
        "cos_shared_BA": cos_shared_BA,
        "cos_shared_CD": cos_shared_CD,
        "var_explained": var_explained,
        "norm_BA": float(v_BA.norm()),
        "norm_CD": float(v_CD.norm()),
    }, path)
    print("saved:", path, flush=True)

meta = {
    "scope": SCOPE,
    "frac": FRAC,
    "layers": [LAYER],
    "counts": {g: len(vecs[g]) for g in vecs},
    "cos_BA_CD": cos_BA_CD,
    "cos_shared_BA": cos_shared_BA,
    "cos_shared_CD": cos_shared_CD,
    "var_explained": var_explained,
    "files": {
        "BA": f"{OUT_DIR}/dir_BA_scope{SCOPE}_f{FRAC:.2f}_L{LAYER}.pt",
        "CD": f"{OUT_DIR}/dir_CD_scope{SCOPE}_f{FRAC:.2f}_L{LAYER}.pt",
        "shared": f"{OUT_DIR}/dir_shared_scope{SCOPE}_f{FRAC:.2f}_L{LAYER}.pt",
    }
}

with open(f"{OUT_DIR}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("SAVED_DIR:", OUT_DIR, flush=True)
