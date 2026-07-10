#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch


MANIFEST = Path("outputs_dream_native_history_full/manifest.jsonl")
OUT_DIR = Path("analysis_output/dream_hidden_directions")
OUT_JSON = OUT_DIR / "dream_direction_geometry.json"

SCOPES = ["harm", "out_mask", "out_unmask"]
LAYERS = [5, 14, 23]
FRACS = [0.05, 0.10, 0.20, 0.35, 0.50, 1.00]


def group_letter(row: dict) -> str:
    v = row.get("variant", "")
    if v and v[0] in {"A", "B", "C", "D"}:
        return v[0]
    g = row.get("group", "")
    if g and g[0] in {"A", "B", "C", "D"}:
        return g[0]
    raise ValueError(f"Cannot infer group from row: {row}")


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.exists():
        return path
    alt = Path.cwd() / path
    if alt.exists():
        return alt
    alt2 = MANIFEST.parent / path.name
    if alt2.exists():
        return alt2
    raise FileNotFoundError(p)


def find_key(d: dict, wanted):
    if wanted in d:
        return wanted

    sw = str(wanted)
    if sw in d:
        return sw

    try:
        iw = int(wanted)
        if iw in d:
            return iw
        if str(iw) in d:
            return str(iw)
    except Exception:
        pass

    try:
        fw = float(wanted)
        for k in d.keys():
            try:
                if abs(float(k) - fw) < 1e-9:
                    return k
            except Exception:
                continue
    except Exception:
        pass

    return None


def get_hidden_vec(rec: dict, scope: str, frac: float, layer: int):
    hidden = rec.get("hidden", None)
    if hidden is None:
        return None

    if scope not in hidden:
        return None

    frac_map = hidden[scope]
    fk = find_key(frac_map, frac)
    if fk is None:
        return None

    layer_map = frac_map[fk]
    lk = find_key(layer_map, layer)
    if lk is None:
        return None

    vec = layer_map[lk]

    if vec is None:
        return None

    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().float().numpy()
    else:
        vec = np.asarray(vec, dtype=np.float32)

    vec = vec.astype(np.float32).reshape(-1)

    if vec.size == 0:
        return None

    if not np.isfinite(vec).all():
        return None

    return vec


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return None

    return float(np.dot(a, b) / (na * nb))


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in MANIFEST.open() if x.strip()]
    print(f"Loaded manifest rows: {len(rows)}")

    records = []
    for i, row in enumerate(rows, 1):
        p = resolve_path(row["path"])
        rec = torch.load(p, map_location="cpu")
        records.append({
            "group": group_letter(row),
            "variant": row.get("variant", ""),
            "case_id": row.get("case_id", row.get("id", "")),
            "path": str(p),
            "record": rec,
        })

        if i % 50 == 0:
            print(f"  loaded {i}/{len(rows)}")

    results = []

    print("\n==============================")
    print("Dream direction geometry")
    print("==============================")

    for scope in SCOPES:
        print(f"\n--- scope: {scope} ---")

        for frac in FRACS:
            pieces = []

            for layer in LAYERS:
                group_vecs = defaultdict(list)

                for item in records:
                    vec = get_hidden_vec(item["record"], scope, frac, layer)
                    if vec is None:
                        continue
                    group_vecs[item["group"]].append(vec)

                needed = ["A", "B", "C", "D"]
                if not all(len(group_vecs[g]) > 0 for g in needed):
                    pieces.append(f"L{layer}=EMPTY")
                    continue

                means = {}
                for g in needed:
                    means[g] = np.stack(group_vecs[g]).mean(axis=0)

                v_BA = means["B"] - means["A"]
                v_CD = means["C"] - means["D"]
                v_BC = means["B"] - means["C"]

                cos_BA_CD = cosine(v_BA, v_CD)
                cos_BA_BC = cosine(v_BA, v_BC)
                cos_CD_BC = cosine(v_CD, v_BC)

                v_shared = unit(unit(v_BA) + unit(v_CD))

                out_prefix = OUT_DIR / f"scope{scope}_f{str(frac).replace('.', 'p')}_L{layer}"

                torch.save(torch.tensor(v_BA, dtype=torch.float32), str(out_prefix) + "_dir_BA.pt")
                torch.save(torch.tensor(v_CD, dtype=torch.float32), str(out_prefix) + "_dir_CD.pt")
                torch.save(torch.tensor(v_shared, dtype=torch.float32), str(out_prefix) + "_dir_shared.pt")

                row = {
                    "scope": scope,
                    "frac": float(frac),
                    "layer": int(layer),
                    "n_A": len(group_vecs["A"]),
                    "n_B": len(group_vecs["B"]),
                    "n_C": len(group_vecs["C"]),
                    "n_D": len(group_vecs["D"]),
                    "cos_BA_CD": cos_BA_CD,
                    "cos_BA_BC": cos_BA_BC,
                    "cos_CD_BC": cos_CD_BC,
                    "dir_BA_path": str(out_prefix) + "_dir_BA.pt",
                    "dir_CD_path": str(out_prefix) + "_dir_CD.pt",
                    "dir_shared_path": str(out_prefix) + "_dir_shared.pt",
                }

                results.append(row)

                pieces.append(
                    f"L{layer}=cos(BA,CD)={cos_BA_CD:.3f}, "
                    f"cos(BA,BC)={cos_BA_BC:.3f}, "
                    f"cos(CD,BC)={cos_CD_BC:.3f}"
                )

            print(f"frac={frac:g} | " + " | ".join(pieces))

    with OUT_JSON.open("w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print(f"  {OUT_JSON}")
    print(f"  direction vectors in {OUT_DIR}/")


if __name__ == "__main__":
    main()
