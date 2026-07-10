#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score


# ==============================
# Config
# ==============================

MANIFEST = Path("outputs_dream_native_history_full/manifest.jsonl")
OUT_DIR = Path("analysis_output")
OUT_JSON = OUT_DIR / "dream_probe_hidden_results.json"

SCOPES = ["harm", "out_mask", "out_unmask"]
LAYERS = [5, 14, 23]
FRACS = [0.05, 0.10, 0.20, 0.35, 0.50, 1.00]

CONTRASTS = {
    "B_vs_A_injection_harmful": ("B", "A"),
    "C_vs_D_injection_neutral": ("C", "D"),
    "B_vs_C_content_danger": ("B", "C"),
}


# ==============================
# Helpers
# ==============================

def group_letter(row: dict) -> str:
    """
    Extract A/B/C/D from manifest row.
    Expected variant examples:
      A_harmful_clean
      B_harmful_injected
    """
    v = row.get("variant", "")
    if v and v[0] in {"A", "B", "C", "D"}:
        return v[0]

    g = row.get("group", "")
    if g and g[0] in {"A", "B", "C", "D"}:
        return g[0]

    raise ValueError(f"Cannot infer group from row: {row}")


def resolve_path(p: str) -> Path:
    """
    Manifest path may be absolute or relative.
    """
    path = Path(p)
    if path.exists():
        return path

    alt = Path.cwd() / path
    if alt.exists():
        return alt

    alt2 = MANIFEST.parent / path.name
    if alt2.exists():
        return alt2

    raise FileNotFoundError(f"Could not find record path: {p}")


def find_key(d: dict, wanted):
    """
    Robust key lookup because frac/layer keys may be:
      0.05, "0.05", "0.1", 5, "5", etc.
    """
    if wanted in d:
        return wanted

    sw = str(wanted)
    if sw in d:
        return sw

    # handle layer int vs string
    try:
        iw = int(wanted)
        if iw in d:
            return iw
        if str(iw) in d:
            return str(iw)
    except Exception:
        pass

    # handle frac float comparison
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
    """
    Return one hidden vector or None if this scope/frac/layer is empty/missing/NaN.
    """
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

    # expected empty vectors can be NaN; skip them
    if not np.isfinite(vec).all():
        return None

    return vec


def manual_standardize_train_test(X_train, X_test):
    """
    Avoid sklearn.pipeline / StandardScaler dependency.
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def run_probe(records, pos_group: str, neg_group: str, scope: str, frac: float, layer: int):
    """
    Build X/y for one grid point and run 5-fold logistic regression.
    Returns None if no valid vectors exist.
    """
    X = []
    y = []
    case_ids = []

    for item in records:
        g = item["group"]

        if g not in {pos_group, neg_group}:
            continue

        vec = get_hidden_vec(item["record"], scope, frac, layer)

        if vec is None:
            continue

        X.append(vec)
        y.append(1 if g == pos_group else 0)
        case_ids.append(item.get("case_id", ""))

    # This is the important empty-scope fix
    if len(X) == 0:
        return None

    # Need both classes
    if len(set(y)) < 2:
        return None

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)

    # Need enough examples for 5 folds
    counts = np.bincount(y, minlength=2)
    min_class = int(counts.min())
    if min_class < 2:
        return None

    n_splits = min(5, min_class)

    aucs = []
    f1s = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    for train_idx, test_idx in cv.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train, X_test = manual_standardize_train_test(X_train, X_test)

        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            random_state=0,
        )

        clf.fit(X_train, y_train)

        scores = clf.decision_function(X_test)
        pred = clf.predict(X_test)

        aucs.append(float(roc_auc_score(y_test, scores)))
        f1s.append(float(f1_score(y_test, pred, average="macro")))

    return {
        "scope": scope,
        "frac": float(frac),
        "layer": int(layer),
        "pos_group": pos_group,
        "neg_group": neg_group,
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }


def fmt_result(r):
    if r is None:
        return "EMPTY"
    return f"{r['auc_mean']:.3f}±{r['auc_std']:.3f}"


# ==============================
# Main
# ==============================

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with MANIFEST.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"Loaded manifest rows: {len(rows)}")

    records = []
    variants = defaultdict(int)

    print("Loading .pt records...")
    for i, row in enumerate(rows, 1):
        g = group_letter(row)
        variants[row.get("variant", g)] += 1

        p = resolve_path(row["path"])
        rec = torch.load(p, map_location="cpu")

        records.append({
            "group": g,
            "variant": row.get("variant", ""),
            "case_id": row.get("case_id", row.get("id", "")),
            "path": str(p),
            "record": rec,
        })

        if i % 50 == 0:
            print(f"  loaded {i}/{len(rows)}")

    print("Variants:")
    for k, v in sorted(variants.items()):
        print(f"  {k}: {v}")

    all_results = []

    for contrast_name, (pos_group, neg_group) in CONTRASTS.items():
        print("\n==============================")
        print(contrast_name)

        for scope in SCOPES:
            print(f"\n--- scope: {scope} ---")

            for frac in FRACS:
                row_results = []

                for layer in LAYERS:
                    r = run_probe(
                        records=records,
                        pos_group=pos_group,
                        neg_group=neg_group,
                        scope=scope,
                        frac=frac,
                        layer=layer,
                    )

                    if r is not None:
                        r["contrast"] = contrast_name
                        all_results.append(r)

                    row_results.append((layer, r))

                pieces = []
                for layer, r in row_results:
                    pieces.append(f"L{layer}={fmt_result(r)}")

                print(f"  frac={frac:g} | " + " | ".join(pieces))

    with OUT_JSON.open("w") as f:
        json.dump(all_results, f, indent=2)

    print("\n==============================")
    print(f"Saved results to: {OUT_JSON}")
    print(f"Total non-empty probe results: {len(all_results)}")


if __name__ == "__main__":
    main()
