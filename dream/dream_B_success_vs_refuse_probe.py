import os
import json
import numpy as np
import torch
from collections import Counter

ROOT = "outputs_dream_native_history_full"
MANIFEST = os.path.join(ROOT, "manifest.jsonl")
JUDGE_JSONL = "analysis_output/dream_B_existing_deepseek_judge.jsonl"

OUT_JSON = "analysis_output/dream_B_success_vs_refuse_probe.json"
OUT_CSV = "analysis_output/dream_B_success_vs_refuse_probe.csv"

os.makedirs("analysis_output", exist_ok=True)
torch.set_num_threads(4)

def auc_score(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    pos = scores[y_true == 1]
    neg = scores[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)

    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1

        avg_rank = (i + j + 2) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    n_pos = len(pos)
    n_neg = len(neg)
    rank_sum_pos = ranks[y_true == 1].sum()

    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    f1s = []

    for cls in [0, 1]:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))

    return float(np.mean(f1s))

def train_probe_torch(X_train, y_train, X_test):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = torch.nn.Linear(X_train.shape[1], 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(250):
        opt.zero_grad()
        logits = model(X_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(X_test).squeeze(-1)).cpu().numpy()

    return scores

def stratified_folds(y, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_folds = np.array_split(pos, n_splits)
    neg_folds = np.array_split(neg, n_splits)

    all_idx = np.arange(len(y))
    folds = []

    for k in range(n_splits):
        test_idx = np.concatenate([pos_folds[k], neg_folds[k]])
        train_idx = np.setdiff1d(all_idx, test_idx)
        folds.append((train_idx, test_idx))

    return folds

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

# Load judge labels for B
labels = {}

for line in open(JUDGE_JSONL):
    if not line.strip():
        continue

    item = json.loads(line)
    cid = item["case_id"]

    if not cid.startswith("B"):
        continue

    success = item.get("judge", {}).get("success", None)

    if success is None:
        continue

    labels[cid] = int(success)

print("judge labels:", len(labels), Counter(labels.values()), flush=True)

# Load records
rows = [json.loads(x) for x in open(MANIFEST)]
records = []

for row in rows:
    rec = torch.load(row["path"], map_location="cpu")
    meta = rec["meta"]
    cid = meta["case_id"]

    if not cid.startswith("B"):
        continue

    if cid not in labels:
        continue

    records.append({
        "case_id": cid,
        "label": labels[cid],
        "hidden": rec["hidden"],
        "counts": rec["counts"],
    })

print("B records with labels:", len(records), Counter(r["label"] for r in records), flush=True)

# Discover grid
grid = set()

for rec in records:
    for scope, frac_map in rec["hidden"].items():
        for frac, layer_map in frac_map.items():
            for layer in layer_map.keys():
                grid.add((scope, float(frac), int(layer)))

grid = sorted(grid, key=lambda x: (x[0], x[1], x[2]))
print("grid points:", len(grid), flush=True)

results = []

for scope, frac, layer in grid:
    X = []
    y = []

    for rec in records:
        v = get_vec(rec, scope, frac, layer)

        if v is None:
            continue

        X.append(v)
        y.append(rec["label"])

    if len(X) < 20 or len(set(y)) < 2:
        continue

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y).astype(int)

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    n_splits = min(5, n_pos, n_neg)

    if n_splits < 2:
        continue

    aucs = []
    f1s = []

    for train_idx, test_idx in stratified_folds(y, n_splits=n_splits, seed=0):
        scores = train_probe_torch(X[train_idx], y[train_idx], X[test_idx])
        preds = (scores >= 0.5).astype(int)

        aucs.append(auc_score(y[test_idx], scores))
        f1s.append(macro_f1(y[test_idx], preds))

    row = {
        "contrast": "B_success_vs_B_refuse",
        "scope": scope,
        "frac": frac,
        "layer": layer,
        "n": int(len(y)),
        "n_success": n_pos,
        "n_refuse": n_neg,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }

    results.append(row)

    print(
        f"{scope:12s} f={frac:<4} L{layer:<2} "
        f"n={len(y):3d} success={n_pos:2d} refuse={n_neg:2d} "
        f"AUC={row['auc_mean']:.3f}±{row['auc_std']:.3f} "
        f"F1={row['f1_mean']:.3f}",
        flush=True,
    )

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

keys = [
    "contrast",
    "scope",
    "frac",
    "layer",
    "n",
    "n_success",
    "n_refuse",
    "auc_mean",
    "auc_std",
    "f1_mean",
    "f1_std",
]

with open(OUT_CSV, "w") as f:
    f.write(",".join(keys) + "\n")
    for r in results:
        f.write(",".join(str(r[k]) for k in keys) + "\n")

print("\nSAVED:", OUT_JSON, flush=True)
print("SAVED:", OUT_CSV, flush=True)

print("\n===== BEST B SUCCESS VS REFUSE =====", flush=True)

for r in sorted(results, key=lambda x: x["auc_mean"], reverse=True)[:20]:
    print(
        f"{r['scope']:12s} f={r['frac']:<4} L{r['layer']:<2} "
        f"n={r['n']:3d} AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f} "
        f"F1={r['f1_mean']:.3f}",
        flush=True,
    )
