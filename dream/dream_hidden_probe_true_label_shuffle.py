import os
import json
import numpy as np
import torch

ROOT = "outputs_dream_native_history_full"
MANIFEST = os.path.join(ROOT, "manifest.jsonl")

OUT_JSON = "analysis_output/dream_hidden_probe_label_shuffle_torch.json"
OUT_CSV = "analysis_output/dream_hidden_probe_label_shuffle_torch.csv"

os.makedirs("analysis_output", exist_ok=True)
torch.set_num_threads(4)

# We only test the most important suspicious/strong settings
TEST_POINTS = [
    ("B_vs_A_injection_harmful", "B", "A", "harm", 0.05, 23),
    ("B_vs_A_injection_harmful", "B", "A", "out_mask", 0.05, 23),
    ("C_vs_D_injection_neutral", "C", "D", "harm", 0.05, 5),
    ("C_vs_D_injection_neutral", "C", "D", "harm", 0.05, 23),
    ("B_vs_C_content_injected", "B", "C", "harm", 0.05, 5),
]

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

for contrast_name, pos_g, neg_g, scope, frac, layer in TEST_POINTS:
    print("\nTEST", contrast_name, scope, "f=", frac, "L", layer, flush=True)

    X = []
    y_true = []

    for rec in records:
        g = rec["group"]

        if g not in (pos_g, neg_g):
            continue

        v = get_vec(rec, scope, frac, layer)

        if v is None:
            continue

        X.append(v)
        y_true.append(1 if g == pos_g else 0)

    X = np.stack(X).astype(np.float32)
    y_true = np.asarray(y_true).astype(int)

    print("n:", len(y_true), "pos:", int(y_true.sum()), "neg:", int((1 - y_true).sum()), flush=True)

    rng = np.random.default_rng(123)
    y_shuf = y_true.copy()
    rng.shuffle(y_shuf)

    n_splits = min(5, int(y_shuf.sum()), int((1 - y_shuf).sum()))

    aucs_true_test_against_true = []
    aucs_shuffle_test_against_shuffle = []

    for train_idx, test_idx in stratified_folds(y_shuf, n_splits=n_splits, seed=0):
        scores = train_probe_torch(X[train_idx], y_shuf[train_idx], X[test_idx])

        # Main label-shuffle sanity: evaluate against shuffled labels
        auc_shuf = auc_score(y_shuf[test_idx], scores)

        # Extra: evaluate same model against original labels to detect accidental structure
        auc_orig = auc_score(y_true[test_idx], scores)

        aucs_shuffle_test_against_shuffle.append(auc_shuf)
        aucs_true_test_against_true.append(auc_orig)

    row = {
        "contrast": contrast_name,
        "scope": scope,
        "frac": frac,
        "layer": layer,
        "n": int(len(y_true)),
        "shuffle_auc_mean": float(np.mean(aucs_shuffle_test_against_shuffle)),
        "shuffle_auc_std": float(np.std(aucs_shuffle_test_against_shuffle)),
        "original_label_auc_when_trained_on_shuffle_mean": float(np.mean(aucs_true_test_against_true)),
        "original_label_auc_when_trained_on_shuffle_std": float(np.std(aucs_true_test_against_true)),
    }

    results.append(row)

    print(
        f"shuffle AUC={row['shuffle_auc_mean']:.3f}±{row['shuffle_auc_std']:.3f} | "
        f"orig-label AUC after shuffled train={row['original_label_auc_when_trained_on_shuffle_mean']:.3f}±{row['original_label_auc_when_trained_on_shuffle_std']:.3f}",
        flush=True,
    )

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

keys = list(results[0].keys())

with open(OUT_CSV, "w") as f:
    f.write(",".join(keys) + "\n")
    for r in results:
        f.write(",".join(str(r[k]) for k in keys) + "\n")

print("\nSAVED:", OUT_JSON, flush=True)
print("SAVED:", OUT_CSV, flush=True)
