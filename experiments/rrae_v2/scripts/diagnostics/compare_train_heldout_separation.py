#!/usr/bin/env python
from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


GROUP_NAMES = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

COMPARISONS = {
    "B_vs_A": (1, 0),
    "C_vs_D": (2, 3),
    "B_vs_C": (1, 2),
    "A_vs_D": (0, 3),
}


def load_hidden(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    H = obj["H"].float().numpy()
    groups = obj["groups"].long().numpy()

    if H.ndim != 2:
        raise ValueError(f"{path}: H must be 2D, got {H.shape}")

    if len(H) != len(groups):
        raise ValueError(
            f"{path}: H rows {len(H)} != group rows {len(groups)}"
        )

    if not np.isfinite(H).all():
        raise ValueError(f"{path}: H contains NaN or Inf")

    return H, groups


def subset_binary(H, groups, positive_group, negative_group):
    mask = np.isin(groups, [positive_group, negative_group])
    X = H[mask]
    y = (groups[mask] == positive_group).astype(np.int64)
    return X, y


def within_dataset_cv(H, groups, pos, neg, folds=5):
    X, y = subset_binary(H, groups, pos, neg)

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    aucs = []
    accs = []
    bal_accs = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="liblinear",
            random_state=42,
        )
        clf.fit(X_train, y[train_idx])

        prob = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)

        aucs.append(roc_auc_score(y[test_idx], prob))
        accs.append(accuracy_score(y[test_idx], pred))
        bal_accs.append(balanced_accuracy_score(y[test_idx], pred))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "n": int(len(y)),
    }


def train_to_heldout(H_train, g_train, H_test, g_test, pos, neg):
    X_train, y_train = subset_binary(H_train, g_train, pos, neg)
    X_test, y_test = subset_binary(H_test, g_test, pos, neg)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        C=1.0,
        max_iter=3000,
        solver="liblinear",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)

    return {
        "auc": float(roc_auc_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred)
        ),
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
    }


def mean_direction(H, groups, first, second):
    return (
        H[groups == first].mean(axis=0)
        - H[groups == second].mean(axis=0)
    )


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def direction_metrics(H_train, g_train, H_test, g_test):
    train_ba = mean_direction(H_train, g_train, 1, 0)
    train_cd = mean_direction(H_train, g_train, 2, 3)

    test_ba = mean_direction(H_test, g_test, 1, 0)
    test_cd = mean_direction(H_test, g_test, 2, 3)

    train_combined = train_ba + train_cd
    test_combined = test_ba + test_cd

    return {
        "cos_train_BA_vs_CD": cosine(train_ba, train_cd),
        "cos_heldout_BA_vs_CD": cosine(test_ba, test_cd),
        "cos_train_vs_heldout_BA": cosine(train_ba, test_ba),
        "cos_train_vs_heldout_CD": cosine(train_cd, test_cd),
        "cos_train_vs_heldout_combined": cosine(
            train_combined,
            test_combined,
        ),
        "norm_train_BA": float(np.linalg.norm(train_ba)),
        "norm_train_CD": float(np.linalg.norm(train_cd)),
        "norm_heldout_BA": float(np.linalg.norm(test_ba)),
        "norm_heldout_CD": float(np.linalg.norm(test_cd)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    H_train, g_train = load_hidden(args.train)
    H_test, g_test = load_hidden(args.heldout)

    rows = []

    for comparison, (pos, neg) in COMPARISONS.items():
        train_cv = within_dataset_cv(
            H_train, g_train, pos, neg
        )
        heldout_cv = within_dataset_cv(
            H_test, g_test, pos, neg
        )
        transfer = train_to_heldout(
            H_train,
            g_train,
            H_test,
            g_test,
            pos,
            neg,
        )

        rows.append({
            "layer": args.layer,
            "comparison": comparison,
            "positive_group": GROUP_NAMES[pos],
            "negative_group": GROUP_NAMES[neg],
            "train_cv_auc": train_cv["auc_mean"],
            "train_cv_auc_std": train_cv["auc_std"],
            "heldout_cv_auc": heldout_cv["auc_mean"],
            "heldout_cv_auc_std": heldout_cv["auc_std"],
            "train_to_heldout_auc": transfer["auc"],
            "train_to_heldout_accuracy": transfer["accuracy"],
            "train_to_heldout_balanced_accuracy":
                transfer["balanced_accuracy"],
        })

    df = pd.DataFrame(rows)

    directions = direction_metrics(
        H_train,
        g_train,
        H_test,
        g_test,
    )

    csv_path = args.out_dir / f"separation_L{args.layer}.csv"
    json_path = args.out_dir / f"directions_L{args.layer}.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(directions, f, indent=2)

    print("\n===== SEPARATION =====")
    print(df.to_string(index=False))

    print("\n===== DIRECTION COSINES =====")
    for key, value in directions.items():
        print(f"{key}: {value:.6f}")

    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
