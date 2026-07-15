#!/usr/bin/env python
"""
Diagnose where an RRAE checkpoint preserves or loses the transferable
DIJA injection direction.

Stages:
    H_raw
    X_standardized
    Z_raw
    Z_low_native
    Z_low_frozen
    X_hat_native_std
    X_hat_frozen_std
    X_hat_native_raw
    X_hat_frozen_raw
    residual_native_std
    residual_frozen_std
    residual_native_raw
    residual_frozen_raw

Two low-rank modes:
1. native:
   Recompute the SVD independently on each evaluated dataset.
   This reproduces the behavior of the current RRAE implementation.

2. frozen:
   Estimate the latent mean and right-singular-vector basis only from
   the checkpoint's original training split, then freeze and apply that
   same projection to both train and held-out datasets.

Held-out data are never used to fit normalization, model weights,
or the frozen low-rank basis.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
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


class RRAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_hidden(path: Path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = ["H", "groups", "metadata"]

    for key in required:
        if key not in obj:
            raise KeyError(f"{path}: missing key {key!r}")

    H = obj["H"].float()
    groups = obj["groups"].long()
    metadata = obj["metadata"]

    if H.ndim != 2:
        raise ValueError(f"{path}: expected H to be 2D, got {H.shape}")

    if H.shape[0] != groups.shape[0]:
        raise ValueError(
            f"{path}: H rows={H.shape[0]} "
            f"but groups rows={groups.shape[0]}"
        )

    if len(metadata) != H.shape[0]:
        raise ValueError(
            f"{path}: metadata rows={len(metadata)} "
            f"but H rows={H.shape[0]}"
        )

    if not torch.isfinite(H).all():
        raise ValueError(f"{path}: H contains non-finite values")

    return {
        "H": H,
        "groups": groups,
        "metadata": metadata,
        "object": obj,
    }


def load_model(checkpoint_path: Path):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    input_dim = int(ckpt["input_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    latent_dim = int(ckpt["latent_dim"])

    model = RRAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt.get("mean", ckpt.get("H_mean")).float()
    std = ckpt.get("std", ckpt.get("H_std")).float().clamp_min(1e-6)

    train_idx = ckpt["train_idx"].long()
    val_idx = ckpt["val_idx"].long()

    return model, ckpt, mean, std, train_idx, val_idx


@torch.no_grad()
def encode_batched(model, X, batch_size=256):
    chunks = []

    for start in range(0, X.shape[0], batch_size):
        chunks.append(
            model.encode(X[start:start + batch_size]).cpu()
        )

    return torch.cat(chunks, dim=0).float()


@torch.no_grad()
def decode_batched(model, Z, batch_size=256):
    chunks = []

    for start in range(0, Z.shape[0], batch_size):
        chunks.append(
            model.decode(Z[start:start + batch_size]).cpu()
        )

    return torch.cat(chunks, dim=0).float()


def fit_svd_basis(Z_fit, rank):
    """
    Fit a PCA/SVD-style low-rank affine subspace:
        z_projected = (z - mean) V V^T + mean
    """

    Z_fit = torch.nan_to_num(
        Z_fit.float(),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )

    latent_mean = Z_fit.mean(dim=0, keepdim=True)
    Z_centered = Z_fit - latent_mean

    U, S, Vh = torch.linalg.svd(
        Z_centered,
        full_matrices=False,
    )

    effective_rank = min(
        int(rank),
        Vh.shape[0],
        Vh.shape[1],
    )

    basis = Vh[:effective_rank].T.contiguous()

    return {
        "mean": latent_mean,
        "basis": basis,
        "singular_values": S,
        "rank": effective_rank,
    }


def apply_svd_basis(Z, fitted):
    Z = Z.float()
    mean = fitted["mean"]
    basis = fitted["basis"]

    Z_centered = Z - mean
    Z_low = (Z_centered @ basis) @ basis.T + mean

    return Z_low.float()


def native_low_rank_projection(Z, rank):
    fitted = fit_svd_basis(Z, rank)
    return apply_svd_basis(Z, fitted), fitted


def case_key(meta):
    if "pair_index" in meta:
        return str(meta["pair_index"])

    if "case_id" in meta:
        case_id = str(meta["case_id"])

        if case_id and case_id[0] in {"A", "B", "C", "D"}:
            return case_id[1:]

        return case_id

    return None


def group_name_from_meta(meta, fallback_group_id):
    group = meta.get("group")

    if group in {"A", "B", "C", "D"}:
        return group

    return GROUP_NAMES[int(fallback_group_id)]


def build_complete_pairs(metadata, groups):
    by_case = {}

    for index, meta in enumerate(metadata):
        key = case_key(meta)

        if key is None:
            continue

        group = group_name_from_meta(
            meta,
            groups[index].item(),
        )

        by_case.setdefault(key, {})[group] = index

    complete = {
        key: value
        for key, value in by_case.items()
        if all(group in value for group in ["A", "B", "C", "D"])
    }

    return complete


def paired_directions(values, groups, metadata):
    """
    Match the vector-extraction pipeline:
    calculate differences within each complete ABCD case, then average.
    """

    pairs = build_complete_pairs(metadata, groups)

    if pairs:
        BA = []
        CD = []
        BC = []
        AD = []

        for ids in pairs.values():
            a = values[ids["A"]]
            b = values[ids["B"]]
            c = values[ids["C"]]
            d = values[ids["D"]]

            BA.append(b - a)
            CD.append(c - d)
            BC.append(b - c)
            AD.append(a - d)

        BA = torch.stack(BA).mean(dim=0)
        CD = torch.stack(CD).mean(dim=0)
        BC = torch.stack(BC).mean(dim=0)
        AD = torch.stack(AD).mean(dim=0)

        method = "paired_case_differences"
        pair_count = len(pairs)

    else:
        BA = (
            values[groups == 1].mean(dim=0)
            - values[groups == 0].mean(dim=0)
        )
        CD = (
            values[groups == 2].mean(dim=0)
            - values[groups == 3].mean(dim=0)
        )
        BC = (
            values[groups == 1].mean(dim=0)
            - values[groups == 2].mean(dim=0)
        )
        AD = (
            values[groups == 0].mean(dim=0)
            - values[groups == 3].mean(dim=0)
        )

        method = "group_mean_differences"
        pair_count = 0

    combined = BA + CD

    return {
        "BA": BA.float(),
        "CD": CD.float(),
        "BC": BC.float(),
        "AD": AD.float(),
        "combined": combined.float(),
        "method": method,
        "pair_count": pair_count,
    }


def cosine(a, b, eps=1e-12):
    denominator = a.norm() * b.norm()

    if denominator.item() <= eps:
        return float("nan")

    return float(
        torch.dot(a.flatten(), b.flatten()) / denominator
    )


def direction_summary(values, groups, metadata):
    directions = paired_directions(
        values,
        groups,
        metadata,
    )

    return {
        "cos_BA_vs_CD": cosine(
            directions["BA"],
            directions["CD"],
        ),
        "norm_BA": float(directions["BA"].norm()),
        "norm_CD": float(directions["CD"].norm()),
        "norm_combined": float(directions["combined"].norm()),
        "direction_method": directions["method"],
        "num_complete_pairs": directions["pair_count"],
        "_directions": directions,
    }


def subset_binary(values, groups, positive, negative):
    mask = torch.logical_or(
        groups == positive,
        groups == negative,
    )

    X = values[mask].float().numpy()
    y = (groups[mask] == positive).long().numpy()

    return X, y


def within_dataset_cv(
    values,
    groups,
    positive,
    negative,
    folds=5,
):
    X, y = subset_binary(
        values,
        groups,
        positive,
        negative,
    )

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    aucs = []
    accuracies = []
    balanced_accuracies = []

    for train_indices, test_indices in splitter.split(X, y):
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X[train_indices])
        X_test = scaler.transform(X[test_indices])

        classifier = LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="liblinear",
            random_state=42,
        )

        classifier.fit(
            X_train,
            y[train_indices],
        )

        probability = classifier.predict_proba(X_test)[:, 1]
        prediction = classifier.predict(X_test)

        aucs.append(
            roc_auc_score(
                y[test_indices],
                probability,
            )
        )

        accuracies.append(
            accuracy_score(
                y[test_indices],
                prediction,
            )
        )

        balanced_accuracies.append(
            balanced_accuracy_score(
                y[test_indices],
                prediction,
            )
        )

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accuracies)),
        "balanced_accuracy_mean": float(
            np.mean(balanced_accuracies)
        ),
    }


def train_to_heldout_probe(
    train_values,
    train_groups,
    heldout_values,
    heldout_groups,
    positive,
    negative,
):
    X_train, y_train = subset_binary(
        train_values,
        train_groups,
        positive,
        negative,
    )

    X_test, y_test = subset_binary(
        heldout_values,
        heldout_groups,
        positive,
        negative,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = LogisticRegression(
        C=1.0,
        max_iter=3000,
        solver="liblinear",
        random_state=42,
    )

    classifier.fit(X_train, y_train)

    probability = classifier.predict_proba(X_test)[:, 1]
    prediction = classifier.predict(X_test)

    return {
        "auc": float(roc_auc_score(y_test, probability)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, prediction)
        ),
    }


def construct_stages(
    model,
    H,
    mean,
    std,
    frozen_basis,
    rank,
):
    X = (H - mean) / std
    Z_raw = encode_batched(model, X)

    Z_low_native, native_basis = native_low_rank_projection(
        Z_raw,
        rank,
    )

    Z_low_frozen = apply_svd_basis(
        Z_raw,
        frozen_basis,
    )

    X_hat_native_std = decode_batched(
        model,
        Z_low_native,
    )

    X_hat_frozen_std = decode_batched(
        model,
        Z_low_frozen,
    )

    X_hat_native_raw = X_hat_native_std * std + mean
    X_hat_frozen_raw = X_hat_frozen_std * std + mean

    stages = {
        "H_raw": H.float(),
        "X_standardized": X.float(),
        "Z_raw": Z_raw.float(),
        "Z_low_native": Z_low_native.float(),
        "Z_low_frozen": Z_low_frozen.float(),
        "X_hat_native_std": X_hat_native_std.float(),
        "X_hat_frozen_std": X_hat_frozen_std.float(),
        "X_hat_native_raw": X_hat_native_raw.float(),
        "X_hat_frozen_raw": X_hat_frozen_raw.float(),
        "residual_native_std": (
            X - X_hat_native_std
        ).float(),
        "residual_frozen_std": (
            X - X_hat_frozen_std
        ).float(),
        "residual_native_raw": (
            H - X_hat_native_raw
        ).float(),
        "residual_frozen_raw": (
            H - X_hat_frozen_raw
        ).float(),
    }

    reconstruction = {
        "native_mse_std": float(
            F.mse_loss(X_hat_native_std, X)
        ),
        "frozen_mse_std": float(
            F.mse_loss(X_hat_frozen_std, X)
        ),
        "native_mse_raw": float(
            F.mse_loss(X_hat_native_raw, H)
        ),
        "frozen_mse_raw": float(
            F.mse_loss(X_hat_frozen_raw, H)
        ),
    }

    return stages, reconstruction, native_basis


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--heldout-data",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--layer",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--rank",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip logistic-regression probes for a faster structural check.",
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RRAE SIGNAL-PATH DIAGNOSTIC")
    print("layer:", args.layer)
    print("rank:", args.rank)
    print("checkpoint:", args.checkpoint)
    print("train data:", args.train_data)
    print("held-out data:", args.heldout_data)

    train = load_hidden(args.train_data)
    heldout = load_hidden(args.heldout_data)

    model, ckpt, mean, std, train_idx, val_idx = load_model(
        args.checkpoint
    )

    expected_rank = int(ckpt["rank_k"])

    if expected_rank != args.rank:
        print(
            f"[WARN] checkpoint rank={expected_rank}, "
            f"requested diagnostic rank={args.rank}"
        )

    if train["H"].shape[1] != int(ckpt["input_dim"]):
        raise ValueError(
            "Train hidden-state dimension does not match checkpoint."
        )

    if heldout["H"].shape[1] != int(ckpt["input_dim"]):
        raise ValueError(
            "Held-out hidden-state dimension does not match checkpoint."
        )

    # Fit the frozen basis only on the original checkpoint training split.
    X_train_all = (train["H"] - mean) / std
    Z_train_all = encode_batched(model, X_train_all)
    Z_basis_fit = Z_train_all[train_idx]

    frozen_basis = fit_svd_basis(
        Z_basis_fit,
        args.rank,
    )

    print(
        "Frozen basis fitted on checkpoint training rows:",
        len(train_idx),
    )
    print("Frozen basis rank:", frozen_basis["rank"])

    train_stages, train_reconstruction, train_native_basis = (
        construct_stages(
            model=model,
            H=train["H"],
            mean=mean,
            std=std,
            frozen_basis=frozen_basis,
            rank=args.rank,
        )
    )

    heldout_stages, heldout_reconstruction, heldout_native_basis = (
        construct_stages(
            model=model,
            H=heldout["H"],
            mean=mean,
            std=std,
            frozen_basis=frozen_basis,
            rank=args.rank,
        )
    )

    direction_rows = []
    probe_rows = []

    raw_train_summary = direction_summary(
        train_stages["H_raw"],
        train["groups"],
        train["metadata"],
    )

    raw_heldout_summary = direction_summary(
        heldout_stages["H_raw"],
        heldout["groups"],
        heldout["metadata"],
    )

    raw_reference_cosine = cosine(
        raw_train_summary["_directions"]["combined"],
        raw_heldout_summary["_directions"]["combined"],
    )

    raw_train_combined_norm = raw_train_summary["norm_combined"]
    raw_heldout_combined_norm = raw_heldout_summary["norm_combined"]

    for stage_name in train_stages:
        train_values = train_stages[stage_name]
        heldout_values = heldout_stages[stage_name]

        train_summary = direction_summary(
            train_values,
            train["groups"],
            train["metadata"],
        )

        heldout_summary = direction_summary(
            heldout_values,
            heldout["groups"],
            heldout["metadata"],
        )

        train_directions = train_summary["_directions"]
        heldout_directions = heldout_summary["_directions"]

        train_heldout_BA = cosine(
            train_directions["BA"],
            heldout_directions["BA"],
        )

        train_heldout_CD = cosine(
            train_directions["CD"],
            heldout_directions["CD"],
        )

        train_heldout_combined = cosine(
            train_directions["combined"],
            heldout_directions["combined"],
        )

        cosine_retention = (
            train_heldout_combined / raw_reference_cosine
            if raw_reference_cosine != 0
            else float("nan")
        )

        train_norm_retention = (
            train_summary["norm_combined"]
            / raw_train_combined_norm
            if raw_train_combined_norm != 0
            else float("nan")
        )

        heldout_norm_retention = (
            heldout_summary["norm_combined"]
            / raw_heldout_combined_norm
            if raw_heldout_combined_norm != 0
            else float("nan")
        )

        direction_rows.append({
            "layer": args.layer,
            "rank": args.rank,
            "stage": stage_name,

            "train_cos_BA_vs_CD":
                train_summary["cos_BA_vs_CD"],

            "heldout_cos_BA_vs_CD":
                heldout_summary["cos_BA_vs_CD"],

            "cos_train_vs_heldout_BA":
                train_heldout_BA,

            "cos_train_vs_heldout_CD":
                train_heldout_CD,

            "cos_train_vs_heldout_combined":
                train_heldout_combined,

            "cosine_retention_vs_raw":
                cosine_retention,

            "train_norm_BA":
                train_summary["norm_BA"],

            "train_norm_CD":
                train_summary["norm_CD"],

            "heldout_norm_BA":
                heldout_summary["norm_BA"],

            "heldout_norm_CD":
                heldout_summary["norm_CD"],

            "train_norm_combined":
                train_summary["norm_combined"],

            "heldout_norm_combined":
                heldout_summary["norm_combined"],

            "train_combined_norm_retention_vs_raw":
                train_norm_retention,

            "heldout_combined_norm_retention_vs_raw":
                heldout_norm_retention,

            "train_direction_method":
                train_summary["direction_method"],

            "heldout_direction_method":
                heldout_summary["direction_method"],

            "train_num_complete_pairs":
                train_summary["num_complete_pairs"],

            "heldout_num_complete_pairs":
                heldout_summary["num_complete_pairs"],
        })

        if not args.skip_probes:
            for comparison, (positive, negative) in COMPARISONS.items():
                train_cv = within_dataset_cv(
                    train_values,
                    train["groups"],
                    positive,
                    negative,
                )

                heldout_cv = within_dataset_cv(
                    heldout_values,
                    heldout["groups"],
                    positive,
                    negative,
                )

                transfer = train_to_heldout_probe(
                    train_values,
                    train["groups"],
                    heldout_values,
                    heldout["groups"],
                    positive,
                    negative,
                )

                probe_rows.append({
                    "layer": args.layer,
                    "rank": args.rank,
                    "stage": stage_name,
                    "comparison": comparison,

                    "train_cv_auc":
                        train_cv["auc_mean"],

                    "train_cv_auc_std":
                        train_cv["auc_std"],

                    "heldout_cv_auc":
                        heldout_cv["auc_mean"],

                    "heldout_cv_auc_std":
                        heldout_cv["auc_std"],

                    "train_to_heldout_auc":
                        transfer["auc"],

                    "train_to_heldout_accuracy":
                        transfer["accuracy"],

                    "train_to_heldout_balanced_accuracy":
                        transfer["balanced_accuracy"],
                })

        print(
            f"{stage_name:25s} "
            f"combined_cos={train_heldout_combined:.6f} "
            f"train_BA_CD={train_summary['cos_BA_vs_CD']:.6f} "
            f"heldout_BA_CD={heldout_summary['cos_BA_vs_CD']:.6f}"
        )

    direction_df = pd.DataFrame(direction_rows)

    direction_csv = (
        args.out_dir
        / f"direction_path__L{args.layer}__r{args.rank}.csv"
    )

    direction_df.to_csv(
        direction_csv,
        index=False,
    )

    probe_csv = None

    if probe_rows:
        probe_df = pd.DataFrame(probe_rows)

        probe_csv = (
            args.out_dir
            / f"probe_path__L{args.layer}__r{args.rank}.csv"
        )

        probe_df.to_csv(
            probe_csv,
            index=False,
        )

    summary = {
        "layer": args.layer,
        "rank": args.rank,
        "checkpoint": str(args.checkpoint),
        "train_data": str(args.train_data),
        "heldout_data": str(args.heldout_data),

        "checkpoint_training_rows_used_for_frozen_basis":
            int(len(train_idx)),

        "checkpoint_validation_rows":
            int(len(val_idx)),

        "raw_train_heldout_combined_cosine":
            raw_reference_cosine,

        "train_reconstruction":
            train_reconstruction,

        "heldout_reconstruction":
            heldout_reconstruction,

        "frozen_basis_top_singular_values": [
            float(value)
            for value in frozen_basis["singular_values"][:20]
        ],

        "train_native_basis_top_singular_values": [
            float(value)
            for value in train_native_basis["singular_values"][:20]
        ],

        "heldout_native_basis_top_singular_values": [
            float(value)
            for value in heldout_native_basis["singular_values"][:20]
        ],

        "methodological_note": (
            "The current RRAE implementation recomputes the low-rank SVD "
            "on every evaluated tensor. Therefore native train and native "
            "held-out projections use different latent bases. The frozen "
            "condition fits one basis only on the checkpoint training split "
            "and applies it unchanged to all data."
        ),

        "direction_csv": str(direction_csv),

        "probe_csv": (
            str(probe_csv)
            if probe_csv is not None
            else None
        ),
    }

    summary_json = (
        args.out_dir
        / f"summary__L{args.layer}__r{args.rank}.json"
    )

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===== RECONSTRUCTION =====")
    print("Train:", json.dumps(train_reconstruction, indent=2))
    print("Held-out:", json.dumps(heldout_reconstruction, indent=2))

    print("\n===== DIRECTION PATH =====")
    display_columns = [
        "stage",
        "train_cos_BA_vs_CD",
        "heldout_cos_BA_vs_CD",
        "cos_train_vs_heldout_BA",
        "cos_train_vs_heldout_CD",
        "cos_train_vs_heldout_combined",
        "cosine_retention_vs_raw",
    ]

    print(
        direction_df[display_columns].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(direction_csv)

    if probe_csv is not None:
        print(probe_csv)

    print(summary_json)


if __name__ == "__main__":
    main()
