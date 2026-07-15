#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RRAE(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=1024,
        latent_dim=512,
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


def case_key(meta):
    if "pair_index" in meta:
        return str(meta["pair_index"])

    case_id = str(meta.get("case_id", ""))

    if case_id and case_id[0] in {"A", "B", "C", "D"}:
        return case_id[1:]

    return case_id


def group_name(meta, fallback_group):
    value = meta.get("group")

    if value in {"A", "B", "C", "D"}:
        return value

    return {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
    }[int(fallback_group)]


def fit_basis(Z, rank):
    Z = Z.float()
    z_mean = Z.mean(dim=0, keepdim=True)
    Zc = Z - z_mean

    _, singular_values, Vh = torch.linalg.svd(
        Zc,
        full_matrices=False,
    )

    rank = min(
        int(rank),
        Vh.shape[0],
        Vh.shape[1],
    )

    basis = Vh[:rank].T.contiguous()

    return {
        "mean": z_mean,
        "basis": basis,
        "singular_values": singular_values,
        "rank": rank,
    }


def project_with_basis(Z, fitted):
    Zc = Z.float() - fitted["mean"]
    basis = fitted["basis"]

    return (Zc @ basis) @ basis.T + fitted["mean"]


@torch.no_grad()
def encode_batched(model, X, batch_size=256):
    output = []

    for start in range(0, len(X), batch_size):
        output.append(
            model.encoder(X[start:start + batch_size]).cpu()
        )

    return torch.cat(output).float()


@torch.no_grad()
def decode_batched(model, Z, batch_size=256):
    output = []

    for start in range(0, len(Z), batch_size):
        output.append(
            model.decoder(Z[start:start + batch_size]).cpu()
        )

    return torch.cat(output).float()


def paired_direction(values, groups, metadata):
    by_case = defaultdict(dict)

    for index, meta in enumerate(metadata):
        key = case_key(meta)

        if not key:
            continue

        group = group_name(meta, groups[index])

        by_case[key][group] = index

    complete_cases = {
        key: ids
        for key, ids in by_case.items()
        if all(group in ids for group in ["A", "B", "C", "D"])
    }

    if not complete_cases:
        raise RuntimeError(
            "No complete paired A/B/C/D cases found in training data."
        )

    BA = []
    CD = []
    BC = []
    AD = []

    for ids in complete_cases.values():
        a = values[ids["A"]]
        b = values[ids["B"]]
        c = values[ids["C"]]
        d = values[ids["D"]]

        BA.append(b - a)
        CD.append(c - d)
        BC.append(b - c)
        AD.append(a - d)

    mean_BA = torch.stack(BA).mean(dim=0)
    mean_CD = torch.stack(CD).mean(dim=0)
    mean_BC = torch.stack(BC).mean(dim=0)
    mean_AD = torch.stack(AD).mean(dim=0)

    combined = mean_BA + mean_CD

    return {
        "mean_BA": mean_BA,
        "mean_CD": mean_CD,
        "mean_BC": mean_BC,
        "mean_AD": mean_AD,
        "combined": combined,
        "num_pairs": len(complete_cases),
    }


def cosine(a, b, eps=1e-12):
    denominator = a.norm() * b.norm()

    if denominator <= eps:
        return float("nan")

    return float(
        torch.dot(a.flatten(), b.flatten()) / denominator
    )


def auc_score(pos, neg):
    pos = torch.as_tensor(pos, dtype=torch.float64).flatten()
    neg = torch.as_tensor(neg, dtype=torch.float64).flatten()

    scores = torch.cat([pos, neg])
    labels = torch.cat([
        torch.ones_like(pos),
        torch.zeros_like(neg),
    ])

    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)

    ranks[order] = torch.arange(
        1,
        len(scores) + 1,
        dtype=torch.float64,
    )

    n_pos = len(pos)
    n_neg = len(neg)

    rank_sum_positive = ranks[labels == 1].sum()

    return float(
        (
            rank_sum_positive
            - n_pos * (n_pos + 1) / 2
        )
        / (n_pos * n_neg)
    )


def projection_auc(values, groups, vector):
    scores = values @ vector

    output = {}

    comparisons = {
        "BA": (1, 0),
        "CD": (2, 3),
        "BC": (1, 2),
        "AD": (0, 3),
    }

    for name, (positive, negative) in comparisons.items():
        output[f"{name}_auc"] = auc_score(
            scores[groups == positive],
            scores[groups == negative],
        )

    return output


def save_vector(
    out_path,
    vector_raw,
    source,
    layer,
    rank,
    checkpoint,
    dataset,
    diagnostics,
    extra_payload,
):
    vector_raw_unnormalized = vector_raw.float()
    vector_raw_normalized = F.normalize(
        vector_raw_unnormalized,
        dim=0,
    )

    payload = {
        "v_injection_raw": vector_raw_normalized.cpu(),
        "v_injection_raw_unnormalized":
            vector_raw_unnormalized.cpu(),
        "diagnostics": diagnostics,
        "metadata": {
            "scope": "input_region",
            "layer": int(layer),
            "rank": int(rank),
            "source": source,
            "dataset": str(dataset),
            "ae_checkpoint": str(checkpoint),
            "intended_steering_layer": int(layer),
            "intended_operation":
                "hidden_new = hidden_old - alpha * v_injection_raw",
        },
        **extra_payload,
    }

    torch.save(payload, out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(
        args.dataset,
        map_location="cpu",
        weights_only=False,
    )

    H = data["H"].float()
    groups = data["groups"].long()
    metadata = data["metadata"]

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = RRAE(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
    )

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt.get(
        "mean",
        ckpt.get("H_mean"),
    ).float()

    std = ckpt.get(
        "std",
        ckpt.get("H_std"),
    ).float().clamp_min(1e-6)

    train_idx = ckpt["train_idx"].long()

    X = (H - mean) / std
    Z_raw = encode_batched(model, X)

    # Fit one frozen basis using only the original checkpoint training split.
    frozen_basis = fit_basis(
        Z_raw[train_idx],
        args.rank,
    )

    Z_low = project_with_basis(
        Z_raw,
        frozen_basis,
    )

    X_hat_std = decode_batched(
        model,
        Z_low,
    )

    H_hat = X_hat_std * std + mean
    residual_raw = H - H_hat

    representations = {
        "raw_BA_plus_CD": H,
        "reconstruction_BA_plus_CD": H_hat,
        "residual_BA_plus_CD": residual_raw,
    }

    directions = {}

    for source, values in representations.items():
        direction = paired_direction(
            values,
            groups,
            metadata,
        )

        combined = direction["combined"]
        normalized = F.normalize(combined, dim=0)

        diagnostics = {
            "source": source,
            "layer": args.layer,
            "rank": args.rank,
            "num_paired_cases": direction["num_pairs"],
            "combined_unnormalized_norm":
                float(combined.norm()),
            "cos_BA_CD": cosine(
                direction["mean_BA"],
                direction["mean_CD"],
            ),
            "cos_combined_BA": cosine(
                combined,
                direction["mean_BA"],
            ),
            "cos_combined_CD": cosine(
                combined,
                direction["mean_CD"],
            ),
            **projection_auc(
                values,
                groups,
                normalized,
            ),
        }

        filename = (
            f"rrae_vector_ablation"
            f"__source-{source}"
            f"__layer-L{args.layer}"
            f"__rank-r{args.rank}"
            f"__train-v2.pt"
        )

        output_path = args.out_dir / filename

        save_vector(
            out_path=output_path,
            vector_raw=combined,
            source=source,
            layer=args.layer,
            rank=args.rank,
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            diagnostics=diagnostics,
            extra_payload={
                "mean_BA_raw": direction["mean_BA"].cpu(),
                "mean_CD_raw": direction["mean_CD"].cpu(),
                "mean_BC_raw": direction["mean_BC"].cpu(),
                "mean_AD_raw": direction["mean_AD"].cpu(),
                "standardization_mean": mean.cpu(),
                "standardization_std": std.cpu(),
                "frozen_latent_mean":
                    frozen_basis["mean"].cpu(),
                "frozen_latent_basis":
                    frozen_basis["basis"].cpu(),
            },
        )

        directions[source] = {
            "path": str(output_path),
            **diagnostics,
        }

        print("\n" + "=" * 100)
        print(source)
        print(json.dumps(diagnostics, indent=2))
        print("saved:", output_path)

    # Compare the three vector directions directly.
    raw_vector = paired_direction(
        H,
        groups,
        metadata,
    )["combined"]

    reconstruction_vector = paired_direction(
        H_hat,
        groups,
        metadata,
    )["combined"]

    residual_vector = paired_direction(
        residual_raw,
        groups,
        metadata,
    )["combined"]

    comparison = {
        "layer": args.layer,
        "rank": args.rank,
        "cos_raw_vs_reconstruction": cosine(
            raw_vector,
            reconstruction_vector,
        ),
        "cos_raw_vs_residual": cosine(
            raw_vector,
            residual_vector,
        ),
        "cos_reconstruction_vs_residual": cosine(
            reconstruction_vector,
            residual_vector,
        ),
        "reconstruction_mse_standardized":
            float(F.mse_loss(X_hat_std, X)),
        "reconstruction_mse_raw":
            float(F.mse_loss(H_hat, H)),
        "vectors": directions,
    }

    summary_path = args.out_dir / (
        f"vector_source_ablation_summary"
        f"__L{args.layer}"
        f"__r{args.rank}.json"
    )

    summary_path.write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("VECTOR COMPARISON")
    print(json.dumps(comparison, indent=2))
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
