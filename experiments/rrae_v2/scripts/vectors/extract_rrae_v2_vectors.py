#!/usr/bin/env python
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RRAE(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024, latent_dim=512):
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
    cid = str(meta["case_id"])
    return cid[1:]


def cosine(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())


def auc_score(pos, neg):
    pos = torch.as_tensor(pos, dtype=torch.float64).flatten()
    neg = torch.as_tensor(neg, dtype=torch.float64).flatten()

    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])

    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)

    n_pos = len(pos)
    n_neg = len(neg)
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def low_rank_project(Z, rank):
    Z = Z.float()
    z_mean = Z.mean(dim=0, keepdim=True)
    Zc = Z - z_mean

    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
    V = Vh[:rank].T
    Zr = (Zc @ V) @ V.T + z_mean
    return Zr.float(), S.float()


def compute_direction_for_rank(model, X, Z, mean, std, metadata, rank):
    Zr, S = low_rank_project(Z, rank)

    with torch.no_grad():
        Xhat_parts = []
        for start in range(0, Zr.shape[0], 256):
            Xhat_parts.append(model.decoder(Zr[start:start + 256]).cpu())
        Xhat = torch.cat(Xhat_parts, dim=0).float()

    residual = X - Xhat
    recon_mse = float(F.mse_loss(Xhat, X).item())

    by_case = defaultdict(dict)
    for i, meta in enumerate(metadata):
        by_case[case_key(meta)][meta["group"]] = i

    valid_cases = [
        k for k, v in by_case.items()
        if all(g in v for g in ["A", "B", "C", "D"])
    ]

    if not valid_cases:
        raise RuntimeError("No complete A/B/C/D paired cases found.")

    BA, CD, BC, AD = [], [], [], []

    for k in valid_cases:
        ids = by_case[k]
        a = residual[ids["A"]]
        b = residual[ids["B"]]
        c = residual[ids["C"]]
        d = residual[ids["D"]]

        BA.append(b - a)
        CD.append(c - d)
        BC.append(b - c)
        AD.append(a - d)

    BA = torch.stack(BA).float()
    CD = torch.stack(CD).float()
    BC = torch.stack(BC).float()
    AD = torch.stack(AD).float()

    mean_BA = BA.mean(dim=0)
    mean_CD = CD.mean(dim=0)
    mean_BC = BC.mean(dim=0)
    mean_AD = AD.mean(dim=0)

    v_std = F.normalize(mean_BA + mean_CD, dim=0)

    v_raw_unnormalized = v_std * std.flatten()
    v_raw = F.normalize(v_raw_unnormalized, dim=0)

    dir_BA = F.normalize(mean_BA, dim=0)
    dir_CD = F.normalize(mean_CD, dim=0)
    dir_BC = F.normalize(mean_BC, dim=0)
    dir_AD = F.normalize(mean_AD, dim=0)

    ba_pos, ba_neg = [], []
    cd_pos, cd_neg = [], []
    bc_pos, bc_neg = [], []

    for k in valid_cases:
        ids = by_case[k]
        a = residual[ids["A"]]
        b = residual[ids["B"]]
        c = residual[ids["C"]]
        d = residual[ids["D"]]

        ba_pos.append(torch.dot(b, v_std))
        ba_neg.append(torch.dot(a, v_std))
        cd_pos.append(torch.dot(c, v_std))
        cd_neg.append(torch.dot(d, v_std))
        bc_pos.append(torch.dot(b, v_std))
        bc_neg.append(torch.dot(c, v_std))

    diagnostics = {
        "rank": int(rank),
        "num_paired_cases": int(len(valid_cases)),
        "recon_mse_after_low_rank": recon_mse,
        "cos_BA_CD": cosine(dir_BA, dir_CD),
        "cos_v_BA": cosine(v_std, dir_BA),
        "cos_v_CD": cosine(v_std, dir_CD),
        "cos_v_BC": cosine(v_std, dir_BC),
        "cos_v_AD": cosine(v_std, dir_AD),
        "BA_auc_using_v": auc_score(torch.stack(ba_pos), torch.stack(ba_neg)),
        "CD_auc_using_v": auc_score(torch.stack(cd_pos), torch.stack(cd_neg)),
        "BC_auc_using_v": auc_score(torch.stack(bc_pos), torch.stack(bc_neg)),
        "v_std_shape": list(v_std.shape),
        "v_raw_shape": list(v_raw.shape),
        "v_std_norm": float(v_std.norm().item()),
        "v_raw_norm": float(v_raw.norm().item()),
        "v_raw_unnormalized_norm": float(v_raw_unnormalized.norm().item()),
        "top_singular_values": [float(x) for x in S[:10].tolist()],
    }

    payload = {
        "v_injection_std": v_std.cpu(),
        "v_injection_raw": v_raw.cpu(),
        "v_injection_raw_unnormalized": v_raw_unnormalized.cpu(),
        "mean_BA_std_residual": mean_BA.cpu(),
        "mean_CD_std_residual": mean_CD.cpu(),
        "mean_BC_std_residual": mean_BC.cpu(),
        "mean_AD_std_residual": mean_AD.cpu(),
        "standardization_mean": mean.cpu(),
        "standardization_std": std.cpu(),
        "diagnostics": diagnostics,
    }

    return payload, diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ranks", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default="abcd_v2_clean_N2000")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    ae_path = Path(args.ae)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100, flush=True)
    print("[LOAD DATASET]", dataset_path, flush=True)

    data = torch.load(dataset_path, map_location="cpu")
    H = data["H"].float()
    metadata = data["metadata"]

    print("H:", tuple(H.shape), "finite:", torch.isfinite(H).all().item(), flush=True)

    print("[LOAD AE]", ae_path, flush=True)
    ckpt = torch.load(ae_path, map_location="cpu")

    hidden_dim = int(ckpt.get("hidden_dim", 1024))
    latent_dim = int(ckpt.get("latent_dim", 512))

    model = RRAE(input_dim=H.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt.get("mean", ckpt.get("H_mean")).float()
    std = ckpt.get("std", ckpt.get("H_std")).float().clamp_min(1e-6)

    X = (H - mean) / std

    with torch.no_grad():
        Z_parts = []
        for start in range(0, X.shape[0], 256):
            z = model.encoder(X[start:start + 256])
            Z_parts.append(z.cpu())
        Z = torch.cat(Z_parts, dim=0).float()

    print("X:", tuple(X.shape), "Z:", tuple(Z.shape), flush=True)

    all_diags = {}

    for rank in args.ranks:
        print("=" * 100, flush=True)
        print(f"[COMPUTE DIRECTION] scope={args.scope} layer={args.layer} rank={rank}", flush=True)

        payload, diagnostics = compute_direction_for_rank(
            model=model,
            X=X,
            Z=Z,
            mean=mean,
            std=std,
            metadata=metadata,
            rank=rank,
        )

        base = (
            f"rrae_v_injection__scope-{args.scope}"
            f"__layer-L{args.layer}"
            f"__rank-r{rank}"
            f"__source-residual_BA_plus_CD"
            f"__{args.tag}"
        )

        pt_path = out_dir / f"{base}.pt"
        json_path = out_dir / f"{base}.diagnostics.json"

        save_obj = {
            **payload,
            "metadata": {
                "scope": args.scope,
                "layer": args.layer,
                "rank": rank,
                "source": "RRAE v2 residual direction: normalize(mean(B-A) + mean(C-D))",
                "dataset": str(dataset_path),
                "ae_checkpoint": str(ae_path),
                "tag": args.tag,
                "intended_steering_layer": args.layer,
                "intended_operation": "hidden_new = hidden_old - alpha * v_injection_raw",
            },
        }

        torch.save(save_obj, pt_path)
        json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

        all_diags[str(rank)] = diagnostics

        print("[SAVED]", pt_path, flush=True)
        print(json.dumps(diagnostics, indent=2), flush=True)

    summary_path = out_dir / (
        f"rrae_v2_vector_summary__scope-{args.scope}"
        f"__layer-L{args.layer}"
        f"__{args.tag}.json"
    )
    summary_path.write_text(json.dumps(all_diags, indent=2), encoding="utf-8")

    print("=" * 100, flush=True)
    print("[SUMMARY]", summary_path, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
