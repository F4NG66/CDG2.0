#!/usr/bin/env python
import json
import math
import argparse
from pathlib import Path

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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, rank_k):
        z = self.encode(x)
        z_low = low_rank_projection(z, rank_k)
        x_hat = self.decode(z_low)
        return x_hat, z, z_low


def low_rank_projection(Z, rank_k):
    rank_k = int(rank_k)
    if rank_k <= 0:
        return torch.zeros_like(Z)

    max_rank = min(Z.shape[0], Z.shape[1])
    rank_k = min(rank_k, max_rank)

    orig_dtype = Z.dtype
    orig_device = Z.device

    Z_clean = torch.nan_to_num(Z.float(), nan=0.0, posinf=1e6, neginf=-1e6)
    z_mean = Z_clean.mean(dim=0, keepdim=True)
    Zc = Z_clean - z_mean

    try:
        U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
    except Exception as e1:
        try:
            if Zc.is_cuda:
                U, S, Vh = torch.linalg.svd(Zc, full_matrices=False, driver="gesvd")
            else:
                raise e1
        except Exception:
            print("[WARN] SVD failed; fallback CPU float64 SVD.", flush=True)
            Z_cpu = Zc.detach().cpu().double()
            U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(Z_cpu, full_matrices=False)
            U = U_cpu.to(orig_device, dtype=torch.float32)
            S = S_cpu.to(orig_device, dtype=torch.float32)
            Vh = Vh_cpu.to(orig_device, dtype=torch.float32)

    Z_low = (U[:, :rank_k] * S[:rank_k].unsqueeze(0)) @ Vh[:rank_k, :] + z_mean
    return Z_low.to(device=orig_device, dtype=orig_dtype)


def cosine(a, b, eps=1e-8):
    return float(torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm() + eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hidden-dim", type=int, default=1024)
    ap.add_argument("--latent-dim", type=int, default=512)
    ap.add_argument("--rank-k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    print("dataset:", dataset_path, flush=True)
    print("out_dir:", out_dir, flush=True)

    data = torch.load(dataset_path, map_location="cpu")
    H = data["H"].float()
    groups = data["groups"].long()
    is_injected = data["is_injected"].long()
    is_harmful = data["is_harmful"].long()
    metadata = data["metadata"]

    assert H.ndim == 2 and H.shape[1] == 4096
    assert torch.isfinite(H).all(), "H contains non-finite values"

    mean = H.mean(dim=0, keepdim=True)
    std = H.std(dim=0, keepdim=True).clamp_min(1e-6)
    X = (H - mean) / std
    N, input_dim = X.shape

    print("=== DATA ===", flush=True)
    print("X shape:", tuple(X.shape), flush=True)
    for name, gid in {"A": 0, "B": 1, "C": 2, "D": 3}.items():
        print(f"{name}: {int((groups == gid).sum())}", flush=True)
    print("injected:", int(is_injected.sum()), flush=True)
    print("harmful:", int(is_harmful.sum()), flush=True)

    print("=== CONFIG ===", flush=True)
    print("rank_k:", args.rank_k, flush=True)
    print("epochs:", args.epochs, flush=True)

    perm = torch.randperm(N)
    n_train = int(0.8 * N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train = X[train_idx].to(device)
    X_val = X[val_idx].to(device)

    model = RRAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()

        x_hat, z, z_low = model(X_train, args.rank_k)
        loss = F.mse_loss(x_hat, X_train)

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_hat, val_z, val_z_low = model(X_val, args.rank_k)
            val_loss = F.mse_loss(val_hat, X_val)

        row = {
            "epoch": epoch,
            "train_mse": float(loss.detach().cpu()),
            "val_mse": float(val_loss.detach().cpu()),
        }
        history.append(row)

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "latent_dim": args.latent_dim,
                    "rank_k": args.rank_k,
                    "mean": mean,
                    "std": std,
                    "H_mean": mean,
                    "H_std": std,
                    "train_idx": train_idx.cpu(),
                    "val_idx": val_idx.cpu(),
                    "dataset_config": data.get("config", {}),
                    "dataset_path": str(dataset_path),
                    "model_class": "RRAE_v2",
                },
                out_dir / "best_model.pt",
            )

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} "
                f"train_mse={float(loss):.6f} "
                f"val_mse={float(val_loss):.6f} "
                f"best_val={best_val:.6f}",
                flush=True,
            )

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        X_all = X.to(device)
        Z_raw = model.encode(X_all)
        Z_low = low_rank_projection(Z_raw, args.rank_k)
        X_hat = model.decode(Z_low)
        recon_mse_all = F.mse_loss(X_hat, X_all).item()

        Z_raw_cpu = Z_raw.cpu()
        Z_low_cpu = Z_low.cpu()
        X_hat_cpu = X_hat.cpu()

    A = groups == 0
    B = groups == 1
    C = groups == 2
    D = groups == 3

    v_BA = Z_low_cpu[B].mean(dim=0) - Z_low_cpu[A].mean(dim=0)
    v_CD = Z_low_cpu[C].mean(dim=0) - Z_low_cpu[D].mean(dim=0)
    v_BC = Z_low_cpu[B].mean(dim=0) - Z_low_cpu[C].mean(dim=0)

    summary = {
        "note": "RRAE v2 training on clean ABCD v2 hidden states.",
        "dataset": str(dataset_path),
        "out_dir": str(out_dir),
        "N": int(N),
        "input_dim": int(input_dim),
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "rank_k": args.rank_k,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_val_mse": float(best_val),
        "recon_mse_all": float(recon_mse_all),
        "group_counts": {
            "A": int(A.sum()),
            "B": int(B.sum()),
            "C": int(C.sum()),
            "D": int(D.sum()),
        },
        "cos_BA_CD_lowrank_latent": cosine(v_BA, v_CD),
        "cos_BA_BC_lowrank_latent": cosine(v_BA, v_BC),
        "cos_CD_BC_lowrank_latent": cosine(v_CD, v_BC),
    }

    with (out_dir / "history.jsonl").open("w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    torch.save(
        {
            "Z_raw": Z_raw_cpu,
            "Z_low": Z_low_cpu,
            "X_hat": X_hat_cpu,
            "groups": groups,
            "is_injected": is_injected,
            "is_harmful": is_harmful,
            "metadata": metadata,
            "summary": summary,
        },
        out_dir / "encoded_outputs.pt",
    )

    print("=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
