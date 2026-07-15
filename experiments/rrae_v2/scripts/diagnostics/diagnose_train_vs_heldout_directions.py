#!/usr/bin/env python
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F


GROUP_TO_ID = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


def load_hidden(path: Path):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        raise ValueError(
            f"{path} contains only a tensor. "
            "Need group labels or metadata as well."
        )

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported object type in {path}: {type(obj)}")

    # Try common hidden-state keys.
    x = None
    for key in ["H", "X", "hidden_states", "hidden", "features"]:
        if key in obj and torch.is_tensor(obj[key]):
            x = obj[key].float()
            break

    if x is None:
        tensor_keys = [
            k for k, v in obj.items()
            if torch.is_tensor(v) and v.ndim == 2
        ]
        if len(tensor_keys) == 1:
            x = obj[tensor_keys[0]].float()
        else:
            raise KeyError(
                f"Could not identify hidden-state matrix in {path}. "
                f"Tensor keys: {tensor_keys}"
            )

    # Try common label/group keys.
    groups = None
    for key in ["groups", "group_ids", "bucket_ids", "labels_group"]:
        if key in obj:
            groups = obj[key]
            break

    if groups is None and "metadata" in obj:
        metadata = obj["metadata"]
        if isinstance(metadata, list):
            vals = []
            for row in metadata:
                g = row.get("group") or row.get("bucket")
                vals.append(GROUP_TO_ID[str(g)])
            groups = torch.tensor(vals)

    if groups is None:
        raise KeyError(
            f"Could not identify A/B/C/D group labels in {path}. "
            f"Available keys: {list(obj.keys())}"
        )

    if not torch.is_tensor(groups):
        converted = []
        for g in groups:
            if isinstance(g, str):
                converted.append(GROUP_TO_ID[g])
            else:
                converted.append(int(g))
        groups = torch.tensor(converted)

    groups = groups.long()

    if len(x) != len(groups):
        raise ValueError(
            f"Row mismatch in {path}: hidden={len(x)}, groups={len(groups)}"
        )

    return x, groups


def group_means(x, groups):
    means = {}
    for name, idx in GROUP_TO_ID.items():
        mask = groups == idx
        if mask.sum() == 0:
            raise ValueError(f"No rows found for group {name}")
        means[name] = x[mask].mean(dim=0)
    return means


def cosine(a, b):
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=1,
    ).item()


def summarize(name, path):
    x, groups = load_hidden(path)
    means = group_means(x, groups)

    ba = means["B"] - means["A"]
    cd = means["C"] - means["D"]
    harm = means["A"] - means["D"]
    harm_inj = means["B"] - means["C"]

    result = {
        "name": name,
        "path": str(path),
        "n": len(x),
        "dim": x.shape[1],
        "BA": ba,
        "CD": cd,
        "harm_clean": harm,
        "harm_injected": harm_inj,
    }

    print(f"\n===== {name} =====")
    print("path:", path)
    print("shape:", tuple(x.shape))
    print("counts:", {
        k: int((groups == v).sum())
        for k, v in GROUP_TO_ID.items()
    })
    print(f"||B-A||: {ba.norm().item():.6f}")
    print(f"||C-D||: {cd.norm().item():.6f}")
    print(f"cos(B-A, C-D): {cosine(ba, cd):.6f}")
    print(f"||A-D||: {harm.norm().item():.6f}")
    print(f"||B-C||: {harm_inj.norm().item():.6f}")
    print(
        "cos(A-D, B-C):",
        f"{cosine(harm, harm_inj):.6f}",
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    args = parser.parse_args()

    train = summarize("TRAIN", args.train)
    heldout = summarize("HELDOUT", args.heldout)

    print("\n===== CROSS-DATASET GENERALIZATION =====")
    print(
        "cos train(B-A), heldout(B-A):",
        f"{cosine(train['BA'], heldout['BA']):.6f}",
    )
    print(
        "cos train(C-D), heldout(C-D):",
        f"{cosine(train['CD'], heldout['CD']):.6f}",
    )
    print(
        "cos train(B-A), heldout(C-D):",
        f"{cosine(train['BA'], heldout['CD']):.6f}",
    )
    print(
        "cos train(C-D), heldout(B-A):",
        f"{cosine(train['CD'], heldout['BA']):.6f}",
    )
    print(
        "cos train clean-harm, heldout clean-harm:",
        f"{cosine(train['harm_clean'], heldout['harm_clean']):.6f}",
    )
    print(
        "cos train injected-harm, heldout injected-harm:",
        f"{cosine(train['harm_injected'], heldout['harm_injected']):.6f}",
    )

    injection_train = (
        F.normalize(train["BA"], dim=0)
        + F.normalize(train["CD"], dim=0)
    )
    injection_heldout = (
        F.normalize(heldout["BA"], dim=0)
        + F.normalize(heldout["CD"], dim=0)
    )

    print(
        "cos combined injection direction:",
        f"{cosine(injection_train, injection_heldout):.6f}",
    )


if __name__ == "__main__":
    main()
