from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


GROUPS = ("A", "B", "C", "D")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def group_of(row: dict) -> str:
    return str(row.get("group") or row.get("variant") or "")[:1].upper()


def family_of(row: dict) -> str:
    return str(row.get("family_id") or row.get("pair_id") or row.get("case_id") or row.get("id"))


def family_split(rows: Sequence[dict], validation_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic family-level split; paired A/B/C/D rows never leak."""
    train, validation = [], []
    for row in rows:
        token = f"{seed}:{family_of(row)}".encode()
        bucket = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
        (validation if bucket < validation_fraction else train).append(row)
    return train, validation


def load_rows(records: str | Path) -> list[dict]:
    """Load an external manifest/directory without assuming a cluster path."""
    path = Path(records).expanduser()
    if path.is_dir():
        manifest = path / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"expected {manifest}")
        rows = _read_jsonl(manifest)
        for row in rows:
            record_path = row.get("path")
            if record_path:
                rp = Path(record_path)
                if not rp.is_absolute():
                    rp = path / rp
                if rp.exists():
                    row["_rec"] = torch.load(rp, map_location="cpu", weights_only=False)
        return rows
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    if path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("records"), list):
            return obj["records"]
    raise ValueError("records must be a manifest directory, JSONL, or .pt records bundle")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _match_fraction(store: dict, fraction: float):
    for key in store:
        try:
            if abs(float(key) - fraction) < 1e-7:
                return key
        except (TypeError, ValueError):
            pass
    return None


def vector_of(row: dict, *, layer: int, scope: str, fraction: float) -> torch.Tensor | None:
    rec = row.get("_rec", row)
    hidden = rec.get("hidden", {})
    store = hidden.get(scope, {})
    frac_key = _match_fraction(store, fraction)
    if frac_key is None:
        return None
    layer_store = store[frac_key]
    vector = layer_store.get(layer, layer_store.get(str(layer)))
    if vector is None:
        return None
    value = torch.as_tensor(vector).float().flatten()
    return None if torch.isnan(value).any() else value


def collect_view(rows: Iterable[dict], *, layer: int, scope: str, fraction: float) -> tuple[torch.Tensor, list[dict]]:
    vectors, kept = [], []
    for row in rows:
        vector = vector_of(row, layer=layer, scope=scope, fraction=fraction)
        if vector is not None and group_of(row) in GROUPS:
            vectors.append(vector)
            kept.append(row)
    if not vectors:
        raise RuntimeError(f"no vectors for layer={layer}, scope={scope}, fraction={fraction}")
    return torch.stack(vectors), kept


def normalized(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def group_means(vectors: torch.Tensor, rows: Sequence[dict]) -> dict[str, torch.Tensor]:
    means = {}
    for group in GROUPS:
        indices = [i for i, row in enumerate(rows) if group_of(row) == group]
        if not indices:
            raise RuntimeError(f"controlled direction requires group {group}")
        means[group] = vectors[indices].mean(0)
    return means


def shared_injection_direction(vectors: torch.Tensor, rows: Sequence[dict]) -> tuple[torch.Tensor, dict]:
    means = group_means(vectors, rows)
    u_ba = normalized(means["B"] - means["A"])
    u_cd = normalized(means["C"] - means["D"])
    matrix = torch.stack((u_ba, u_cd))
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    direction = normalized(vh[0])
    if torch.dot(direction, u_ba) < 0:
        direction = -direction
    return direction, {
        "cos_ba_cd": float(torch.dot(u_ba, u_cd)),
        "u_ba": u_ba,
        "u_cd": u_cd,
    }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
