from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import RRAECheckpoint
from .utils import collect_view, load_rows, normalized, shared_injection_direction


def _behavior_direction(rows: list[dict], vectors: torch.Tensor, label_field: str) -> torch.Tensor:
    safe = [i for i, row in enumerate(rows) if str(row.get(label_field, "")).lower() in {"safe", "refusal", "redirection"}]
    harmful = [i for i, row in enumerate(rows) if str(row.get(label_field, "")).lower() in {"harmful", "compliance", "unsafe"}]
    if not safe or not harmful:
        raise RuntimeError(f"safety direction needs both safe and harmful labels in '{label_field}'")
    return normalized(vectors[safe].mean(0) - vectors[harmful].mean(0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled residual-space injection and separate safety directions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", default="outputs/directions/final_directions.pt")
    parser.add_argument("--safety-records", help="Optional paired harmful-response/safe-response records.")
    parser.add_argument("--safety-label-field", default="behavior_label")
    parser.add_argument("--safety-layer", type=int, default=16)
    parser.add_argument("--safety-scope", default="out_mask")
    parser.add_argument("--safety-frac", type=float, default=0.05)
    args = parser.parse_args()
    checkpoint = RRAECheckpoint.load(args.checkpoint)
    meta = checkpoint.metadata
    rows = load_rows(args.records)
    hidden, kept = collect_view(rows, layer=int(meta["layer"]), scope=str(meta["scope"]), fraction=float(meta["fraction"]))
    residual = checkpoint.residual(hidden)
    v_inj, geometry = shared_injection_direction(residual, kept)
    payload: dict = {
        "v_inj": v_inj,
        "u_ba": geometry["u_ba"],
        "u_cd": geometry["u_cd"],
        "v_safety": None,
        "metadata": {
            "construction": "first right singular vector of normalized B-A and C-D residual contrasts",
            "canonical_sign": "aligned with B-A",
            "cos_ba_cd": geometry["cos_ba_cd"],
            "checkpoint": str(args.checkpoint),
            "warning": "-v_inj is not v_safety",
        },
    }
    if args.safety_records:
        safety_rows = load_rows(args.safety_records)
        safety_hidden, safety_rows = collect_view(
            safety_rows, layer=args.safety_layer, scope=args.safety_scope, fraction=args.safety_frac
        )
        payload["v_safety"] = _behavior_direction(safety_rows, safety_hidden, args.safety_label_field)
        payload["metadata"]["safety_construction"] = "mean(safe refusal/redirection) - mean(harmful compliance)"
        payload["metadata"]["safety_view"] = {
            "layer": args.safety_layer, "scope": args.safety_scope, "fraction": args.safety_frac
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(".json").write_text(json.dumps(payload["metadata"], indent=2), encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
