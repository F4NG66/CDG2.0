from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .utils import (
    binary_auc,
    collect_view,
    group_of,
    load_rows,
    normalized,
    shared_injection_direction,
)


def score_view(vectors: torch.Tensor, rows: list[dict]) -> dict:
    direction, geometry = shared_injection_direction(vectors, rows)
    scores = (vectors @ direction).numpy()
    injection = np.array([group_of(row) in {"B", "C"} for row in rows], dtype=int)
    harmful = np.array([group_of(row) in {"A", "B"} for row in rows], dtype=int)
    position_counts = np.array([float(row.get("position_count", 0.0)) for row in rows])
    corr = 0.0
    if position_counts.std() > 0 and scores.std() > 0:
        corr = float(np.corrcoef(position_counts, scores)[0, 1])
    positive_consistency = []
    for group in ("B", "C"):
        selected = scores[[group_of(row) == group for row in rows]]
        positive_consistency.extend(selected > np.median(scores))
    return {
        "injection_auc": binary_auc(injection, scores),
        "cos_ba_cd": geometry["cos_ba_cd"],
        "positive_pair_consistency": float(np.mean(positive_consistency)) if positive_consistency else float("nan"),
        "harmfulness_auc_guardrail": binary_auc(harmful, scores),
        "position_count_correlation": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen candidate hidden-state views using controlled A/B and C/D contrasts.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 11, 16, 26])
    parser.add_argument("--scopes", nargs="+", default=["harm", "out_mask", "tpl_mask"])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35, 0.50, 1.0])
    parser.add_argument("--output", default="outputs/rrae/representation_screen.json")
    args = parser.parse_args()
    rows = load_rows(args.records)
    results = []
    for layer, scope, fraction in itertools.product(args.layers, args.scopes, args.fractions):
        try:
            vectors, kept = collect_view(rows, layer=layer, scope=scope, fraction=fraction)
            metrics = score_view(vectors, kept)
            results.append({"layer": layer, "scope": scope, "fraction": fraction, "n": len(kept), **metrics})
        except RuntimeError as exc:
            results.append({"layer": layer, "scope": scope, "fraction": fraction, "n": 0, "error": str(exc)})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
