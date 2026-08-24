from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import RRAE, RRAECheckpoint
from .utils import collect_view, family_split, group_of, seed_everything, shared_injection_direction
from .screen_representations import score_view


def train_one(x_train: torch.Tensor, rank: int, *, epochs: int, lr: float) -> tuple[RRAE, list[float], list[float]]:
    mean = x_train.mean(0)
    scale = x_train.std(0, unbiased=False).clamp_min(1e-6)
    x = (x_train - mean) / scale
    model = RRAE(x.shape[1], rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), x)
        loss.backward()
        optimizer.step()
    return model.eval(), mean.tolist(), scale.tolist()


def selection_score(row: dict) -> float:
    """Prioritize injection geometry while penalizing content/position shortcuts."""
    return (
        float(row["validation_injection_auc"])
        + float(row["validation_cos_ba_cd"])
        + float(row["train_validation_direction_cosine"])
        - abs(float(row["validation_harmfulness_auc_guardrail"]) - 0.5)
        - abs(float(row["validation_position_count_correlation"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the finalized RRAE rank sweep.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--frac", type=float, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 24, 32, 48, 64])
    parser.add_argument("--output-dir", default="outputs/rrae")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    from .utils import load_rows
    seed_everything(args.seed)
    rows = load_rows(args.records)
    train_rows, validation_rows = family_split(rows, args.validation_fraction, args.seed)
    x_train, train_rows = collect_view(train_rows, layer=args.layer, scope=args.scope, fraction=args.frac)
    x_val, validation_rows = collect_view(validation_rows, layer=args.layer, scope=args.scope, fraction=args.frac)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for rank in args.ranks:
        seed_everything(args.seed + rank)
        model, mean, scale = train_one(x_train, rank, epochs=args.epochs, lr=args.learning_rate)
        mean_t, scale_t = torch.tensor(mean), torch.tensor(scale)
        r_train = model.residual((x_train - mean_t) / scale_t)
        r_val = model.residual((x_val - mean_t) / scale_t)
        train_dir, train_geo = shared_injection_direction(r_train, train_rows)
        val_dir, val_geo = shared_injection_direction(r_val, validation_rows)
        direction_cosine = abs(float(torch.dot(train_dir, val_dir)))
        val_loss = float(torch.nn.functional.mse_loss(model((x_val - mean_t) / scale_t), (x_val - mean_t) / scale_t))
        validation_metrics = score_view(r_val.detach(), validation_rows)
        metadata = {
            "paper_path": True, "layer": args.layer, "scope": args.scope,
            "fraction": args.frac, "seed": args.seed, "train_n": len(train_rows),
            "validation_n": len(validation_rows), "validation_reconstruction_mse": val_loss,
            "validation_cos_ba_cd": val_geo["cos_ba_cd"],
            "train_validation_direction_cosine": direction_cosine,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "note": "Selection prioritizes injection geometry and guardrails, not reconstruction alone.",
        }
        checkpoint = RRAECheckpoint(x_train.shape[1], rank, mean, scale, model.state_dict(), metadata)
        checkpoint.save(output / f"rrae_rank{rank}.pt")
        summary.append({"rank": rank, **metadata})
    (output / "rank_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    selected = max(summary, key=lambda row: (selection_score(row), -int(row["rank"])))
    selection = {
        "selected_checkpoint": str(output / f"rrae_rank{selected['rank']}.pt"),
        "selected_rank": selected["rank"],
        "selection_score": selection_score(selected),
        "principle": "maximize controlled injection geometry and stability with harmfulness/position guardrails; reconstruction MSE is reported but not optimized for selection",
    }
    (output / "selected_model.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
