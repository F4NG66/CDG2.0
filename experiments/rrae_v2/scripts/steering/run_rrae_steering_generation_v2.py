#!/usr/bin/env python
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch


RRAE_DATA = "/path/to/rrae_data"
if RRAE_DATA not in sys.path:
    sys.path.insert(0, RRAE_DATA)

from cdg.config import get_backend_config
from cdg.backends import build_runner
from cdg.data import load_cdg_root


def select_cases(cases, groups, limit_per_group):
    groups = set(g.upper() for g in groups)
    counts = {g: 0 for g in groups}
    selected = []

    for case in cases:
        g = case.group_letter.upper()
        if g not in groups:
            continue
        if counts[g] >= limit_per_group:
            continue
        selected.append(case)
        counts[g] += 1

    return selected, counts


def safe_text(x):
    if x is None:
        return ""
    return str(x)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--prompt-root", required=True)
    ap.add_argument("--vector-path", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--backend", default="llada_attack")
    ap.add_argument("--model-path", default="/path/to/LLaDA-8B-Instruct")
    ap.add_argument("--sae-root", default="/path/to/rrae_data/saes")

    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--rank", type=int, default=32)

    ap.add_argument("--groups", nargs="+", default=["B", "C"])
    ap.add_argument("--limit-per-group", type=int, default=2)

    ap.add_argument("--alphas", nargs="+", type=float, default=[-0.25, -0.5, -1.0])
    ap.add_argument("--scope-region", default="template")
    ap.add_argument("--pos", default="mask", choices=["mask", "unmask", "all"])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / (
        f"results__rank{args.rank}__L{args.layer}"
        f"__scope-{args.scope_region}__pos-{args.pos}"
        f"__alphas-{'-'.join(str(a).replace('-', 'neg').replace('.', 'p') for a in args.alphas)}.jsonl"
    )

    meta_path = out_dir / "run_metadata.json"

    print("=" * 100, flush=True)
    print("[LOAD VECTOR]", args.vector_path, flush=True)
    vec_obj = torch.load(args.vector_path, map_location="cpu")
    v_raw = vec_obj["v_injection_raw"].float()
    print("v_raw:", tuple(v_raw.shape), "norm=", float(v_raw.norm()), "finite=", torch.isfinite(v_raw).all().item(), flush=True)

    print("=" * 100, flush=True)
    print("[BUILD RUNNER]", flush=True)

    cfg = get_backend_config(args.backend)
    cfg.model_id = args.model_path

    runner = build_runner(
        cfg,
        sae_root=args.sae_root,
        device=args.device,
        dummy=False,
    )

    print("runner:", type(runner).__name__, flush=True)
    print("decode.gen_length:", cfg.decode.gen_length, "steps:", cfg.decode.steps, "temperature:", cfg.decode.temperature, flush=True)

    print("=" * 100, flush=True)
    print("[LOAD CASES]", args.prompt_root, flush=True)

    cases = load_cdg_root(args.prompt_root)
    selected, counts = select_cases(cases, args.groups, args.limit_per_group)

    print("selected cases:", len(selected), "counts:", counts, flush=True)

    run_meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "prompt_root": args.prompt_root,
        "vector_path": args.vector_path,
        "out_dir": str(out_dir),
        "results_path": str(results_path),
        "backend": args.backend,
        "model_path": args.model_path,
        "sae_root": args.sae_root,
        "layer": args.layer,
        "rank": args.rank,
        "groups": args.groups,
        "limit_per_group": args.limit_per_group,
        "alphas": args.alphas,
        "scope_region": args.scope_region,
        "pos": args.pos,
        "seed": args.seed,
        "operation": "runner.set_steering(vectors={layer: v_raw}, alpha=negative_alpha, scope_region=template, pos=mask)",
        "note": "HookManager applies h = h + alpha * vec, so negative alpha is intended to subtract the injection direction.",
    }

    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    vectors = {args.layer: v_raw}

    n_written = 0

    with results_path.open("w", encoding="utf-8") as f:
        for case_idx, case in enumerate(selected):
            print("=" * 100, flush=True)
            print(f"[CASE {case_idx+1}/{len(selected)}] {case.case_id} {case.variant}", flush=True)

            # Baseline
            torch.manual_seed(args.seed)
            if args.device == "cuda":
                torch.cuda.manual_seed_all(args.seed)

            runner.clear_steering()
            baseline_output = runner.generate(case, recorder=None)

            baseline_row = {
                "case_id": case.case_id,
                "group": case.group_letter,
                "variant": case.variant,
                "content_type": case.content_type,
                "has_template": case.has_template,
                "attack_method": case.attack_method,
                "mode": "baseline",
                "alpha": 0.0,
                "layer": args.layer,
                "rank": args.rank,
                "scope_region": args.scope_region,
                "pos": args.pos,
                "seed": args.seed,
                "behavior": safe_text(case.behavior),
                "output": safe_text(baseline_output),
            }

            f.write(json.dumps(baseline_row, ensure_ascii=False) + "\n")
            f.flush()
            n_written += 1

            print("[BASELINE DONE]", "chars=", len(safe_text(baseline_output)), flush=True)

            # Steered generations
            for alpha in args.alphas:
                torch.manual_seed(args.seed)
                if args.device == "cuda":
                    torch.cuda.manual_seed_all(args.seed)

                runner.clear_steering()
                runner.set_steering(
                    vectors=vectors,
                    alpha=alpha,
                    scope_region=args.scope_region,
                    pos=args.pos,
                )

                steered_output = runner.generate(case, recorder=None)
                runner.clear_steering()

                row = {
                    "case_id": case.case_id,
                    "group": case.group_letter,
                    "variant": case.variant,
                    "content_type": case.content_type,
                    "has_template": case.has_template,
                    "attack_method": case.attack_method,
                    "mode": "steered",
                    "alpha": alpha,
                    "layer": args.layer,
                    "rank": args.rank,
                    "scope_region": args.scope_region,
                    "pos": args.pos,
                    "seed": args.seed,
                    "behavior": safe_text(case.behavior),
                    "output": safe_text(steered_output),
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_written += 1

                print("[STEERED DONE]", "alpha=", alpha, "chars=", len(safe_text(steered_output)), flush=True)

    print("=" * 100, flush=True)
    print("[DONE]", flush=True)
    print("results:", results_path, flush=True)
    print("rows_written:", n_written, flush=True)


if __name__ == "__main__":
    main()
