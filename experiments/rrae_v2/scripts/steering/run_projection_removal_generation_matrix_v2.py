#!/usr/bin/env python
import os
import sys
import json
import argparse
import re
import dataclasses
from pathlib import Path
from datetime import datetime

import torch


RRAE_DATA = "/path/to/rrae_data"
if RRAE_DATA not in sys.path:
    sys.path.insert(0, RRAE_DATA)

from cdg.config import get_backend_config
from cdg.backends import build_runner
from cdg.data import load_cdg_root


VALID_REGIONS = {"template", "output", "harm", "full"}
VALID_POS = {"mask", "unmask", "all"}


def slugify(s):
    s = s.strip()
    s = s.replace("+", "_plus_")
    s = s.replace(":", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_position_spec(spec):
    if not spec or not spec.strip():
        raise ValueError("empty --position-spec")

    parts = []
    for chunk in spec.split("+"):
        chunk = chunk.strip()
        if not chunk:
            continue

        if ":" not in chunk:
            raise ValueError(
                f"Bad position spec chunk '{chunk}'. Expected region:pos, e.g. template:mask"
            )

        region, pos = chunk.split(":", 1)
        region = region.strip()
        pos = pos.strip()

        if region not in VALID_REGIONS:
            raise ValueError(f"Invalid region '{region}'. Valid regions: {sorted(VALID_REGIONS)}")
        if pos not in VALID_POS:
            raise ValueError(f"Invalid pos '{pos}'. Valid pos: {sorted(VALID_POS)}")

        parts.append((region, pos))

    if not parts:
        raise ValueError(f"Could not parse --position-spec={spec}")

    return parts


def safe_text(x):
    if x is None:
        return ""
    return str(x)


def json_safe(x, max_str=20000):
    if x is None:
        return None

    if isinstance(x, (str, int, float, bool)):
        if isinstance(x, str) and len(x) > max_str:
            return x[:max_str] + " ... [TRUNCATED]"
        return x

    if isinstance(x, (list, tuple)):
        return [json_safe(v, max_str=max_str) for v in x]

    if isinstance(x, dict):
        return {str(k): json_safe(v, max_str=max_str) for k, v in x.items()}

    if dataclasses.is_dataclass(x):
        try:
            return json_safe(dataclasses.asdict(x), max_str=max_str)
        except Exception:
            pass

    if hasattr(x, "__dict__"):
        out = {}
        for k, v in vars(x).items():
            if k.startswith("_"):
                continue
            try:
                out[k] = json_safe(v, max_str=max_str)
            except Exception:
                out[k] = repr(v)
        return out

    return repr(x)


def case_metadata(case):
    meta = json_safe(case)

    if not isinstance(meta, dict):
        meta = {"repr": repr(case)}

    for attr in [
        "case_id",
        "variant",
        "group_letter",
        "content_type",
        "has_template",
        "attack_method",
        "behavior",
        "template",
        "prompt",
        "goal",
        "source",
    ]:
        if hasattr(case, attr) and attr not in meta:
            try:
                meta[attr] = json_safe(getattr(case, attr))
            except Exception:
                meta[attr] = repr(getattr(case, attr))

    return meta


def select_cases(cases, groups, limit_per_group):
    groups = set(g.upper() for g in groups)
    counts = {g: 0 for g in groups}
    selected = []

    for case in cases:
        g = case.group_letter.upper()
        if g not in groups:
            continue
        if limit_per_group > 0 and counts[g] >= limit_per_group:
            continue
        selected.append(case)
        counts[g] += 1

    return selected, counts


def shard_cases(cases, num_shards, shard_index):
    if num_shards <= 1:
        return cases

    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}], got {shard_index}")

    return [case for i, case in enumerate(cases) if (i % num_shards) == shard_index]


def case_key(row):
    return (
        str(row.get("case_id")),
        str(row.get("variant")),
        str(row.get("position_config")),
        str(row.get("alpha")),
        str(row.get("mode")),
        str(row.get("seed")),
    )


def load_done_keys(path):
    done = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add(case_key(r))
            except Exception:
                continue

    return done


def parse_build_inputs_output(obj):
    x = None
    regions = None

    if isinstance(obj, dict):
        regions = obj.get("regions")
        for k in ["x", "input_ids", "ids"]:
            if k in obj and torch.is_tensor(obj[k]):
                x = obj[k]
                break

    elif isinstance(obj, (tuple, list)):
        for item in obj:
            if torch.is_tensor(item) and item.ndim == 2 and x is None:
                x = item
            if isinstance(item, dict):
                if any(k in item for k in ["template", "output", "harm"]):
                    regions = item

    if x is None or regions is None:
        raise RuntimeError(
            "Could not parse runner.build_inputs(case). "
            f"type={type(obj)}, repr_head={repr(obj)[:500]}"
        )

    return x, regions


def region_to_mask(region_value, x, name):
    B, T = x.shape
    m = torch.zeros((B, T), dtype=torch.bool)

    if region_value is None:
        return m

    if torch.is_tensor(region_value):
        rv = region_value.detach().cpu()

        if rv.dtype == torch.bool:
            if rv.ndim == 1:
                if rv.numel() != T:
                    raise RuntimeError(f"Region {name} bool length mismatch: {rv.numel()} vs T={T}")
                return rv.unsqueeze(0).expand(B, T).clone()

            if rv.ndim == 2:
                return rv.bool().clone()

        flat = rv.view(-1).long().tolist()
        for p in flat:
            if 0 <= p < T:
                m[:, p] = True
        return m

    if isinstance(region_value, tuple) and len(region_value) == 2:
        s, e = int(region_value[0]), int(region_value[1])
        s = max(0, s)
        e = min(T, e)
        if e > s:
            m[:, s:e] = True
        return m

    if isinstance(region_value, list):
        if len(region_value) == 2 and all(isinstance(z, int) for z in region_value):
            s, e = int(region_value[0]), int(region_value[1])
            s = max(0, s)
            e = min(T, e)
            if e > s:
                m[:, s:e] = True
            return m

        for item in region_value:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                s, e = int(item[0]), int(item[1])
                s = max(0, s)
                e = min(T, e)
                if e > s:
                    m[:, s:e] = True
            else:
                p = int(item)
                if 0 <= p < T:
                    m[:, p] = True
        return m

    raise RuntimeError(f"Unsupported region format for {name}: {type(region_value)} {repr(region_value)[:200]}")


def build_position_mask(runner, case, position_definition):
    built = runner.build_inputs(case)
    x, regions = parse_build_inputs_output(built)

    B, T = x.shape
    combined = torch.zeros((B, T), dtype=torch.bool)

    mask_id = getattr(runner, "mask_id", 126336)

    for region_name, pos_kind in position_definition:
        if region_name == "full":
            region_mask = torch.ones((B, T), dtype=torch.bool)
        else:
            if region_name not in regions:
                continue
            region_mask = region_to_mask(regions.get(region_name), x, region_name)

        if pos_kind == "all":
            posmask = region_mask
        elif pos_kind == "mask":
            posmask = region_mask & (x.detach().cpu() == mask_id)
        elif pos_kind == "unmask":
            posmask = region_mask & (x.detach().cpu() != mask_id)
        else:
            raise ValueError(f"bad pos_kind={pos_kind}")

        combined |= posmask

    return combined


def pad_posmask_for_generation(posmask, position_definition, gen_length):
    """
    runner.build_inputs(case) returns the prompt/template length.
    During generation, DLMRunner appends gen_length mask tokens.
    The hook position mask must match the full sequence length.

    If the config includes output:mask/output:all/full:all, we steer the
    appended generation positions too. Otherwise we pad them as False.
    """
    if gen_length <= 0:
        return posmask

    B = posmask.shape[0]

    steer_generated = False
    for region_name, pos_kind in position_definition:
        if region_name == "full" and pos_kind == "all":
            steer_generated = True
        if region_name == "output" and pos_kind in {"mask", "all"}:
            steer_generated = True

    pad_value = True if steer_generated else False
    pad = torch.full((B, gen_length), pad_value, dtype=torch.bool)

    return torch.cat([posmask.cpu(), pad], dim=1)


def generate_with_manual_positions(runner, case, layer, alpha, vec, posmask):
    orig_reset_steer = runner.hooks.reset_steer
    orig_reset_all = runner.hooks.reset_all

    def noop():
        return None

    runner.hooks.reset_steer = noop
    runner.hooks.reset_all = noop

    try:
        runner.hooks.set_steer(
            layer,
            alpha,
            vec,
            positions=posmask,
            mode="projection_removal",
        )

        if hasattr(runner, "_steer"):
            runner._steer = None

        out = runner.generate(case, recorder=None)

    finally:
        runner.hooks.reset_steer = orig_reset_steer
        runner.hooks.reset_all = orig_reset_all
        runner.hooks.reset_steer()

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--prompt-root", required=True)
    ap.add_argument("--vector-path", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument(
        "--position-spec",
        required=True,
        help='Custom position config, e.g. "template:mask+output:mask".',
    )

    ap.add_argument("--backend", default="llada_attack")
    ap.add_argument("--model-path", default="/path/to/LLaDA-8B-Instruct")
    ap.add_argument("--sae-root", default="/path/to/rrae_data/saes")

    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--rank", type=int, default=32)

    ap.add_argument("--groups", nargs="+", default=["B", "C"])
    ap.add_argument("--limit-per-group", type=int, default=0, help="0 means full group")
    ap.add_argument("--alphas", nargs="+", type=float, default=[-0.5, -1.0, -2.0])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)

    args = ap.parse_args()

    position_definition = parse_position_spec(args.position_spec)
    position_name = "custom_" + slugify(args.position_spec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_suffix = ""
    if args.num_shards > 1:
        shard_suffix = f"__shard{args.shard_index:03d}_of_{args.num_shards:03d}"

    results_path = out_dir / f"results__{position_name}__rank{args.rank}__L{args.layer}{shard_suffix}.jsonl"
    meta_path = out_dir / f"metadata__{position_name}__rank{args.rank}__L{args.layer}{shard_suffix}.json"

    print("=" * 100, flush=True)
    print("[POSITION NAME]", position_name, flush=True)
    print("[POSITION SPEC]", args.position_spec, flush=True)
    print("[POSITION DEFINITION]", position_definition, flush=True)
    print("[NUM SHARDS]", args.num_shards, flush=True)
    print("[SHARD INDEX]", args.shard_index, flush=True)

    print("=" * 100, flush=True)
    print("[LOAD VECTOR]", args.vector_path, flush=True)
    vec_obj = torch.load(args.vector_path, map_location="cpu")
    v_raw = vec_obj["v_injection_raw"].float()
    print(
        "v_raw:",
        tuple(v_raw.shape),
        "norm=",
        float(v_raw.norm()),
        "finite=",
        torch.isfinite(v_raw).all().item(),
        flush=True,
    )

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
    print(
        "decode.gen_length:",
        cfg.decode.gen_length,
        "steps:",
        cfg.decode.steps,
        "block_length:",
        cfg.decode.block_length,
        "temperature:",
        cfg.decode.temperature,
        "fill_all_masks:",
        cfg.decode.fill_all_masks,
        flush=True,
    )

    print("=" * 100, flush=True)
    print("[LOAD CASES]", args.prompt_root, flush=True)

    cases = load_cdg_root(args.prompt_root)
    selected_all, counts = select_cases(cases, args.groups, args.limit_per_group)
    selected = shard_cases(selected_all, args.num_shards, args.shard_index)

    print("selected cases before sharding:", len(selected_all), "counts:", counts, flush=True)
    print("selected cases in this shard:", len(selected), flush=True)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "prompt_root": args.prompt_root,
        "vector_path": args.vector_path,
        "out_dir": str(out_dir),
        "results_path": str(results_path),
        "position_config": position_name,
        "position_spec": args.position_spec,
        "position_config_definition": position_definition,
        "backend": args.backend,
        "model_path": args.model_path,
        "sae_root": args.sae_root,
        "decode_config": {
            "gen_length": cfg.decode.gen_length,
            "steps": cfg.decode.steps,
            "block_length": cfg.decode.block_length,
            "temperature": cfg.decode.temperature,
            "fill_all_masks": cfg.decode.fill_all_masks,
        },
        "layer": args.layer,
        "rank": args.rank,
        "groups": args.groups,
        "limit_per_group": args.limit_per_group,
        "alphas": args.alphas,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "note": "Projection-removal experiment. The hook must remove lambda times the activation projection onto the raw injection direction.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    done = load_done_keys(results_path)
    print("resume existing rows:", len(done), flush=True)

    n_written = 0
    n_skipped = 0

    with results_path.open("a", encoding="utf-8") as f:
        for case_idx, case in enumerate(selected):
            print("=" * 100, flush=True)
            print(f"[CASE {case_idx+1}/{len(selected)}] {case.case_id} {case.variant}", flush=True)

            case_meta = case_metadata(case)

            # Cache the deterministic baseline so zero-position controls
            # such as A/D under R0 do not regenerate the same output once
            # for every alpha.
            baseline_output_cached = None

            base_stub = {
                "case_id": case.case_id,
                "variant": case.variant,
                "group": case.group_letter,
                "position_config": position_name,
                "position_spec": args.position_spec,
                "position_definition": position_definition,
                "alpha": 0.0,
                "mode": "baseline",
                "seed": args.seed,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            }

            if case_key(base_stub) in done:
                n_skipped += 1
            else:
                torch.manual_seed(args.seed)
                if args.device == "cuda":
                    torch.cuda.manual_seed_all(args.seed)

                runner.clear_steering()
                baseline_output = runner.generate(case, recorder=None)
                baseline_output_cached = baseline_output

                row = {
                    **base_stub,
                    "content_type": case.content_type,
                    "has_template": case.has_template,
                    "attack_method": case.attack_method,
                    "layer": args.layer,
                    "rank": args.rank,
                    "prompt_root": args.prompt_root,
                    "vector_path": args.vector_path,
                    "model_path": args.model_path,
                    "backend": args.backend,
                    "decode_config": meta["decode_config"],
                    "case_index_in_shard": case_idx,
                    "case_metadata": case_meta,
                    "position_true_count": None,
                    "behavior": safe_text(case.behavior),
                    "output": safe_text(baseline_output),
                    "error": None,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done.add(case_key(row))
                n_written += 1
                print("[BASELINE DONE]", "chars=", len(safe_text(baseline_output)), flush=True)

            try:
                posmask = build_position_mask(runner, case, position_definition)
                position_true_count = int(posmask.sum().item())
                print("[POSMASK]", "true_count=", position_true_count, "shape=", tuple(posmask.shape), flush=True)
            except Exception as e:
                print("[POSMASK ERROR]", repr(e), flush=True)
                posmask = None
                position_true_count = None

            for alpha in args.alphas:
                stub = {
                    "case_id": case.case_id,
                    "variant": case.variant,
                    "group": case.group_letter,
                    "position_config": position_name,
                    "position_spec": args.position_spec,
                    "position_definition": position_definition,
                    "alpha": alpha,
                    "mode": "steered",
                    "seed": args.seed,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                }

                if case_key(stub) in done:
                    n_skipped += 1
                    continue

                if posmask is None:
                    row = {
                        **stub,
                        "content_type": case.content_type,
                        "has_template": case.has_template,
                        "attack_method": case.attack_method,
                        "layer": args.layer,
                        "rank": args.rank,
                        "prompt_root": args.prompt_root,
                        "vector_path": args.vector_path,
                        "model_path": args.model_path,
                        "backend": args.backend,
                        "decode_config": meta["decode_config"],
                        "case_index_in_shard": case_idx,
                        "case_metadata": case_meta,
                        "position_true_count": None,
                        "full_position_true_count": None,
                        "steering_applied": False,
                        "behavior": safe_text(case.behavior),
                        "output": "",
                        "error": "failed_to_build_posmask",
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    done.add(case_key(row))
                    n_written += 1
                    print("[STEERED FAILED POSMASK]", "alpha=", alpha, flush=True)
                    continue

                torch.manual_seed(args.seed)
                if args.device == "cuda":
                    torch.cuda.manual_seed_all(args.seed)

                try:
                    posmask_for_generation = pad_posmask_for_generation(
                        posmask=posmask,
                        position_definition=position_definition,
                        gen_length=cfg.decode.gen_length,
                    )

                    prompt_position_count = int(posmask.sum().item())
                    full_position_count = int(
                        posmask_for_generation.sum().item()
                    )

                    print(
                        "[POSMASK PADDED]",
                        "from=",
                        tuple(posmask.shape),
                        "to=",
                        tuple(posmask_for_generation.shape),
                        "prompt_true_count=",
                        prompt_position_count,
                        "full_true_count=",
                        full_position_count,
                        flush=True,
                    )

                    if full_position_count == 0:
                        # Valid no-op control, e.g. clean A/D under R0.
                        # Reuse the baseline output instead of regenerating
                        # the same deterministic output for every alpha.
                        if baseline_output_cached is None:
                            runner.clear_steering()
                            baseline_output_cached = runner.generate(
                                case,
                                recorder=None,
                            )

                        steered_output = baseline_output_cached
                        steering_applied = False

                        print(
                            "[NO-OP STEERING: REUSED BASELINE]",
                            "alpha=",
                            alpha,
                            flush=True,
                        )
                    else:
                        steered_output = generate_with_manual_positions(
                            runner=runner,
                            case=case,
                            layer=args.layer,
                            alpha=alpha,
                            vec=v_raw,
                            posmask=posmask_for_generation,
                        )
                        steering_applied = True

                    error = None

                except Exception as e:
                    steered_output = ""
                    error = repr(e)
                    prompt_position_count = int(posmask.sum().item())
                    full_position_count = None
                    steering_applied = False

                row = {
                    **stub,
                    "content_type": case.content_type,
                    "has_template": case.has_template,
                    "attack_method": case.attack_method,
                    "layer": args.layer,
                    "rank": args.rank,
                    "prompt_root": args.prompt_root,
                    "vector_path": args.vector_path,
                    "model_path": args.model_path,
                    "backend": args.backend,
                    "decode_config": meta["decode_config"],
                    "case_index_in_shard": case_idx,
                    "case_metadata": case_meta,
                    "position_true_count": prompt_position_count,
                    "full_position_true_count": full_position_count,
                    "steering_applied": steering_applied,
                    "behavior": safe_text(case.behavior),
                    "output": safe_text(steered_output),
                    "error": error,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done.add(case_key(row))
                n_written += 1

                print(
                    "[STEERED DONE]",
                    "alpha=",
                    alpha,
                    "chars=",
                    len(safe_text(steered_output)),
                    "error=",
                    error,
                    flush=True,
                )

    print("=" * 100, flush=True)
    print("[DONE]", flush=True)
    print("results:", results_path, flush=True)
    print("written:", n_written, "skipped:", n_skipped, flush=True)


if __name__ == "__main__":
    main()
