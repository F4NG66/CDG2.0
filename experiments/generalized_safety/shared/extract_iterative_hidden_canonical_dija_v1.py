#!/usr/bin/env python3
"""Canonical RRAE hidden-state recorder for finalized ABCD data.

A/D use standard LLaDA assistant generation with 128 appended masks.
B/C use canonical DIJA infilling: only interleaved masks already embedded
in the refined prompt are denoised, with no appended assistant-output masks.
Fractions are measured relative to each case's own denoising trajectory, and
hidden states are recorded before transfer at each requested fraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MASK_TOKEN = "<|mdm_mask|>"
MASK_ID_DEFAULT = 126336
GROUP_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
EXPECTED_BUCKETS = set(GROUP_TO_ID)
DEFAULT_LAYERS = (4, 8, 11, 16, 20, 24, 26, 28)
DEFAULT_FRACTIONS = (0.05, 0.10, 0.20, 0.35, 0.50, 1.00)
SCOPES = {
    "tpl_mask": ("template", "mask"),
    "tpl_ctx": ("template", "unmask"),
    "harm": ("harm", "unmask"),
    "out_mask": ("output", "mask"),
    "out_unmask": ("output", "unmask"),
}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json_write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def pair_index_from_row(row: dict) -> int:
    paired_a_id = str(row.get("paired_A_id", ""))
    match = re.fullmatch(r"A_(\d{4})", paired_a_id)
    if not match:
        raise ValueError(
            f"Invalid paired_A_id={paired_a_id!r} for row id={row.get('id')!r}"
        )
    return int(match.group(1))


def validate_dataset(
    rows: list[dict],
    dataset_path: Path,
    expected_sha256: str | None,
) -> dict[int, dict[str, dict]]:
    actual_sha256 = sha256_file(dataset_path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise AssertionError(
            f"Dataset SHA256 mismatch: expected={expected_sha256} actual={actual_sha256}"
        )
    if len(rows) != 2000:
        raise AssertionError(f"Expected 2000 rows, got {len(rows)}")
    ids = [str(row.get("id", "")) for row in rows]
    if len(set(ids)) != 2000:
        raise AssertionError("Dataset IDs are not unique")
    counts = Counter(str(row.get("bucket", "")) for row in rows)
    expected_counts = Counter({"A": 500, "B": 500, "C": 500, "D": 500})
    if counts != expected_counts:
        raise AssertionError(f"Unexpected bucket counts: {dict(counts)}")

    by_pair: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        bucket = str(row.get("bucket", ""))
        if bucket not in EXPECTED_BUCKETS:
            raise ValueError(f"Unknown bucket: {bucket!r}")
        pair_index = pair_index_from_row(row)
        if pair_index not in range(500):
            raise ValueError(f"Out-of-range pair index: {pair_index}")
        if bucket in by_pair[pair_index]:
            raise AssertionError(f"Duplicate {bucket} in pair {pair_index}")
        by_pair[pair_index][bucket] = row

    if set(by_pair) != set(range(500)):
        raise AssertionError("Pair index set is not exactly 0..499")

    for pair_index, group_rows in by_pair.items():
        if set(group_rows) != EXPECTED_BUCKETS:
            raise AssertionError(
                f"Incomplete pair {pair_index}: {sorted(group_rows)}"
            )
        expected_a = f"A_{pair_index:04d}"
        expected_d = f"D_{pair_index:04d}"
        for row in group_rows.values():
            if row.get("paired_A_id") != expected_a:
                raise AssertionError(f"paired_A_id mismatch in pair {pair_index}")
            if row.get("paired_D_id") != expected_d:
                raise AssertionError(f"paired_D_id mismatch in pair {pair_index}")

        a, b, c, d = (
            group_rows["A"],
            group_rows["B"],
            group_rows["C"],
            group_rows["D"],
        )
        if str(a["prompt"]) != str(b["clean_prompt"]):
            raise AssertionError(f"A/B clean prompt mismatch in pair {pair_index}")
        if str(d["prompt"]) != str(c["clean_prompt"]):
            raise AssertionError(f"D/C clean prompt mismatch in pair {pair_index}")
        if not str(b["prompt"]).startswith(str(a["prompt"])):
            raise AssertionError(f"B source prefix mismatch in pair {pair_index}")
        if not str(c["prompt"]).startswith(str(d["prompt"])):
            raise AssertionError(f"C source prefix mismatch in pair {pair_index}")

    return dict(by_pair)


def expand_mask_spans(text: str) -> tuple[str, int]:
    total = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal total
        count = int(match.group(1))
        if count <= 0:
            raise ValueError(f"Invalid mask span count: {count}")
        total += count
        return MASK_TOKEN * count

    expanded = re.sub(r"<mask:(\d+)>", replace, str(text))
    if re.search(r"<mask:[^>]*>", expanded):
        raise ValueError("Malformed or unexpanded <mask:...> marker remains")
    total += expanded.count(MASK_TOKEN) - total
    return expanded, total


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int]:
    if not needle:
        return (-1, -1)
    width = len(needle)
    for start in range(0, len(haystack) - width + 1):
        if haystack[start : start + width] == needle:
            return (start, start + width)
    return (-1, -1)


def tolerant_find(haystack: list[int], needle: list[int]) -> tuple[int, int]:
    candidates = (needle, needle[1:], needle[:-1], needle[1:-1])
    for candidate in candidates:
        if not candidate:
            continue
        start, end = find_subsequence(haystack, candidate)
        if start >= 0:
            return (start, end)
    return (-1, -1)


def tokenize_without_specials(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_inputs_and_regions(
    tokenizer,
    row: dict,
    mask_id: int,
    gen_length: int,
) -> tuple[torch.Tensor, dict[str, tuple[int, int] | None], dict]:
    bucket = str(row["bucket"])
    clean_prompt = str(row.get("clean_prompt") or row["prompt"])
    raw_prompt = str(row["prompt"])
    is_injected = bucket in {"B", "C"}

    template_span: tuple[int, int] | None = None
    output_span: tuple[int, int] | None = None
    matching_count: int | None = None

    if is_injected:
        # Canonical DIJA order: apply chat template first, then expand <mask:N>.
        tagged_content = f"<<TPL>>{raw_prompt}<</TPL>>"
        prompt_unexpanded = tokenizer.apply_chat_template(
            [{"role": "user", "content": tagged_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_string, expected_template_masks = expand_mask_spans(
            prompt_unexpanded
        )
        ids = tokenize_without_specials(tokenizer, prompt_string)

        open_ids = tokenize_without_specials(tokenizer, "<<TPL>>")
        close_ids = tokenize_without_specials(tokenizer, "<</TPL>>")
        open_start, open_end = tolerant_find(ids, open_ids)
        close_start, close_end = tolerant_find(ids, close_ids)
        if open_start < 0 or close_start < 0 or open_end > close_start:
            raise RuntimeError(
                f"Could not resolve template sentinels for id={row.get('id')}"
            )
        inner = ids[open_end:close_start]
        ids = ids[:open_start] + inner + ids[close_end:]
        template_span = (open_start, open_start + len(inner))

        vanilla_string = tokenizer.apply_chat_template(
            [{"role": "user", "content": clean_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        vanilla_ids = tokenize_without_specials(tokenizer, vanilla_string)
        matching_count = min(
            sum(int(a == b) for a, b in zip(vanilla_ids, ids)),
            len(vanilla_ids),
        )

        # DIJA fills only masks already embedded in the refined prompt.
        full_ids = ids
        generation_mode = "canonical_dija_infill"
        appended_output_mask_count = 0
    else:
        expected_template_masks = 0
        prompt_string = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tokenize_without_specials(tokenizer, prompt_string)
        prompt_len = len(ids)
        output_span = (prompt_len, prompt_len + gen_length)
        full_ids = ids + [mask_id] * gen_length
        generation_mode = "standard_assistant_generation"
        appended_output_mask_count = gen_length

    clean_ids = tokenize_without_specials(tokenizer, clean_prompt)
    harm_start, harm_end = tolerant_find(ids, clean_ids)
    if harm_start < 0:
        raise RuntimeError(
            f"Could not locate clean/harm prompt in tokenized input for id={row.get('id')}"
        )
    harm_span = (harm_start, harm_end)

    actual_template_masks = 0
    if template_span is not None:
        lo, hi = template_span
        actual_template_masks = sum(token == mask_id for token in full_ids[lo:hi])
        if actual_template_masks != expected_template_masks:
            raise AssertionError(
                "Template mask-token count mismatch for "
                f"id={row.get('id')}: expected={expected_template_masks} "
                f"actual={actual_template_masks}"
            )
    elif expected_template_masks != 0:
        raise AssertionError("Expected template masks without a template span")

    initial_mask_positions = [
        index for index, token in enumerate(full_ids) if token == mask_id
    ]
    initial_fillable_mask_count = len(initial_mask_positions)
    if initial_fillable_mask_count <= 0:
        raise AssertionError(f"No fillable masks for id={row.get('id')}")

    metadata = {
        "prompt_len": len(ids),
        "total_len": len(full_ids),
        "generation_mode": generation_mode,
        "appended_output_mask_count": appended_output_mask_count,
        "expected_template_masks": expected_template_masks,
        "actual_template_masks": actual_template_masks,
        "initial_fillable_mask_count": initial_fillable_mask_count,
        "initial_mask_positions": initial_mask_positions,
        "matching_count": matching_count,
        "prompt_sha256": sha256_text(raw_prompt),
        "clean_prompt_sha256": sha256_text(clean_prompt),
    }
    regions = {
        "template": template_span,
        "harm": harm_span,
        "output": output_span,
    }
    return torch.tensor([full_ids], dtype=torch.long), regions, metadata

def fractions_to_steps(
    fractions: tuple[float, ...],
    total_steps: int,
) -> dict[int, tuple[float, ...]]:
    """Map requested fractions to discrete pre-transfer trajectory steps.

    Short trajectories can make multiple requested fractions resolve to the
    same discrete step. Those fractions intentionally share the exact same
    captured state rather than being shifted to scientifically different steps.
    """
    result: dict[int, list[float]] = {}
    for fraction in fractions:
        if not (0 < fraction <= 1):
            raise ValueError(f"Fraction must be in (0,1], got {fraction}")
        step = max(1, min(total_steps, round(fraction * total_steps)))
        result.setdefault(step, []).append(fraction)
    return {
        step: tuple(values)
        for step, values in sorted(result.items())
    }


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    result = torch.zeros(
        mask_num.size(0),
        steps,
        device=mask_index.device,
        dtype=torch.int64,
    ) + base
    for row_index in range(mask_num.size(0)):
        result[row_index, : remainder[row_index]] += 1
    return result


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits_f64 = logits.to(torch.float64)
    noise = torch.rand_like(logits_f64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits_f64.exp() / gumbel_noise


def confidence_for_predictions(
    logits: torch.Tensor,
    predictions: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    if strategy == "random":
        return torch.rand(
            predictions.shape,
            device=predictions.device,
            dtype=torch.float64,
        )
    probabilities = F.softmax(logits.to(torch.float64), dim=-1)
    if strategy == "low_confidence":
        return torch.gather(
            probabilities,
            -1,
            predictions.unsqueeze(-1),
        ).squeeze(-1)
    raise NotImplementedError(strategy)


def get_blocks(model):
    candidates = (
        "model.transformer.blocks",
        "transformer.blocks",
        "model.model.layers",
        "model.layers",
        "model.transformer.h",
        "transformer.h",
    )
    for dotted in candidates:
        current = model
        valid = True
        for part in dotted.split("."):
            if not hasattr(current, part):
                valid = False
                break
            current = getattr(current, part)
        if valid and isinstance(current, (torch.nn.ModuleList, list)) and current:
            return current, dotted
    raise AttributeError("Could not resolve transformer block list")


def output_to_hidden(output):
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(hidden):
        raise TypeError(f"Unexpected block output type: {type(output).__name__}")
    return hidden


def selected_indices(
    x_row: torch.Tensor,
    span: tuple[int, int],
    position_kind: str,
    mask_id: int,
) -> torch.Tensor:
    lo, hi = span
    segment = x_row[lo:hi]
    is_mask = segment == mask_id
    if position_kind == "mask":
        local = is_mask.nonzero(as_tuple=True)[0]
    elif position_kind == "unmask":
        local = (~is_mask).nonzero(as_tuple=True)[0]
    else:
        raise ValueError(position_kind)
    return local + lo


def decode_nonmask(tokenizer, ids: torch.Tensor, mask_id: int) -> str:
    ids = ids[ids != mask_id]
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def record_state(
    *,
    x: torch.Tensor,
    captured: dict[int, torch.Tensor],
    regions: dict[str, tuple[int, int] | None],
    layers: tuple[int, ...],
    fraction: float,
    mask_id: int,
    hidden_store: dict,
    counts_store: dict,
    decoded_store: dict,
    tokenizer,
    store_decoded: bool,
) -> None:
    x_row = x[0]
    for scope_name, (region_name, position_kind) in SCOPES.items():
        span = regions.get(region_name)
        if span is None:
            continue
        indices = selected_indices(x_row, span, position_kind, mask_id)
        count = int(indices.numel())
        counts_store.setdefault(scope_name, {})[fraction] = count
        fraction_store = hidden_store.setdefault(scope_name, {}).setdefault(
            fraction, {}
        )
        for layer in layers:
            hidden = captured.get(layer)
            if hidden is None:
                raise RuntimeError(f"Layer {layer} was not captured")
            hidden_row = hidden[0]
            if count == 0:
                # Empty scopes are represented by a finite zero vector.
                # The accompanying count remains the authoritative signal
                # that no tokens were present in this scope.
                vector = torch.zeros(
                    (hidden_row.shape[-1],),
                    dtype=torch.float16,
                )
            else:
                vector = (
                    hidden_row.index_select(0, indices)
                    .float()
                    .mean(dim=0)
                    .half()
                    .cpu()
                    .contiguous()
                )
                if vector.ndim != 1 or vector.shape[0] != 4096:
                    raise RuntimeError(
                        f"Unexpected vector shape for L{layer}: {tuple(vector.shape)}"
                    )
                if not torch.isfinite(vector).all():
                    raise RuntimeError(
                        f"Non-finite vector with non-empty positions: {scope_name} "
                        f"f={fraction} L{layer}"
                    )
            fraction_store[layer] = vector

    if store_decoded:
        for region_name, span in regions.items():
            if span is None:
                continue
            lo, hi = span
            decoded_store.setdefault(region_name, {})[fraction] = decode_nonmask(
                tokenizer,
                x_row[lo:hi],
                mask_id,
            )


def stable_generation_seed(base_seed: int, pair_index: int, bucket: str) -> int:
    return int(base_seed) * 10_000 + int(pair_index) * 4 + GROUP_TO_ID[bucket]


def extract_case(
    *,
    model,
    tokenizer,
    hooks,
    captured,
    row: dict,
    pair_index: int,
    dataset_path: Path,
    dataset_sha256: str,
    protocol_core: dict,
    protocol_sha256: str,
    device: torch.device,
    args,
) -> dict:
    bucket = str(row["bucket"])
    generation_seed = stable_generation_seed(args.base_seed, pair_index, bucket)
    torch.manual_seed(generation_seed)
    torch.cuda.manual_seed_all(generation_seed)

    x_cpu, regions, input_metadata = build_inputs_and_regions(
        tokenizer,
        row,
        args.mask_id,
        args.gen_length,
    )
    x = x_cpu.to(device)
    initial_x = x.clone()
    attention_mask = torch.ones_like(x, dtype=torch.long, device=device)
    initial_masks = x == args.mask_id
    initial_mask_count = int(initial_masks.sum().item())

    if input_metadata["generation_mode"] == "canonical_dija_infill":
        trajectory_steps = initial_mask_count
    else:
        trajectory_steps = args.steps
        if initial_mask_count != args.gen_length:
            raise AssertionError(
                f"A/D initial masks must equal gen_length: "
                f"{initial_mask_count} != {args.gen_length}"
            )

    if trajectory_steps <= 0:
        raise AssertionError("Trajectory has no denoising steps")

    transfer_schedule = get_num_transfer_tokens(
        initial_masks,
        trajectory_steps,
    )
    step_to_fraction = fractions_to_steps(
        tuple(args.fractions),
        trajectory_steps,
    )

    hidden_store: dict = {}
    counts_store: dict = {}
    decoded_store: dict = {}

    for global_step in range(1, trajectory_steps + 1):
        mask_index = x == args.mask_id
        captured.clear()
        with torch.inference_mode():
            output = model(x, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output[0]
        if set(captured) != set(args.layers):
            raise RuntimeError(
                f"Captured layers mismatch: expected={args.layers} got={sorted(captured)}"
            )

        if global_step in step_to_fraction:
            for fraction in step_to_fraction[global_step]:
                record_state(
                    x=x,
                    captured=captured,
                    regions=regions,
                    layers=tuple(args.layers),
                    fraction=fraction,
                    mask_id=args.mask_id,
                    hidden_store=hidden_store,
                    counts_store=counts_store,
                    decoded_store=decoded_store,
                    tokenizer=tokenizer,
                    store_decoded=args.store_decoded,
                )

        logits_noised = add_gumbel_noise(logits, args.temperature)
        prediction = torch.argmax(logits_noised, dim=-1)
        confidence = confidence_for_predictions(
            logits,
            prediction,
            args.remask,
        )
        prediction = torch.where(mask_index, prediction, x)
        negative_infinity = torch.tensor(
            -np.inf,
            device=device,
            dtype=confidence.dtype,
        )
        confidence = torch.where(mask_index, confidence, negative_infinity)

        transfer = torch.zeros_like(prediction, dtype=torch.bool)
        for batch_index in range(confidence.shape[0]):
            k = int(transfer_schedule[batch_index, global_step - 1])
            if k > 0:
                _, selected = torch.topk(confidence[batch_index], k=k)
                transfer[batch_index, selected] = True
        x[transfer] = prediction[transfer]

    remaining_masks = int((x == args.mask_id).sum().item())
    if remaining_masks != 0:
        raise AssertionError("Denoising finished with masks still present")

    initial_mask_positions = input_metadata["initial_mask_positions"]
    position_tensor = torch.tensor(
        initial_mask_positions,
        dtype=torch.long,
        device=x.device,
    )
    filled_mask_token_ids = x[0].index_select(0, position_tensor).tolist()

    if input_metadata["generation_mode"] == "canonical_dija_infill":
        matching_count = int(input_metadata["matching_count"])
        final_output = tokenizer.batch_decode(
            x[:, matching_count:].detach().cpu(),
            skip_special_tokens=True,
        )[0]
        final_output = final_output.split("assistant\n")[0]
    else:
        output_span = regions["output"]
        if output_span is None:
            raise AssertionError("A/D output span is missing")
        output_lo, output_hi = output_span
        final_output = tokenizer.decode(
            x[0, output_lo:output_hi].tolist(),
            skip_special_tokens=True,
        )

    final_template = None
    if regions["template"] is not None:
        template_lo, template_hi = regions["template"]
        final_template = tokenizer.decode(
            x[0, template_lo:template_hi].tolist(),
            skip_special_tokens=True,
        )

    metadata = {
        "protocol": "RRAE_CANONICAL_DIJA_ITERATIVE_HIDDEN_EXTRACTION_V1",
        "protocol_sha256": protocol_sha256,
        "protocol_core": protocol_core,
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": dataset_sha256,
        "source_row_id": row["id"],
        "pair_index": pair_index,
        "bucket": bucket,
        "group_id": GROUP_TO_ID[bucket],
        "paired_A_id": row["paired_A_id"],
        "paired_D_id": row["paired_D_id"],
        "label": row.get("label"),
        "domain": row.get("domain"),
        "variant": row.get("variant"),
        "injection_type": row.get("injection_type"),
        "source_dataset_name": row.get("source_dataset"),
        "generation_seed": generation_seed,
        "recording_semantics": "pre_transfer_at_fraction_of_case_trajectory",
        "trajectory_steps": trajectory_steps,
        "fraction_to_step": {
            str(fraction): step
            for step, fractions_at_step in step_to_fraction.items()
            for fraction in fractions_at_step
        },
        "fraction_effective_position": {
            str(fraction): step / trajectory_steps
            for step, fractions_at_step in step_to_fraction.items()
            for fraction in fractions_at_step
        },
        "fraction_step_collisions": {
            str(step): list(fractions_at_step)
            for step, fractions_at_step in step_to_fraction.items()
            if len(fractions_at_step) > 1
        },
        "remaining_mask_count": remaining_masks,
        "input": input_metadata,
    }
    return {
        "metadata": metadata,
        "layout": {
            "prompt_len": input_metadata["prompt_len"],
            "configured_ad_gen_length": args.gen_length,
            "appended_output_mask_count": input_metadata[
                "appended_output_mask_count"
            ],
            "trajectory_steps": trajectory_steps,
            "total": input_metadata["total_len"],
        },
        "regions": regions,
        "counts": counts_store,
        "hidden": hidden_store,
        "decoded": decoded_store,
        "final_output": final_output,
        "final_template": final_template,
        "full_reconstructed_text": tokenizer.decode(
            x[0].tolist(),
            skip_special_tokens=True,
        ),
        "filled_mask_token_ids": filled_mask_token_ids,
        "filled_mask_text": tokenizer.decode(
            filled_mask_token_ids,
            skip_special_tokens=True,
        ),
        "initial_ids_sha256": hashlib.sha256(
            initial_x.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "final_ids_sha256": hashlib.sha256(
            x.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
    }

def validate_existing_record(
    path: Path,
    protocol_sha256: str,
    dataset_sha256: str,
    pair_index: int,
    bucket: str,
) -> dict:
    record = torch.load(path, map_location="cpu", weights_only=False)
    metadata = record.get("metadata", {})
    checks = {
        "protocol_sha256": metadata.get("protocol_sha256") == protocol_sha256,
        "dataset_sha256": metadata.get("source_dataset_sha256") == dataset_sha256,
        "pair_index": int(metadata.get("pair_index", -1)) == pair_index,
        "bucket": metadata.get("bucket") == bucket,
    }
    if not all(checks.values()):
        raise AssertionError(f"Existing record validation failed for {path}: {checks}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS)
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--remask", choices=["low_confidence", "random"], default="low_confidence")
    parser.add_argument("--mask-id", type=int, default=MASK_ID_DEFAULT)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-end", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--store-decoded", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.resume:
        raise ValueError("Choose only one of --overwrite or --resume")
    if not (0 <= args.pair_start < args.pair_end <= 500):
        raise ValueError("Require 0 <= pair-start < pair-end <= 500")
    if args.steps <= 0 or args.gen_length <= 0:
        raise ValueError("steps and gen-length must be positive")
    if args.steps != 128 or args.gen_length != 128:
        raise ValueError("Official A/D path requires steps=128 and gen-length=128")
    args.layers = sorted(set(args.layers))
    args.fractions = sorted(set(float(value) for value in args.fractions))
    fractions_to_steps(tuple(args.fractions), args.steps)

    dataset_path = Path(args.dataset).resolve()
    model_path = Path(args.model_path).resolve()
    out_root = Path(args.out_root).resolve()
    rows = read_jsonl(dataset_path)
    by_pair = validate_dataset(
        rows,
        dataset_path,
        args.expected_dataset_sha256,
    )
    dataset_sha256 = sha256_file(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Iterative extraction requires CUDA")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenized_mask = tokenize_without_specials(tokenizer, MASK_TOKEN)
    if tokenized_mask != [args.mask_id]:
        raise AssertionError(
            f"Mask token mismatch: tokenizer={tokenized_mask}, expected={[args.mask_id]}"
        )

    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    blocks, blocks_attr = get_blocks(model)
    if max(args.layers) >= len(blocks):
        raise ValueError(
            f"Requested layer {max(args.layers)} but model has {len(blocks)} blocks"
        )

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer in args.layers:
        def make_hook(layer_number: int):
            def hook(_module, _inputs, output):
                captured[layer_number] = output_to_hidden(output).detach()
            return hook
        hooks.append(blocks[layer].register_forward_hook(make_hook(layer)))

    protocol_core = {
        "protocol": "RRAE_CANONICAL_DIJA_ITERATIVE_HIDDEN_EXTRACTION_V1",
        "dataset_sha256": dataset_sha256,
        "model_path": str(model_path),
        "model_config_name_or_path": getattr(model.config, "_name_or_path", None),
        "blocks_attr": blocks_attr,
        "num_blocks": len(blocks),
        "layers": args.layers,
        "scopes": SCOPES,
        "fractions": args.fractions,
        "fraction_to_step": "dynamic_per_case_round_fraction_times_trajectory_steps",
        "trajectory_policy": {
            "A_D": "128 appended assistant masks; 128 one-token transfers",
            "B_C": "canonical DIJA; zero appended masks; one transfer per embedded template mask",
        },
        "ad_steps": args.steps,
        "ad_gen_length": args.gen_length,
        "temperature": args.temperature,
        "remask": args.remask,
        "fill_all_masks": True,
        "mask_token": MASK_TOKEN,
        "mask_id": args.mask_id,
        "record_before_transfer": True,
        "hidden_saved_dtype": "float16",
        "empty_scope_representation": "finite_float16_zero_vector",
        "pool": "mean_over_current_scope_positions",
        "generation_seed_policy": "base_seed*10000 + pair_index*4 + group_id",
        "base_seed": args.base_seed,
        "scientific_status": "official_rebuild_after_canonical_DIJA_correction",
    }
    protocol_sha256 = canonical_json_sha256(protocol_core)

    shard_name = f"pairs_{args.pair_start:03d}_{args.pair_end:03d}"
    shard_dir = out_root / shard_name
    records_dir = shard_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(protocol_core, shard_dir / "PROTOCOL.json")

    print("===== RRAE CANONICAL DIJA ITERATIVE EXTRACTION =====", flush=True)
    print("dataset:", dataset_path, flush=True)
    print("dataset_sha256:", dataset_sha256, flush=True)
    print("model_path:", model_path, flush=True)
    print("blocks_attr:", blocks_attr, flush=True)
    print("layers:", args.layers, flush=True)
    print("fractions:", args.fractions, flush=True)
    print("fraction_to_step:", protocol_core["fraction_to_step"], flush=True)
    print("pair_range:", [args.pair_start, args.pair_end], flush=True)
    print("rows:", (args.pair_end - args.pair_start) * 4, flush=True)
    print("protocol_sha256:", protocol_sha256, flush=True)

    manifest_rows: list[dict] = []
    completed = 0
    skipped_existing = 0
    selected_pairs = list(range(args.pair_start, args.pair_end))

    try:
        for pair_index in selected_pairs:
            for bucket in ("A", "B", "C", "D"):
                row = by_pair[pair_index][bucket]
                record_path = (
                    records_dir
                    / bucket
                    / f"{row['id']}__seed{args.base_seed}.pt"
                )

                if record_path.exists():
                    if args.overwrite:
                        record_path.unlink()
                    elif args.resume:
                        record = validate_existing_record(
                            record_path,
                            protocol_sha256,
                            dataset_sha256,
                            pair_index,
                            bucket,
                        )
                        skipped_existing += 1
                    else:
                        raise FileExistsError(
                            f"Record exists; use --resume or --overwrite: {record_path}"
                        )
                if not record_path.exists():
                    record = extract_case(
                        model=model,
                        tokenizer=tokenizer,
                        hooks=hooks,
                        captured=captured,
                        row=row,
                        pair_index=pair_index,
                        dataset_path=dataset_path,
                        dataset_sha256=dataset_sha256,
                        protocol_core=protocol_core,
                        protocol_sha256=protocol_sha256,
                        device=device,
                        args=args,
                    )
                    atomic_torch_save(record, record_path)
                    completed += 1

                manifest_rows.append(
                    {
                        "pair_index": pair_index,
                        "bucket": bucket,
                        "source_row_id": row["id"],
                        "generation_seed": stable_generation_seed(
                            args.base_seed, pair_index, bucket
                        ),
                        "record_path": str(record_path),
                        "record_path_relative": str(record_path.relative_to(out_root)),
                        "record_sha256": sha256_file(record_path),
                        "protocol_sha256": protocol_sha256,
                        "dataset_sha256": dataset_sha256,
                    }
                )
                done = len(manifest_rows)
                total = len(selected_pairs) * 4
                if done == 1 or done % 10 == 0 or done == total:
                    print(
                        f"progress: {done}/{total} newly_completed={completed} "
                        f"resumed={skipped_existing}",
                        flush=True,
                    )
    finally:
        for hook in hooks:
            hook.remove()

    manifest_path = shard_dir / "manifest.jsonl"
    temporary_manifest = manifest_path.with_suffix(
        manifest_path.suffix + f".tmp.{os.getpid()}"
    )
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for item in manifest_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temporary_manifest, manifest_path)

    summary = {
        "protocol": "RRAE_CANONICAL_DIJA_ITERATIVE_HIDDEN_EXTRACTION_V1",
        "protocol_sha256": protocol_sha256,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "model_path": str(model_path),
        "pair_start": args.pair_start,
        "pair_end": args.pair_end,
        "pair_count": len(selected_pairs),
        "row_count": len(manifest_rows),
        "newly_completed": completed,
        "resumed_existing": skipped_existing,
        "group_counts": dict(Counter(item["bucket"] for item in manifest_rows)),
        "manifest": str(manifest_path),
        "validation_pass": (
            len(manifest_rows) == len(selected_pairs) * 4
            and Counter(item["bucket"] for item in manifest_rows)
            == Counter({bucket: len(selected_pairs) for bucket in "ABCD"})
        ),
    }
    atomic_json_write(summary, shard_dir / "SHARD_SUMMARY.json")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["validation_pass"]:
        raise SystemExit(1)
    print("ITERATIVE EXTRACTION: PASS", flush=True)


if __name__ == "__main__":
    main()
