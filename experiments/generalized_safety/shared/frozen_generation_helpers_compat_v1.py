"""Generation-only functions copied verbatim from hash-bound canonical source; no extraction code."""
from __future__ import annotations
import re
import torch
import torch.nn.functional as F
MASK_TOKEN = "<|mdm_mask|>"

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
    harm_span = None if harm_start < 0 else (harm_start, harm_end)

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
