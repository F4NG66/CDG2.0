#!/usr/bin/env python
import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModel


ROOT = Path("/path/to/rrae_steering_work_v2")
DEFAULT_DATASET = ROOT / "data/abcd/ABCD_clean_v2_DIJA_Qwen_A_B_C_D.jsonl"
DEFAULT_OUT_DIR = ROOT / "hidden_states/abcd_v2"
DEFAULT_MODEL = Path("/path/to/LLaDA-8B-Instruct")

MASK_TOKEN = "<|mdm_mask|>"
MASK_ID_DEFAULT = 126336

GROUP_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_GROUP = {v: k for k, v in GROUP_TO_ID.items()}


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def expand_mask_n(text: str) -> str:
    """
    DIJA runtime-compatible expansion:
      <mask:N> -> <|mdm_mask|> repeated N times
    """
    def repl(m):
        n = int(m.group(1))
        return MASK_TOKEN * n

    return re.sub(r"<mask:(\d+)>", repl, str(text))


def make_chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback only if chat template fails.
        return f"User: {prompt}\nAssistant:"


def positions_from_offsets(tokenizer, chat_text: str, prompt_text: str):
    """
    Locate user prompt span inside chat_text using tokenizer offsets.
    Returns token indices in the tokenized chat_text.
    """
    enc = tokenizer(
        chat_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    offsets = enc.get("offset_mapping", None)
    ids = enc["input_ids"]

    start_char = chat_text.find(prompt_text)
    if start_char < 0:
        return None, ids

    end_char = start_char + len(prompt_text)

    if offsets is None:
        return None, ids

    positions = []
    for i, (a, b) in enumerate(offsets):
        # Keep tokens overlapping the prompt substring.
        if b > start_char and a < end_char:
            positions.append(i)

    if not positions:
        return None, ids

    return positions, ids


def find_subsequence_positions(full_ids, sub_ids):
    """
    Fallback span matching if offsets fail.
    """
    if not sub_ids:
        return None

    n = len(sub_ids)
    for i in range(0, len(full_ids) - n + 1):
        if full_ids[i:i+n] == sub_ids:
            return list(range(i, i+n))
    return None


def prompt_positions(tokenizer, chat_text: str, prompt_text: str):
    """
    Robustly locate prompt-token positions inside chat template.
    """
    pos, chat_ids = positions_from_offsets(tokenizer, chat_text, prompt_text)
    if pos is not None:
        return pos, chat_ids

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    pos = find_subsequence_positions(chat_ids, prompt_ids)
    if pos is not None:
        return pos, chat_ids

    raise RuntimeError("Could not locate prompt span inside chat template.")


def select_first_frac_positions(positions, frac: float):
    """
    Match old RRAE/SAE-style fraction logic: use first fraction of the selected region.
    """
    positions = list(positions)
    if len(positions) == 0:
        return []
    k = max(1, int(math.ceil(len(positions) * frac)))
    return positions[:k]


def normalize_pair_key(row):
    """
    Build a stable A/B/C/D pair key.

    The old vector extraction script groups by case number after stripping
    the first character from case_id. To preserve that behavior, we assign
    case_id = A0000/B0000/C0000/D0000 using paired_A_id order.
    """
    if row.get("paired_A_id"):
        return str(row["paired_A_id"])
    if row.get("bucket") == "A":
        return str(row["id"])
    return str(row.get("id"))


def bucket_to_variant(bucket):
    return {
        "A": "A_harmful_clean",
        "B": "B_harmful_injected",
        "C": "C_benign_injected",
        "D": "D_benign_clean",
    }[bucket]


def row_content_type(bucket):
    return "harmful" if bucket in ["A", "B"] else "benign"


def build_input_and_regions(tokenizer, row, mask_id: int, gen_length: int):
    bucket = row["bucket"]
    raw_prompt = str(row["prompt"])

    if bucket in ["B", "C"]:
        prompt_for_model = expand_mask_n(raw_prompt)
    else:
        prompt_for_model = raw_prompt

    chat_text = make_chat_text(tokenizer, prompt_for_model)

    prompt_pos, chat_ids = prompt_positions(tokenizer, chat_text, prompt_for_model)

    # Append output mask tokens to mimic LLaDA diffusion generation context.
    input_ids = list(chat_ids) + [mask_id] * gen_length
    output_region = (len(chat_ids), len(chat_ids) + gen_length)

    if bucket in ["A", "D"]:
        actual_scope = "harm"
        selected_region_positions = prompt_pos
        regions = {
            "template": None,
            "harm": (min(prompt_pos), max(prompt_pos) + 1),
            "output": output_region,
        }
    elif bucket in ["B", "C"]:
        actual_scope = "tpl_mask"
        # Only mask tokens inside the user prompt/template span.
        prompt_pos_set = set(prompt_pos)
        selected_region_positions = [
            i for i in prompt_pos
            if i < len(chat_ids) and chat_ids[i] == mask_id and i in prompt_pos_set
        ]

        if not selected_region_positions:
            # Fallback: sometimes tokenizer may not expose mask as exactly mask_id
            # in this environment. Fail loudly because tpl_mask is essential.
            raise RuntimeError(
                f"No tpl_mask positions found for row id={row.get('id')} bucket={bucket}"
            )

        regions = {
            "template": (min(prompt_pos), max(prompt_pos) + 1),
            "harm": None,
            "output": output_region,
        }
    else:
        raise ValueError(f"Unknown bucket: {bucket}")

    return {
        "input_ids": input_ids,
        "prompt_for_model": prompt_for_model,
        "chat_text": chat_text,
        "selected_region_positions": selected_region_positions,
        "regions": regions,
        "actual_scope": actual_scope,
    }


def get_block(model, layer: int):
    """
    LLaDA local model uses model.model.transformer.blocks[layer].
    """
    return model.model.transformer.blocks[layer]


def extract_one(model, input_ids, selected_positions, layers, device):
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    captured = {}

    handles = []

    def make_hook(layer):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer] = h.detach()
        return hook

    for layer in layers:
        handles.append(get_block(model, layer).register_forward_hook(make_hook(layer)))

    try:
        with torch.inference_mode():
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    out = {}
    pos = torch.tensor(selected_positions, dtype=torch.long, device=device)

    for layer in layers:
        if layer not in captured:
            raise RuntimeError(f"Did not capture layer {layer}")

        h = captured[layer][0]  # [seq, hidden]
        if pos.max().item() >= h.shape[0]:
            raise RuntimeError(
                f"Position out of range: max_pos={pos.max().item()} seq_len={h.shape[0]}"
            )

        vec = h.index_select(0, pos).float().mean(dim=0).detach().cpu()
        out[layer] = vec

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL))
    ap.add_argument("--layers", nargs="+", type=int, default=[11, 16])
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--mask-id", type=int, default=MASK_ID_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(dataset_path)
    if args.limit is not None:
        rows = rows[:args.limit]

    print("dataset:", dataset_path, flush=True)
    print("rows:", len(rows), flush=True)
    print("layers:", args.layers, flush=True)
    print("frac:", args.frac, flush=True)
    print("gen_length:", args.gen_length, flush=True)
    print("mask_id:", args.mask_id, flush=True)

    assert len(rows) > 0

    # Stable pair-index map based on paired_A_id order.
    pair_keys = []
    for r in rows:
        pk = normalize_pair_key(r)
        if pk not in pair_keys:
            pair_keys.append(pk)
    pair_to_idx = {pk: i for i, pk in enumerate(pair_keys)}

    print("num_pairs_seen:", len(pair_to_idx), flush=True)
    print("bucket counts:", Counter(r["bucket"] for r in rows), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    H_by_layer = {layer: [] for layer in args.layers}
    metadata_by_layer = {layer: [] for layer in args.layers}
    skipped = []

    for idx, row in enumerate(rows):
        bucket = row["bucket"]
        pair_key = normalize_pair_key(row)
        pair_idx = pair_to_idx[pair_key]
        case_id = f"{bucket}{pair_idx:04d}"

        try:
            built = build_input_and_regions(
                tokenizer=tokenizer,
                row=row,
                mask_id=args.mask_id,
                gen_length=args.gen_length,
            )

            frac_positions = select_first_frac_positions(
                built["selected_region_positions"],
                args.frac,
            )

            if not frac_positions:
                raise RuntimeError("empty selected positions after frac")

            vecs = extract_one(
                model=model,
                input_ids=built["input_ids"],
                selected_positions=frac_positions,
                layers=args.layers,
                device=device,
            )

            for layer in args.layers:
                h = vecs[layer].float().contiguous()

                if h.ndim != 1 or h.shape[0] != 4096:
                    raise RuntimeError(f"bad hidden shape: {tuple(h.shape)}")

                if not torch.isfinite(h).all():
                    raise RuntimeError("non_finite")

                H_by_layer[layer].append(h)

                metadata_by_layer[layer].append({
                    "case_id": case_id,
                    "source_row_id": row.get("id"),
                    "paired_A_id": row.get("paired_A_id"),
                    "paired_D_id": row.get("paired_D_id"),
                    "pair_index": pair_idx,
                    "variant": bucket_to_variant(bucket),
                    "group": bucket,
                    "group_id": GROUP_TO_ID[bucket],
                    "content_type": row_content_type(bucket),
                    "label": row.get("label"),
                    "domain": row.get("domain"),
                    "has_template": bucket in ["B", "C"],
                    "attack_method": "DIJA_Qwen_refined_maskN" if bucket in ["B", "C"] else "none",
                    "is_neutral": bucket in ["C", "D"],
                    "model_name": "llada",
                    "seed": 0,
                    "source_dataset": str(dataset_path),
                    "scope": "input_region",
                    "actual_scope": built["actual_scope"],
                    "frac": args.frac,
                    "layer": layer,
                    "regions": built["regions"],
                    "num_region_positions_total": len(built["selected_region_positions"]),
                    "num_region_positions_used": len(frac_positions),
                    "input_seq_len": len(built["input_ids"]),
                    "gen_length": args.gen_length,
                    "mask_id": args.mask_id,
                })

        except Exception as e:
            skipped.append({
                "idx": idx,
                "row_id": row.get("id"),
                "bucket": bucket,
                "pair_key": pair_key,
                "error": repr(e),
            })
            print(f"[SKIP] idx={idx} id={row.get('id')} bucket={bucket} err={e}", flush=True)
            continue

        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"[progress] {idx+1}/{len(rows)} skipped={len(skipped)}", flush=True)

    for layer in args.layers:
        frac_str = str(args.frac).replace(".", "p")
        out_path = out_dir / f"H_input_region_L{layer}_f{frac_str}.pt"

        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists. Use --overwrite: {out_path}")

        Hs = H_by_layer[layer]
        metadata = metadata_by_layer[layer]

        if not Hs:
            raise RuntimeError(f"No hidden states collected for layer {layer}")

        H = torch.stack(Hs, dim=0).float()
        groups = torch.tensor([m["group_id"] for m in metadata], dtype=torch.long)
        is_injected = torch.tensor([1 if m["has_template"] else 0 for m in metadata], dtype=torch.long)
        is_harmful = torch.tensor([1 if m["content_type"] == "harmful" else 0 for m in metadata], dtype=torch.long)

        counts = {g: int((groups == gid).sum()) for g, gid in GROUP_TO_ID.items()}
        actual_scope_counts = Counter(m["actual_scope"] for m in metadata)

        payload = {
            "H": H,
            "groups": groups,
            "is_injected": is_injected,
            "is_harmful": is_harmful,
            "metadata": metadata,
            "config": {
                "source_dataset": str(dataset_path),
                "scope": "input_region",
                "frac": args.frac,
                "layer": layer,
                "layers_extracted_together": args.layers,
                "input_dim": 4096,
                "group_to_id": GROUP_TO_ID,
                "variant_to_group": {
                    "A_harmful_clean": "A",
                    "B_harmful_injected": "B",
                    "C_benign_injected": "C",
                    "D_benign_clean": "D",
                },
                "num_rows_input": len(rows),
                "num_rows_saved": int(H.shape[0]),
                "num_skipped": len(skipped),
                "hidden_dtype_saved": "float32",
                "mask_expansion": "<mask:N> -> <|mdm_mask|> repeated N for B/C",
                "input_region_logic": {
                    "A": "actual_scope=harm",
                    "D": "actual_scope=harm",
                    "B": "actual_scope=tpl_mask",
                    "C": "actual_scope=tpl_mask",
                },
                "gen_length": args.gen_length,
                "mask_id": args.mask_id,
            },
            "skipped": skipped,
            "global_balanced_dropped": [],
        }

        torch.save(payload, out_path)

        print("=" * 100, flush=True)
        print("[SAVED]", out_path, flush=True)
        print("H:", tuple(H.shape), flush=True)
        print("counts:", counts, flush=True)
        print("is_injected:", int(is_injected.sum()), flush=True)
        print("is_harmful:", int(is_harmful.sum()), flush=True)
        print("actual_scope_counts:", dict(actual_scope_counts), flush=True)
        print("skipped:", len(skipped), flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
