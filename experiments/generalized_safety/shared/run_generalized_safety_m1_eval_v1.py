#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
GS = ROOT / "analysis_output/rrae_development/generalized_safety_v1"

MODEL = Path(__import__("os").environ["CDG_LLADA_MODEL"])

MANIFEST = GS / "GENERALIZED_SAFETY_M1_EVAL_MANIFEST_V1.jsonl"
PREFLIGHT = GS / "GENERALIZED_SAFETY_M1_EVAL_PREFLIGHT_FREEZE_V1.json"

DIRECTIONS = (
    GS
    / "generalized_safety_directions_v1/"
      "GENERALIZED_SAFETY_DIRECTIONS.pt"
)

HIST_PATH = (
    ROOT
    / "scripts/official_rrae/"
      "run_canonical_train_layer_matched_safety_dose_response_canary_v1.py"
)

BASELINE_PATH = (
    ROOT
    / "scripts/rrae_development/"
      "run_generalized_safety_direction_construction_baselines_v2.py"
)

HELPER_PATH = (
    ROOT
    / "scripts/official_rrae/"
      "extract_iterative_hidden_canonical_dija_v1.py"
)

EXPECTED_DIRECTIONS_SHA = (
    "605739afd0959289af99c4b85731a5e3bf6fe6d275884fb5df8f505f3e4da6a5"
)

EXPECTED_HIST_SHA = (
    "cca5bb26111932c78815ef5a75bdc4898ce9a1a023550c1b9a06dbcbacdae224"
)

EXPECTED_BASELINE_SHA = (
    "fa5ad065c0c1e8b69228bc12356e8ee1cdb10c38ea864d2e21f3d8bdabbfae62"
)

EXPECTED_HELPER_SHA = (
    "67832c5cb193699b2fe83b545b6013b290dc8cddbd8d35504c983bda1cb92680"
)

LAYER = 16
MASK_ID = 126336
GEN_LENGTH = 128
TEMPERATURE = 0.2
REMASK = "low_confidence"


def require(x, msg):
    if not x:
        raise RuntimeError(msg)


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_sha256(t):
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(
        x.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(obj, path):
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_jsonl(rows, path):
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def standard_decode(tokenizer, x, regions):
    require(regions["output"] is not None, "Missing standard output region")
    lo, hi = regions["output"]
    ids = [int(v) for v in x[0, lo:hi].tolist()]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text, ids


def build_input(item, tokenizer, helper, baseline):
    family = item["attack_family"]

    if family == "DIJA":
        adapter = {
            "bucket": str(item["condition"]),
            "id": (
                item.get("canonical_record_id")
                or f"{item['condition']}_{int(item['pair_id']):04d}"
            ),
            "prompt": item["prompt"],
            "clean_prompt": item["clean_semantic_request"],
        }

        x_cpu, regions, metadata = helper.build_inputs_and_regions(
            tokenizer,
            adapter,
            MASK_ID,
            GEN_LENGTH,
        )

        require(
            metadata["generation_mode"] == "canonical_dija_infill",
            "DIJA runtime mode changed",
        )
        require(
            int(metadata["appended_output_mask_count"]) == 0,
            "DIJA unexpectedly appended output masks",
        )
        require(
            regions["output"] is None,
            "DIJA output region unexpectedly exists",
        )

        return x_cpu, regions, metadata

    if family == "RENELLM":
        x_cpu, regions, metadata = baseline.direct_standard_builder(
            tokenizer,
            item["prompt"],
            torch,
        )

        require(
            metadata["generation_mode"] == "standard_assistant_generation",
            "ReNeLLM runtime mode changed",
        )
        require(
            int(metadata["appended_output_mask_count"]) == 128,
            "ReNeLLM output-mask count changed",
        )
        require(regions["output"] is not None, "ReNeLLM output region missing")

        return x_cpu, regions, metadata

    raise RuntimeError(f"Unsupported family: {family}")


def generate_one(
    *,
    item,
    model,
    tokenizer,
    helper,
    baseline,
    banks,
    device,
):
    set_seed(int(item["generation_seed"]))

    x_cpu, regions, metadata = build_input(
        item,
        tokenizer,
        helper,
        baseline,
    )

    x = x_cpu.to(device)
    initial_x = x.detach().clone()

    initial_mask = x == MASK_ID
    initial_count = int(initial_mask.sum().item())

    require(initial_count > 0, "No fillable masks")

    family = item["attack_family"]

    if family == "DIJA":
        trajectory_steps = initial_count
    else:
        require(initial_count == 128, "ReNeLLM initial mask count drift")
        trajectory_steps = 128

    for bank in banks.values():
        bank.reset_run(
            layer=None,
            beta=0.0,
            active_steps=[],
        )

    vector_key = item["vector_key"]

    selected_bank = None

    if vector_key is not None:
        selected_bank = banks[vector_key]

        active_steps = list(range(1, trajectory_steps + 1))

        selected_bank.reset_run(
            layer=LAYER,
            beta=float(item["beta"]),
            active_steps=active_steps,
        )
    else:
        active_steps = []

    transfer_schedule = helper.get_num_transfer_tokens(
        initial_mask,
        trajectory_steps,
    )

    attention_mask = torch.ones_like(
        x,
        dtype=torch.long,
        device=device,
    )

    started = time.monotonic()

    for global_step in range(1, trajectory_steps + 1):

        current_mask = x == MASK_ID

        if selected_bank is not None:
            selected_bank.set_step_context(
                global_step=global_step,
                current_mask=current_mask,
                input_ids_sha256=tensor_sha256(x),
            )

        with torch.inference_mode():
            output = model(
                x,
                attention_mask=attention_mask,
            )

        logits = output.logits if hasattr(output, "logits") else output[0]

        logits_noised = helper.add_gumbel_noise(
            logits,
            TEMPERATURE,
        )

        prediction = torch.argmax(
            logits_noised,
            dim=-1,
        )

        confidence = helper.confidence_for_predictions(
            logits,
            prediction,
            REMASK,
        )

        prediction = torch.where(
            current_mask,
            prediction,
            x,
        )

        neg_inf = torch.tensor(
            -np.inf,
            device=device,
            dtype=confidence.dtype,
        )

        confidence = torch.where(
            current_mask,
            confidence,
            neg_inf,
        )

        transfer = torch.zeros_like(
            prediction,
            dtype=torch.bool,
        )

        for batch_index in range(confidence.shape[0]):
            k = int(
                transfer_schedule[
                    batch_index,
                    global_step - 1
                ]
            )

            if k > 0:
                _, selected = torch.topk(
                    confidence[batch_index],
                    k=k,
                )

                transfer[
                    batch_index,
                    selected
                ] = True

        x[transfer] = prediction[transfer]


    require(
        int((x == MASK_ID).sum().item()) == 0,
        "Unresolved masks remain",
    )

    if vector_key is None:
        application_count = 0
        step_audits = []
    else:
        application_count = selected_bank.application_count
        step_audits = selected_bank.step_audits

        require(
            application_count == trajectory_steps,
            (
                f"Persistent application mismatch: "
                f"{application_count} != {trajectory_steps}"
            ),
        )

        beta = float(item["beta"])
        tolerance = max(0.25, 0.02 * abs(beta))

        for audit in step_audits:
            require(
                abs(float(audit["target_mean_delta_l2"]) - beta)
                <= tolerance,
                "M1 delta magnitude audit failed",
            )

            require(
                float(audit["target_min_cosine_to_v_safety"]) >= 0.995,
                "M1 direction cosine audit failed",
            )

            require(
                float(audit["non_target_max_abs_delta"]) == 0.0,
                "Non-target hidden state changed",
            )


    if family == "DIJA":
        decoded_output, final_template = hist.decode_final(
            tokenizer,
            x,
            regions,
            metadata,
        )

        generated_ids = None
    else:
        decoded_output, generated_ids = standard_decode(
            tokenizer,
            x,
            regions,
        )

        final_template = None


    return {
        "schema":
            "GENERALIZED_SAFETY_M1_RESULT_V1",

        "item_id":
            item["item_id"],

        "split":
            "EVAL",

        "attack_family":
            family,

        "condition":
            item["condition"],

        "pair_id":
            int(item["pair_id"]),

        "domain":
            item["domain"],

        "runtime_mode":
            item["runtime_mode"],

        "prompt_sha256":
            item["prompt_sha256"],

        "generation_seed":
            int(item["generation_seed"]),

        "steering_condition":
            item["steering_condition"],

        "vector_key":
            vector_key,

        "layer_zero_based":
            item["layer_zero_based"],

        "token_scope":
            item["token_scope"],

        "schedule":
            item["schedule"],

        "beta":
            float(item["beta"]),

        "trajectory_steps":
            trajectory_steps,

        "initial_mask_count":
            initial_count,

        "application_count":
            application_count,

        "step_audits":
            step_audits,

        "initial_ids_sha256":
            tensor_sha256(initial_x),

        "final_ids_sha256":
            tensor_sha256(x),

        "decoded_output":
            decoded_output,

        "decoded_output_sha256":
            sha_text(decoded_output),

        "generated_token_ids":
            generated_ids,

        "final_template":
            final_template,

        "elapsed_seconds":
            time.monotonic() - started,

        "judge_used":
            False,

        "behavioral_label_seen":
            False,
    }


def main():
    global hist

    ap = argparse.ArgumentParser()

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expected-preflight-sha", required=True)
    ap.add_argument("--pair-id", type=int, required=True)

    args = ap.parse_args()

    out = Path(args.out_dir)

    require(not out.exists(), f"Output exists: {out}")

    require(
        sha_file(PREFLIGHT) == args.expected_preflight_sha,
        "Preflight SHA mismatch",
    )

    require(
        sha_file(DIRECTIONS) == EXPECTED_DIRECTIONS_SHA,
        "Direction artifact SHA mismatch",
    )

    require(
        sha_file(HIST_PATH) == EXPECTED_HIST_SHA,
        "Historical runner SHA mismatch",
    )

    require(
        sha_file(BASELINE_PATH) == EXPECTED_BASELINE_SHA,
        "Baseline helper SHA mismatch",
    )

    require(
        sha_file(HELPER_PATH) == EXPECTED_HELPER_SHA,
        "Canonical helper SHA mismatch",
    )

    preflight = json.loads(PREFLIGHT.read_text())

    require(preflight["status"] == "PASS", "Preflight not PASS")
    require(preflight["smoke_authorized"] is True, "Smoke not authorized")
    require(preflight["full_eval_authorized"] is False, "Full unexpectedly authorized")

    require(
        int(args.pair_id) == int(preflight["smoke_pair_id"]),
        "Only frozen smoke pair is allowed",
    )

    manifest = [
        json.loads(line)
        for line in MANIFEST.read_text().splitlines()
        if line.strip()
    ]

    require(
        sha_file(MANIFEST) == preflight["manifest_sha256"],
        "Manifest SHA mismatch",
    )

    items = [
        x for x in manifest
        if int(x["pair_id"]) == int(args.pair_id)
    ]

    require(len(items) == 16, f"Expected 16 smoke items, got {len(items)}")

    cell_counts = {}
    for x in items:
        key = (x["attack_family"], x["condition"])
        cell_counts[key] = cell_counts.get(key, 0) + 1

    require(
        cell_counts == {
            ("DIJA", "B"): 4,
            ("DIJA", "C"): 4,
            ("RENELLM", "B"): 4,
            ("RENELLM", "C"): 4,
        },
        f"Smoke condition drift: {cell_counts}",
    )

    directions = torch.load(
        DIRECTIONS,
        map_location="cpu",
        weights_only=False,
    )

    require(int(directions["layer"]) == 16, "Direction layer drift")

    vectors = {
        "V_DIJA": directions["v_DIJA"].detach().float().view(-1),
        "V_RENELLM": directions["v_RENELLM"].detach().float().view(-1),
        "V_ALL": directions["v_ALL"].detach().float().view(-1),
    }

    for key, v in vectors.items():
        require(v.numel() == 4096, f"{key} dimension")
        require(torch.isfinite(v).all(), f"{key} nonfinite")
        require(
            math.isclose(float(v.norm().item()), 1.0, abs_tol=1e-5),
            f"{key} not unit norm",
        )

    hist = load_module("m1_hist", HIST_PATH)
    baseline = load_module("m1_baseline", BASELINE_PATH)
    helper = load_module("m1_helper", HELPER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "CUDA required")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL),
        trust_remote_code=True,
        local_files_only=True,
    )

    require(
        helper.tokenize_without_specials(tokenizer, "<|mdm_mask|>")
        == [MASK_ID],
        "Mask-token mismatch",
    )

    model = AutoModel.from_pretrained(
        str(MODEL),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)

    blocks, blocks_attr, candidates = hist.resolve_transformer_blocks(model)

    require(len(blocks) == 32, "Expected 32 blocks")

    banks = {
        key: hist.SafetyLayerLocationHookBank(
            blocks,
            vector,
            (LAYER,),
        )
        for key, vector in vectors.items()
    }

    out.mkdir(parents=True)

    results = []

    try:
        for item in items:
            result = generate_one(
                item=item,
                model=model,
                tokenizer=tokenizer,
                helper=helper,
                baseline=baseline,
                banks=banks,
                device=device,
            )

            results.append(result)

            print(
                f"completed "
                f"family={result['attack_family']} "
                f"condition={result['condition']} "
                f"steering={result['steering_condition']} "
                f"steps={result['trajectory_steps']} "
                f"applications={result['application_count']}",
                flush=True,
            )

    finally:
        for bank in banks.values():
            bank.close()


    require(len(results) == 16, "Smoke result count mismatch")


    # Paired-randomness audit:
    grouped = {}

    for r in results:
        key = (
            r["attack_family"],
            r["condition"],
            r["pair_id"],
        )
        grouped.setdefault(key, []).append(r)

    for key, group in grouped.items():
        require(len(group) == 4, f"{key}: expected four conditions")

        seeds = {r["generation_seed"] for r in group}
        initial_hashes = {r["initial_ids_sha256"] for r in group}

        require(len(seeds) == 1, f"{key}: generation seed drift")
        require(len(initial_hashes) == 1, f"{key}: initial token drift")


    results_path = out / "M1_SMOKE_RESULTS_PRIVATE.jsonl"
    atomic_jsonl(results, results_path)


    audit = {
        "protocol":
            "GENERALIZED_SAFETY_M1_SMOKE_AUDIT_V1",

        "status":
            "PASS",

        "pair_id":
            int(args.pair_id),

        "result_count":
            len(results),

        "source_cells":
            4,

        "conditions_per_source":
            4,

        "runtime_families": [
            "DIJA",
            "RENELLM",
        ],

        "layer_zero_based":
            16,

        "scope":
            "current_mask",

        "schedule":
            "persistent",

        "intervention":
            "h_new = h + beta * v",

        "all_masks_resolved":
            True,

        "paired_seed_audit_pass":
            True,

        "paired_initial_token_audit_pass":
            True,

        "steering_application_count_audit_pass":
            True,

        "steering_delta_magnitude_audit_pass":
            True,

        "steering_direction_cosine_audit_pass":
            True,

        "non_target_unchanged_audit_pass":
            True,

        "behavioral_efficacy_inspected":
            False,

        "judge_used":
            False,

        "maskforge_used":
            False,

        "M2_used":
            False,

        "M3_used":
            False,

        "model": {
            "blocks_attr": blocks_attr,
            "num_blocks": len(blocks),
            "resolver_candidates": candidates,
        },

        "source_hashes": {
            "preflight":
                sha_file(PREFLIGHT),

            "manifest":
                sha_file(MANIFEST),

            "directions":
                sha_file(DIRECTIONS),

            "historical_runner":
                sha_file(HIST_PATH),

            "baseline_builder":
                sha_file(BASELINE_PATH),

            "canonical_helper":
                sha_file(HELPER_PATH),
        },

        "results_sha256":
            sha_file(results_path),
    }

    audit_path = out / "M1_SMOKE_AUDIT.json"
    atomic_json(audit, audit_path)


    freeze = {
        "protocol":
            "GENERALIZED_SAFETY_M1_SMOKE_FREEZE_V1",

        "status":
            "PASS",

        "pair_id":
            int(args.pair_id),

        "runtime_validation_pass":
            True,

        "behavioral_selection_used":
            False,

        "behavioral_efficacy_inspected":
            False,

        "judge_used":
            False,

        "maskforge_used":
            False,

        "full_eval_authorized":
            False,

        "authorized_next":
            (
                "Create a separate full-evaluation authorization freeze "
                "using only this runtime/audit PASS, without using smoke "
                "behavioral outcomes."
            ),

        "artifacts": {
            "results_sha256":
                sha_file(results_path),

            "audit_sha256":
                sha_file(audit_path),
        },
    }

    freeze_path = out / "M1_SMOKE_FREEZE.json"
    atomic_json(freeze, freeze_path)

    print("GENERALIZED_SAFETY_M1_SMOKE_STATUS=PASS")
    print("PAIR_ID=", args.pair_id)
    print("RESULT_COUNT=16")
    print("RUNTIME_AUDIT=PASS")
    print("BEHAVIORAL_EFFICACY_INSPECTED=False")
    print("JUDGE_USED=False")
    print("MASKFORGE_USED=False")
    print("FULL_EVAL_AUTHORIZED=False")
    print("RESULTS_SHA256=", sha_file(results_path))
    print("SMOKE_FREEZE_SHA256=", sha_file(freeze_path))


if __name__ == "__main__":
    main()
