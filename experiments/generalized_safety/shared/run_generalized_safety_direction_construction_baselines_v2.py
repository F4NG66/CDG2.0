#!/usr/bin/env python3
"""Fail-closed V2 Stage C1 baseline runner.

Standard SAFE_A/ReNeLLM inputs use a direct builder. Only DIJA uses the
canonical DIJA region/mask-span builder. Importing this module has no GPU or
model side effects.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
SAFE = ROOT / "analysis_output/rrae_development/generalized_safety_v1"
MODEL = Path(__import__("os").environ["CDG_LLADA_MODEL"])
MANIFEST = SAFE / "SAFETY_DIRECTION_CONSTRUCTION_BASELINE_MANIFEST_V1.jsonl"
MANIFEST_AUDIT = SAFE / "SAFETY_DIRECTION_CONSTRUCTION_BASELINE_MANIFEST_AUDIT_V1.json"
FAILURE_AUDIT = SAFE / "SAFETY_DIRECTION_CONSTRUCTION_BASELINE_V1_FAILURE_AUDIT.json"
PRE_RUN_AUDIT = SAFE / "SAFETY_DIRECTION_CONSTRUCTION_BASELINE_V2_PRE_RUN_AUDIT.json"
OUTPUT = SAFE / "safety_direction_construction_baselines_v1"
REPAIR_OUTPUT = SAFE / "safety_direction_construction_baselines_v2_repair_smoke"
CANONICAL_HELPERS = ROOT / "scripts/official_rrae/extract_iterative_hidden_canonical_dija_v1.py"
V1_RUNNER = ROOT / "scripts/rrae_development/run_generalized_safety_direction_construction_baselines_v1.py"
REPAIR_SBATCH = ROOT / "jobs/rrae_development/generalized_safety_direction_construction_v2_repair_smoke.sbatch"
RESUME_SBATCH = ROOT / "jobs/rrae_development/generalized_safety_direction_construction_baselines_v2_resume.sbatch"
MASK_ID = 126336
MASK_TOKEN = "<|mdm_mask|>"
MANIFEST_SHA = "adf1143bde86933a37d9778f9694cd7d9d7cc425f417706ac61945e0e4bb9acb"
V1_RUNNER_SHA = "f2597e0af2320554459e1e3ae102f14a1e9043731ba1c583b49ce1740168940f"
FINAL_SMOKE_SHA = "469ac827136a4cbf9a56fe4dd7bb1911ef9bb07196cb9caefb3577404f1e41a0"
PREREG_SHA = "f4ce13404dd29325114e38fefbb2e125df52491edf1eb028373eed9d17cd8e93"
DATASET_SHA = "cc3b9228460f12a91286a940a48e4bae64bc2d5187956235dd0c62d5d4e31286"
HELPER_SHA = "67832c5cb193699b2fe83b545b6013b290dc8cddbd8d35504c983bda1cb92680"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
MODEL_WEIGHTS_IDENTITY = "e55f5c03d8bb5a482ef47f3838b6cbff7bb3fcc6e9f98a2a06030b23fddb70cf"
MODEL_IMPLEMENTATION_SHA = "98bac7e53fef0bb7ca01e3716c11a7f710d183e10dbb9783b88db9dbba2e3766"
EXPECTED_EXISTING = (
    ("SAFE_A", 2), ("HARMFUL_DIJA_B", 2), ("HARMFUL_RENELLM_B", 2),
    ("SAFE_A", 3), ("HARMFUL_DIJA_B", 3),
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any, *, forbid_existing: bool = False) -> None:
    if forbid_existing and path.exists():
        raise RuntimeError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest_rows() -> list[dict[str, Any]]:
    if sha_file(MANIFEST) != MANIFEST_SHA or sha_file(V1_RUNNER) != V1_RUNNER_SHA:
        raise RuntimeError("Frozen manifest or V1 runner SHA mismatch")
    audit = json.loads(MANIFEST_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("manifest_sha256") != MANIFEST_SHA:
        raise RuntimeError("Frozen manifest audit mismatch")
    rows = read_jsonl(MANIFEST)
    if len(rows) != 240:
        raise RuntimeError(f"Expected 240 manifest rows, got {len(rows)}")
    for row in rows:
        if row["split"] != "TRAIN" or row["family"] not in {"CLEAN", "DIJA", "RENELLM"}:
            raise RuntimeError("Manifest violates construction scope")
        if sha_text(row["generation_prompt"]) != row["generation_prompt_sha256"]:
            raise RuntimeError("Manifest prompt SHA mismatch")
    return rows


def direct_standard_builder(tokenizer, prompt: str, torch_module):
    """Build output-only standard generation without substring/region search."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    output_region = (len(prompt_ids), len(prompt_ids) + 128)
    full_ids = prompt_ids + [MASK_ID] * 128
    metadata = {
        "prompt_len": len(prompt_ids),
        "total_len": len(full_ids),
        "generation_mode": "standard_assistant_generation",
        "initial_fillable_mask_count": 128,
        "appended_output_mask_count": 128,
        "prompt_sha256": sha_text(prompt),
        "substring_location_performed": False,
    }
    return torch_module.tensor([full_ids], dtype=torch_module.long), {"output": output_region}, metadata


def build_for_row(row, tokenizer, helper, torch_module):
    if row["construction_slot"] == "HARMFUL_DIJA_B":
        adapter = {
            "bucket": "B",
            "id": f"HARMFUL_DIJA_B_{int(row['pair_id']):04d}",
            "prompt": row["generation_prompt"],
            "clean_prompt": row["clean_base_request"],
        }
        x, regions, metadata = helper.build_inputs_and_regions(tokenizer, adapter, MASK_ID, 128)
        if metadata["generation_mode"] != "canonical_dija_infill":
            raise RuntimeError("DIJA builder mode changed")
        if metadata["appended_output_mask_count"] != 0 or regions["output"] is not None:
            raise RuntimeError("DIJA builder appended output masks")
        return x, regions, metadata
    if row["construction_slot"] in {"SAFE_A", "HARMFUL_RENELLM_B"}:
        return direct_standard_builder(tokenizer, row["generation_prompt"], torch_module)
    raise RuntimeError(f"Unknown construction slot: {row['construction_slot']}")


def row_by_identity(rows, slot: str, pair_id: int):
    matches = [row for row in rows if row["construction_slot"] == slot and int(row["pair_id"]) == pair_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one manifest row for {slot}:{pair_id}, got {len(matches)}")
    return matches[0]


def result_path(output: Path, row) -> Path:
    return output / f"{row['construction_slot'].lower()}_pair_{int(row['pair_id']):04d}.json"


def validate_existing(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Existing result unreadable; refusing overwrite: {path}: {error}") from error
    expected = {
        "construction_slot": row["construction_slot"], "family": row["family"],
        "pair_id": int(row["pair_id"]), "condition": row["condition"], "split": "TRAIN",
        "domain": row["domain"], "runtime_mode": row["runtime_mode"],
        "generation_seed": int(row["expected_seed"]),
        "source_prompt_sha256": row["generation_prompt_sha256"],
        "source_dataset_sha256": DATASET_SHA, "construction_manifest_sha256": MANIFEST_SHA,
        "manifest_row_sha256": canonical_sha(row), "canonical_runtime_helper_sha256": HELPER_SHA,
        "model_path": str(MODEL), "model_revision": MODEL_REVISION,
        "model_weights_identity": MODEL_WEIGHTS_IDENTITY,
        "model_implementation_sha256": MODEL_IMPLEMENTATION_SHA,
    }
    for key, wanted in expected.items():
        if result.get(key) != wanted:
            raise RuntimeError(f"Existing result provenance mismatch; refusing overwrite: {path}: {key}")
    if result.get("final_unresolved_mask_count") != 0 or result.get("output_empty") is not False:
        raise RuntimeError(f"Existing result unresolved or empty; refusing overwrite: {path}")
    if result.get("finite_logits_validation") is not True:
        raise RuntimeError(f"Existing result failed finite-logit validation: {path}")
    ids = result.get("generated_token_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(token, int) for token in ids):
        raise RuntimeError(f"Existing result lacks generated token IDs: {path}")
    if canonical_sha(ids) != result.get("generated_token_ids_sha256"):
        raise RuntimeError(f"Existing result token SHA mismatch: {path}")
    decoded = result.get("decoded_output")
    if not isinstance(decoded, str) or not decoded.strip() or sha_text(decoded) != result.get("decoded_output_sha256"):
        raise RuntimeError(f"Existing result output SHA mismatch: {path}")
    model_prov, runtime_prov = result.get("model_provenance"), result.get("runtime_provenance")
    if not isinstance(model_prov, dict) or not isinstance(runtime_prov, dict):
        raise RuntimeError(f"Existing result lacks model/runtime provenance: {path}")
    if (model_prov.get("path") != str(MODEL) or model_prov.get("revision") != MODEL_REVISION
            or model_prov.get("weights_identity") != MODEL_WEIGHTS_IDENTITY
            or model_prov.get("implementation_sha256") != MODEL_IMPLEMENTATION_SHA
            or model_prov.get("loader") != "transformers.AutoModel.from_pretrained"
            or model_prov.get("tokenizer_loader") != "transformers.AutoTokenizer.from_pretrained"):
        raise RuntimeError(f"Existing model provenance invalid: {path}")
    if (runtime_prov.get("canonical_helper_sha256") != HELPER_SHA
            or runtime_prov.get("temperature") != 0.2
            or runtime_prov.get("remask") != "low_confidence"
            or runtime_prov.get("base_seed") != 0):
        raise RuntimeError(f"Existing runtime provenance invalid: {path}")
    expected_appended = 0 if row["construction_slot"] == "HARMFUL_DIJA_B" else 128
    expected_steps = result.get("initial_fillable_mask_count") if expected_appended == 0 else 128
    if result.get("appended_output_mask_count") != expected_appended or result.get("trajectory_step_count") != expected_steps:
        raise RuntimeError(f"Existing generation semantics invalid: {path}")
    return result


def cpu_equivalence(rows, tokenizer, helper, torch_module) -> dict[str, Any]:
    safe2 = row_by_identity(rows, "SAFE_A", 2)
    ren2 = row_by_identity(rows, "HARMFUL_RENELLM_B", 2)
    ren3 = row_by_identity(rows, "HARMFUL_RENELLM_B", 3)
    details: dict[str, Any] = {}
    for label, row in (("SAFE_A_PAIR2", safe2), ("RENELLM_PAIR2", ren2)):
        direct_x, direct_regions, direct_meta = direct_standard_builder(tokenizer, row["generation_prompt"], torch_module)
        canonical_adapter = {
            "bucket": "A", "id": f"{label}_CANONICAL_COMPARISON",
            "prompt": row["generation_prompt"], "clean_prompt": row["clean_base_request"],
        }
        canonical_x, canonical_regions, canonical_meta = helper.build_inputs_and_regions(tokenizer, canonical_adapter, MASK_ID, 128)
        identical = (direct_x.tolist() == canonical_x.tolist()
                     and direct_regions["output"] == canonical_regions["output"]
                     and direct_meta["initial_fillable_mask_count"] == canonical_meta["initial_fillable_mask_count"] == 128
                     and direct_meta["appended_output_mask_count"] == canonical_meta["appended_output_mask_count"] == 128
                     and int(direct_x.shape[1]) == int(canonical_x.shape[1]))
        details[label] = {
            "status": "PASS" if identical else "FAIL",
            "input_token_ids_sha256": canonical_sha(direct_x[0].tolist()),
            "canonical_input_token_ids_sha256": canonical_sha(canonical_x[0].tolist()),
            "initial_sequence_length": int(direct_x.shape[1]),
            "output_region": list(direct_regions["output"]),
            "initial_fillable_mask_count": 128, "appended_output_mask_count": 128,
        }
        if not identical:
            raise RuntimeError(f"{label} direct-standard equivalence failed")
    existing_ren2 = validate_existing(result_path(OUTPUT, ren2), ren2)
    details["RENELLM_PAIR2"]["existing_result_sha256"] = sha_file(result_path(OUTPUT, ren2))
    details["RENELLM_PAIR2"]["existing_initial_sequence_length"] = existing_ren2["initial_sequence_length"]
    if existing_ren2["initial_sequence_length"] != details["RENELLM_PAIR2"]["initial_sequence_length"]:
        raise RuntimeError("ReNeLLM pair-2 existing result does not bind equivalent input length")
    ren3_x, ren3_regions, ren3_meta = direct_standard_builder(tokenizer, ren3["generation_prompt"], torch_module)
    if ren3_meta["substring_location_performed"] or ren3_meta["initial_fillable_mask_count"] != 128:
        raise RuntimeError("ReNeLLM pair-3 direct static build failed")
    details["RENELLM_PAIR3"] = {
        "status": "PASS", "input_token_ids_sha256": canonical_sha(ren3_x[0].tolist()),
        "initial_sequence_length": int(ren3_x.shape[1]), "output_region": list(ren3_regions["output"]),
        "initial_fillable_mask_count": 128, "appended_output_mask_count": 128,
        "substring_location_performed": False,
    }
    return details


def run_static_audit() -> int:
    if PRE_RUN_AUDIT.exists():
        raise RuntimeError(f"Refusing to overwrite existing pre-run audit: {PRE_RUN_AUDIT}")
    rows = manifest_rows()
    failure = json.loads(FAILURE_AUDIT.read_text(encoding="utf-8"))
    if failure.get("classification") != "RUNTIME_STANDARD_GENERATION_BUILDER_BINDING_ERROR":
        raise RuntimeError("V1 failure root cause is not bound")
    import torch
    from transformers import AutoTokenizer
    helper = import_path("safety_direction_v2_static_canonical_helpers", CANONICAL_HELPERS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.encode(MASK_TOKEN, add_special_tokens=False) != [MASK_ID]:
        raise RuntimeError("Native LLaDA mask token identity mismatch")
    equivalence = cpu_equivalence(rows, tokenizer, helper, torch)
    found = valid = 0
    bindings: dict[str, str] = {}
    for slot, pair_id in EXPECTED_EXISTING:
        row = row_by_identity(rows, slot, pair_id)
        path = result_path(OUTPUT, row)
        if path.exists():
            found += 1
            validate_existing(path, row)
            valid += 1
            bindings[f"{slot}:{pair_id}"] = sha_file(path)
    if found != 5 or valid != 5:
        raise RuntimeError(f"Expected five valid existing results, found={found} valid={valid}")
    audit = {
        "schema": "SAFETY_DIRECTION_CONSTRUCTION_BASELINE_V2_PRE_RUN_AUDIT",
        "status": "PASS",
        "assertions": {
            "V1_FAILURE_ROOT_CAUSE_CONFIRMED": True,
            "STANDARD_SAFE_BUILDER": "PASS", "STANDARD_RENELLM_BUILDER": "PASS",
            "DIJA_BUILDER_UNCHANGED": True,
            "SAFE_A_PAIR2_EQUIVALENCE": equivalence["SAFE_A_PAIR2"]["status"],
            "RENELLM_PAIR2_EQUIVALENCE": equivalence["RENELLM_PAIR2"]["status"],
            "RENELLM_PAIR3_STATIC_BUILD": equivalence["RENELLM_PAIR3"]["status"],
            "EXISTING_RESULTS_FOUND": found, "EXISTING_RESULTS_VALID": valid,
            "EXPECTED_FULL_ROWS": len(rows), "MODEL_CALLS": 0, "GPU_GENERATIONS": 0,
            "JUDGE_CALLS": 0, "API_CALLS": 0, "SLURM_SUBMITTED": False,
        },
        "builder_routing": {
            "SAFE_A": "v2_direct_standard_builder", "HARMFUL_RENELLM_B": "v2_direct_standard_builder",
            "HARMFUL_DIJA_B": "unchanged canonical DIJA build_inputs_and_regions",
        },
        "cpu_tokenizer_equivalence": equivalence,
        "existing_result_sha256_bindings": bindings,
        "bindings": {
            "construction_manifest_sha256": MANIFEST_SHA, "v1_runner_sha256": V1_RUNNER_SHA,
            "final_runtime_smoke_sha256": FINAL_SMOKE_SHA, "intervention_prereg_sha256": PREREG_SHA,
            "canonical_dija_helper_sha256": HELPER_SHA, "v1_failure_audit_sha256": sha_file(FAILURE_AUDIT),
            "runner_v2_sha256": sha_file(Path(__file__)),
            "repair_smoke_sbatch_sha256": sha_file(REPAIR_SBATCH),
            "full_resume_sbatch_sha256": sha_file(RESUME_SBATCH),
        },
        "execution_authorized": False,
    }
    atomic_json(PRE_RUN_AUDIT, audit, forbid_existing=True)
    print(json.dumps({"status": "PASS", "model_calls": 0, "gpu_generations": 0}))
    return 0


def immutable_fields(row, model_source: Path) -> dict[str, Any]:
    return {
        "construction_slot": row["construction_slot"], "family": row["family"],
        "pair_id": int(row["pair_id"]), "condition": row["condition"], "split": "TRAIN",
        "domain": row["domain"], "runtime_mode": row["runtime_mode"],
        "generation_seed": int(row["expected_seed"]), "source_prompt_sha256": row["generation_prompt_sha256"],
        "source_dataset_sha256": DATASET_SHA, "construction_manifest_sha256": MANIFEST_SHA,
        "manifest_row_sha256": canonical_sha(row), "canonical_runtime_helper_sha256": HELPER_SHA,
        "model_path": str(MODEL), "model_revision": MODEL_REVISION,
        "model_weights_identity": MODEL_WEIGHTS_IDENTITY,
        "model_implementation_path": str(model_source), "model_implementation_sha256": sha_file(model_source),
    }


def generate(row, model, tokenizer, helper, torch, model_source: Path) -> dict[str, Any]:
    seed = int(row["expected_seed"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x_cpu, regions, metadata = build_for_row(row, tokenizer, helper, torch)
    x = x_cpu.cuda(); initial_masks = x == MASK_ID
    initial_count = int(initial_masks.sum().item())
    steps = initial_count if row["construction_slot"] == "HARMFUL_DIJA_B" else 128
    schedule = helper.get_num_transfer_tokens(initial_masks, steps)
    finite = True; started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        mask_index = x == MASK_ID
        with torch.inference_mode(): output = model(x, attention_mask=torch.ones_like(x))
        logits = output.logits if hasattr(output, "logits") else output[0]
        finite = finite and bool(torch.isfinite(logits).all().item())
        prediction = torch.argmax(helper.add_gumbel_noise(logits, 0.2), dim=-1)
        confidence = helper.confidence_for_predictions(logits, prediction, "low_confidence")
        prediction = torch.where(mask_index, prediction, x)
        confidence = confidence.masked_fill(~mask_index, float("-inf"))
        transfer = torch.zeros_like(mask_index); count = int(schedule[0, step].item())
        if count: transfer[0, torch.topk(confidence[0], k=count).indices] = True
        x[transfer] = prediction[transfer]
    unresolved = int((x == MASK_ID).sum().item())
    if unresolved or not finite:
        raise RuntimeError(f"Invalid generation: unresolved={unresolved} finite={finite}")
    if row["construction_slot"] == "HARMFUL_DIJA_B":
        positions = torch.tensor(metadata["initial_mask_positions"], device=x.device)
        generated_ids = [int(value) for value in x[0].index_select(0, positions).tolist()]
    else:
        lo, hi = regions["output"]
        generated_ids = [int(value) for value in x[0, lo:hi].tolist()]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    fields = immutable_fields(row, model_source)
    return {
        **fields, "initial_sequence_length": int(x_cpu.shape[1]),
        "initial_fillable_mask_count": initial_count, "trajectory_step_count": steps,
        "appended_output_mask_count": int(metadata["appended_output_mask_count"]),
        "final_unresolved_mask_count": unresolved, "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": canonical_sha(generated_ids), "decoded_output": decoded,
        "decoded_output_sha256": sha_text(decoded), "output_empty": not bool(decoded.strip()),
        "finite_logits_validation": finite, "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "model_provenance": {
            "path": str(MODEL), "revision": MODEL_REVISION, "weights_identity": MODEL_WEIGHTS_IDENTITY,
            "implementation_path": str(model_source), "implementation_sha256": sha_file(model_source),
            "loader": "transformers.AutoModel.from_pretrained", "tokenizer_loader": "transformers.AutoTokenizer.from_pretrained",
            "trust_remote_code": True, "local_files_only": True, "dtype": "bfloat16",
        },
        "runtime_provenance": {
            "runner_lineage": "V2", "runner_sha256": sha_file(Path(__file__)),
            "input_builder": "canonical_dija_helper" if row["construction_slot"] == "HARMFUL_DIJA_B" else "v2_direct_standard_builder",
            "canonical_helper_path": str(CANONICAL_HELPERS), "canonical_helper_sha256": HELPER_SHA,
            "temperature": 0.2, "remask": "low_confidence", "base_seed": 0,
            "seed_semantics": "base_seed*10000 + pair_id*4 + bucket_id; A=0,B=1",
            "torch_version": torch.__version__,
        },
    }


def write_progress(output: Path, expected: int, completed: list[str], failed: list[dict[str, str]]) -> None:
    atomic_json(output / "progress_state.json", {
        "expected": expected, "completed": len(completed), "remaining": expected - len(completed),
        "failed": failed, "completed_case_files": completed, "runner_lineage": "V2",
    })


def execute(repair_smoke: bool) -> int:
    if not PRE_RUN_AUDIT.exists() or json.loads(PRE_RUN_AUDIT.read_text()).get("status") != "PASS":
        raise RuntimeError("Passing V2 static pre-run audit is required")
    all_rows = manifest_rows()
    rows = all_rows
    if repair_smoke:
        rows = [row_by_identity(rows, "HARMFUL_RENELLM_B", 3)]
        output = REPAIR_OUTPUT
    else:
        output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []; failed: list[dict[str, str]] = []
    import torch
    from transformers import AutoModel, AutoTokenizer
    helper = import_path("safety_direction_v2_runtime_canonical_helpers", CANONICAL_HELPERS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.encode(MASK_TOKEN, add_special_tokens=False) != [MASK_ID]:
        raise RuntimeError("Native LLaDA mask token identity mismatch")
    # Mandatory CPU-only gate on every execution path. This occurs before
    # AutoModel loading and before any explicit CUDA operation.
    cpu_equivalence(all_rows, tokenizer, helper, torch)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True, local_files_only=True).eval().cuda()
    model_source = Path(inspect.getfile(model.__class__)).resolve()
    if model.__class__.__name__ != "LLaDAModelLM" or "MaskForge" in str(model_source) or sha_file(model_source) != MODEL_IMPLEMENTATION_SHA:
        raise RuntimeError(f"Canonical model provenance mismatch: {model_source}")
    ren2_equivalence = json.loads(PRE_RUN_AUDIT.read_text())["assertions"]["RENELLM_PAIR2_EQUIVALENCE"] == "PASS"
    for row in rows:
        path = result_path(output, row)
        try:
            if path.exists():
                if row["construction_slot"] == "HARMFUL_RENELLM_B" and int(row["pair_id"]) == 2 and not ren2_equivalence:
                    raise RuntimeError("ReNeLLM pair-2 reuse blocked by failed equivalence")
                validate_existing(path, row)
            else:
                atomic_json(path, generate(row, model, tokenizer, helper, torch, model_source), forbid_existing=True)
                validate_existing(path, row)
            completed.append(path.name); write_progress(output, len(rows), completed, failed)
        except Exception as error:
            failed.append({"case": f"{row['construction_slot']}:{row['pair_id']}", "error": str(error)})
            write_progress(output, len(rows), completed, failed)
            raise
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--repair-smoke", action="store_true")
    modes.add_argument("--static-audit", action="store_true")
    args = parser.parse_args()
    if args.static_audit:
        return run_static_audit()
    return execute(args.repair_smoke)


if __name__ == "__main__":
    raise SystemExit(main())
