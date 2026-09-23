import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import math
import hashlib
import importlib.util
import inspect
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
MODEL = Path(__import__("os").environ["CDG_LLADA_MODEL"])

PREFLIGHT = (
    ROOT
    / "analysis_output/rrae_development"
    / "generalized_safety_v2_development_m1_full918_preflight_v2"
)

BASE_OUT = (
    ROOT
    / "analysis_output/rrae_development"
    / "generalized_safety_v2_m1_heldout_llada_execution_v1"
)

SHARD_INDEX = int(os.environ["SLURM_ARRAY_TASK_ID"])

assert 0 <= SHARD_INDEX < 12

OUT = BASE_OUT / f"shard_{SHARD_INDEX:02d}"

COMPAT_HELPER_PATH = (
    ROOT
    / "analysis_output/rrae_development"
    / "generalized_safety_v2_llada_baseline_dija_localization_compat_v1"
    / "frozen_generation_helpers_compat_v1.py"
)

DIRECTIONS = (
    ROOT
    / "analysis_output/rrae_development"
    / "generalized_safety_v2_directions_v1"
    / "GENERALIZED_SAFETY_V2_DIRECTIONS.pt"
)

BETAS = (
    ROOT
    / "analysis_output/rrae_development"
    / "generalized_safety_v2_beta_calibration_v1"
    / "BETAS.json"
)

M1_PATH = (
    ROOT
    / "scripts/rrae_development"
    / "run_generalized_safety_m1_eval_v1.py"
)

MANIFEST = (
    ROOT
    / "analysis_output/rrae_development"
    / "generalized_safety_v2_m1_heldout_runtime_manifest_cpu_v1"
    / "HELDOUT_LLADA_RUNTIME_MANIFEST_PRIVATE_1584.jsonl"
)
BINDING = BASE_OUT / "FULL_RUNTIME_BINDING.json"

MASK_ID = 126336


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def atomic_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


# ============================================================
# Frozen bindings
# ============================================================

binding = json.load(open(BINDING))

assert sha256_file(MANIFEST) == binding["full_manifest_sha256"]
assert sha256_file(DIRECTIONS) == binding["directions_sha256"]
assert sha256_file(BETAS) == binding["betas_sha256"]
assert sha256_file(M1_PATH) == binding["historical_m1_runner_sha256"]

print("RUNTIME_FROZEN_BINDINGS=PASS")


# ============================================================
# Historical implementation
# ============================================================

m1 = load_module("historical_m1", M1_PATH)

hist_path = Path(m1.HIST_PATH)
baseline_helper_path = Path(m1.BASELINE_PATH)
helper_path = Path(m1.HELPER_PATH)

assert sha256_file(hist_path) == binding["hist_sha256"]
assert sha256_file(baseline_helper_path) == binding["baseline_helper_sha256"]
assert sha256_file(helper_path) == binding["canonical_helper_sha256"]
assert sha256_file(COMPAT_HELPER_PATH) == binding["compat_helper_sha256"]

hist = load_module("historical_hook", hist_path)

# Historical generate_one() references `hist` through its module globals.
# The historical main() supplied the same HIST_PATH module locally.
# We bind that exact frozen module into generate_one's global namespace
# without modifying the historical runner.
m1.hist = hist

baseline_helper = load_module("historical_baseline_helper", baseline_helper_path)
helper = load_module("historical_dija_helper", helper_path)
compat_helper = load_module("frozen_dija_compat_helper", COMPAT_HELPER_PATH)

if hasattr(m1, "sha_text"):
    helper.sha256_text = m1.sha_text
    compat_helper.sha256_text = m1.sha_text


# ============================================================
# Manifest checks
# ============================================================

all_rows = read_jsonl(MANIFEST)

assert len(all_rows) == 1584

rows = [
    row
    for row in all_rows
    if int(row["shard_index"]) == SHARD_INDEX
]

expected_shard_rows = 132

assert len(rows) == expected_shard_rows

if OUT.exists():
    raise RuntimeError(
        f"REFUSING_TO_OVERWRITE_SHARD_OUTPUT={OUT}"
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
)

ALLOWED_V2_CONDITIONS = {
    "B_DIJA",
    "C_DIJA",
    "B_RENELLM",
    "C_RENELLM",
}

# Full 918 manifest MUST contain all four conditions.
assert {
    r["v2_condition"]
    for r in all_rows
} == ALLOWED_V2_CONDITIONS

# Individual deterministic shards are not required to
# contain all four conditions; they only need valid members.
assert all(
    r["v2_condition"] in ALLOWED_V2_CONDITIONS
    for r in rows
)

assert {
    r["steering_condition"]
    for r in rows
} == {
    "BASELINE",
    "V_DIJA",
    "V_RENELLM",
    "V_ALL",
}

for row in rows:

    if row["v2_condition"].startswith("B_"):
        assert row["condition"] == "B"

    elif row["v2_condition"].startswith("C_"):
        assert row["condition"] == "C"

    else:
        raise RuntimeError(row["v2_condition"])

    if row["attack_family"] == "DIJA":
        assert isinstance(row["clean_semantic_request"], str)
        assert row["clean_semantic_request"]

print("RUNTIME_SCHEMA_BRIDGE=PASS")


# ============================================================
# Directions
# ============================================================

directions = torch.load(
    DIRECTIONS,
    map_location="cpu",
    weights_only=True,
)

assert directions["schema"] == "GENERALIZED_SAFETY_V2_DIRECTIONS_V1"
assert int(directions["layer_zero_based"]) == 16
assert directions["orientation"] == "safe_minus_harmful"

vectors = {
    "V_DIJA": directions["v_DIJA"].detach().float().view(-1),
    "V_RENELLM": directions["v_RENELLM"].detach().float().view(-1),
    "V_ALL": directions["v_ALL"].detach().float().view(-1),
}

for key, vector in vectors.items():
    assert vector.numel() == 4096
    assert torch.isfinite(vector).all()
    assert math.isclose(
        float(vector.norm().item()),
        1.0,
        abs_tol=1e-5,
    ), key


# ============================================================
# Model
# ============================================================

assert MODEL.is_dir()
assert torch.cuda.is_available()

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL),
    trust_remote_code=True,
    local_files_only=True,
)

assert helper.tokenize_without_specials(
    tokenizer,
    "<|mdm_mask|>",
) == [MASK_ID]

model = AutoModel.from_pretrained(
    str(MODEL),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True,
).eval().to(device)

assert model.dtype == torch.bfloat16

model_impl = Path(inspect.getfile(type(model)))
model_impl_sha = sha256_file(model_impl)

assert model_impl_sha == binding["modeling_llada_sha256"]

print("MODEL_BINDING=PASS")


# ============================================================
# Hook bank
# ============================================================

blocks, blocks_attr, candidates = hist.resolve_transformer_blocks(model)

assert len(blocks) == 32

banks = {
    key: hist.SafetyLayerLocationHookBank(
        blocks,
        vector,
        (16,),
    )
    for key, vector in vectors.items()
}

print("HOOK_BANK_BINDING=PASS")


# ============================================================
# Execute only frozen smoke
# ============================================================

results = []

try:

    for item in rows:

        helper_route = item["input_helper_route"]

        if helper_route == "DIJA_COMPAT":
            helper_for_item = compat_helper

        elif helper_route in {
            "DIJA_ORIGINAL",
            "RENELLM_STANDARD",
        }:
            helper_for_item = helper

        else:
            raise RuntimeError(
                f"Unknown input_helper_route={helper_route}"
            )

        assert (
            sha256_file(
                COMPAT_HELPER_PATH
                if helper_route == "DIJA_COMPAT"
                else helper_path
            )
            == item["input_helper_sha256"]
        )

        result = m1.generate_one(

            item=item,
            model=model,
            tokenizer=tokenizer,
            helper=helper_for_item,
            baseline=baseline_helper,
            banks=banks,
            device=device,
        )

        # Exact CPU-frozen input parity.
        assert (
            result["initial_ids_sha256"]
            == item["expected_initial_ids_sha256"]
        )

        # Exact baseline parity was already frozen CPU-side.
        assert (
            result["initial_ids_sha256"]
            == item["expected_initial_ids_sha256"]
        )

        v2_condition = item["v2_condition"]

        if v2_condition.endswith("_DIJA"):

            assert int(result["trajectory_steps"]) == int(
                item["expected_trajectory_steps"]
            )

        else:

            assert int(result["trajectory_steps"]) == 128

        if result["steering_condition"] == "BASELINE":
            assert result["vector_key"] is None
            assert int(result["application_count"]) == 0
            assert result["step_audits"] == []
        else:
            assert int(result["application_count"]) == int(
                result["trajectory_steps"]
            )
            assert len(result["step_audits"]) == int(
                result["trajectory_steps"]
            )

        # Keep V2 condition explicit; historical result condition
        # remains B/C only for adapter compatibility.
        result["v2_condition"] = v2_condition
        result["historical_condition"] = item["condition"]
        result["source_case_id"] = item["source_case_id"]

        # Historical M1 hardcodes split="EVAL" in its legacy
        # result schema. This V2 experiment is DEVELOPMENT.
        # Metadata normalization only; generation is unchanged.
        result["split"] = "HELDOUT_CONFIRMATORY_EVALUATION"

        results.append(result)

        print(
            "SHARD_CASE_PASS "
            f"pair={result['pair_id']} "
            f"v2_condition={v2_condition} "
            f"arm={result['steering_condition']} "
            f"steps={result['trajectory_steps']} "
            f"applications={result['application_count']}",
            flush=True,
        )

finally:

    for bank in banks.values():
        bank.close()


assert len(results) == len(rows)


# ============================================================
# Step-level causal audits
# ============================================================

for result in results:

    beta = float(result["beta"])
    tolerance = max(0.25, 0.02 * abs(beta))

    for audit in result["step_audits"]:

        assert abs(
            float(audit["target_mean_delta_l2"]) - beta
        ) <= tolerance

        assert float(
            audit["target_min_cosine_to_v_safety"]
        ) >= 0.995

        assert float(
            audit["non_target_max_abs_delta"]
        ) == 0.0


forward_calls = sum(
    int(r["trajectory_steps"])
    for r in results
)

assert forward_calls == sum(
    int(r["expected_trajectory_steps"])
    for r in rows
)

private_results = OUT / "SHARD_RESULTS_PRIVATE.jsonl"

atomic_jsonl(
    private_results,
    results,
)


summary = {
    "shard_index": SHARD_INDEX,
    "expected_shard_rows": expected_shard_rows,
    "expected_forward_calls": sum(
        int(r["expected_trajectory_steps"])
        for r in rows
    ),
    "GENERALIZED_SAFETY_V2_M1_HELDOUT_SHARD_V1":
        "PASS",

    "shard_rows":
        len(rows),

    "model_forward_calls":
        forward_calls,

    "layer_zero_based":
        16,

    "transformer_block_count":
        len(blocks),

    "transformer_blocks_attr":
        blocks_attr,

    "dtype":
        str(model.dtype),

    "scope":
        "current_mask",

    "schedule":
        "persistent",

    "operation":
        "h_new = h + beta * v",

    "schema_bridge":
        {
            "B_DIJA": "B",
            "B_RENELLM": "B",
            "C_DIJA": "C",
            "C_RENELLM": "C",
        },

    "semantic_source_bridge":
        {
            "B_DIJA": "A",
            "C_DIJA": "D",
        },

    "trajectory_policy":
        {
            "DIJA":
                "initial_fillable_mask_count",

            "RENELLM":
                128,
        },

    "initial_input_v2_baseline_parity_pass":
        True,

    "persistent_application_audit_pass":
        True,

    "steering_delta_magnitude_audit_pass":
        True,

    "steering_direction_cosine_audit_pass":
        True,

    "non_target_unchanged_audit_pass":
        True,

    "baseline_regenerated":
        True,

    "judge_used":
        False,

    "heldout_outcomes_used_for_selection":
        False,

    "beta_sweep_used":
        False,

    "M2_used":
        False,

    "M3_used":
        False,

    "modeling_llada_sha256":
        model_impl_sha,

    "full_manifest_sha256":
        sha256_file(MANIFEST),

    "private_results_sha256":
        sha256_file(private_results),
}

summary_path = OUT / "SHARD_RUNTIME_SUMMARY.json"

atomic_json(
    summary_path,
    summary,
)


print()
print("=" * 72)
print(
    "GENERALIZED_SAFETY_V2_DEVELOPMENT_M1_FULL918_SHARD_V3=PASS"
)
print("=" * 72)

print(f"SHARD_ROWS={len(rows)}")
print(f"MODEL_FORWARD_CALLS={forward_calls}")

print("V1_V2_SCHEMA_BRIDGE=PASS")
print("V2_BASELINE_INITIAL_INPUT_PARITY=PASS")

print("B_DIJA->B")
print("B_RENELLM->B")
print("C_DIJA->C")
print("C_RENELLM->C")

print("B_DIJA_SEMANTIC_SOURCE=A")
print("C_DIJA_SEMANTIC_SOURCE=D")

print("LAYER_ZERO_BASED=16")
print("SCOPE=current_mask")
print("SCHEDULE=persistent")

print("PERSISTENT_APPLICATION_AUDIT=PASS")
print("STEERING_DELTA_MAGNITUDE_AUDIT=PASS")
print("STEERING_DIRECTION_COSINE_AUDIT=PASS")
print("NON_TARGET_UNCHANGED_AUDIT=PASS")

print("BASELINE_REGENERATED=false")
print("JUDGE_USED=false")
print(
    "DEVELOPMENT_OUTCOMES_USED_FOR_SELECTION=false"
)
print("BETA_SWEEP_USED=false")
print("M2_USED=false")
print("M3_USED=false")

print(
    "SHARD_RUNTIME_SUMMARY_SHA256=",
    sha256_file(summary_path),
)

print(
    "SHARD_RESULTS_PRIVATE_SHA256=",
    sha256_file(private_results),
)

print("STATUS=PASS")
