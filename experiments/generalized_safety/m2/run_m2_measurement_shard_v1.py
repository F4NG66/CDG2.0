"""
M2 beta-calibration measurement pass - GPU EXECUTOR (one shard).

Runs the frozen 306 UNSTEERED DEV trajectories and records, for every
current_mask token position at L16 of every denoising step:

    ||h||_2  over the 4096 hidden dim, and  <h,v>  for V_DIJA, V_RENELLM, V_ALL.

The hook is MEASUREMENT ONLY: it returns None and never modifies a hidden state.
No steering is applied. No decoded text is recorded. No outcome label is read.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json, math, hashlib, importlib.util, inspect
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV  = ROOT / "analysis_output/rrae_development"
MODEL = Path(__import__("os").environ["CDG_LLADA_MODEL"])
HERE = DEV / "generalized_safety_v2_m2_beta_calibration_measurement_v1"
CONTRACT = DEV / "generalized_safety_v2_m2_contract_v1"

SHARD_INDEX = int(os.environ["SLURM_ARRAY_TASK_ID"])
assert 0 <= SHARD_INDEX < 4
OUT = HERE / f"shard_{SHARD_INDEX:02d}"

ITEMS = HERE / "MEASUREMENT_ITEMS_306.jsonl"
PREFLIGHT = HERE / "PREFLIGHT_FREEZE.json"
DIRECTIONS = DEV / "generalized_safety_v2_directions_v1/GENERALIZED_SAFETY_V2_DIRECTIONS.pt"
M1_PATH = ROOT / "scripts/rrae_development/run_generalized_safety_m1_eval_v1.py"
COMPAT_HELPER_PATH = (DEV / "generalized_safety_v2_llada_baseline_dija_localization_compat_v1"
                          / "frozen_generation_helpers_compat_v1.py")

MASK_ID = 126336
LAYER = 16
ARM_ORDER = ["V_DIJA", "V_RENELLM", "V_ALL"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Frozen bindings
# ============================================================
pre = json.load(open(PREFLIGHT))
assert pre["status"] == "PASS_FROZEN"
assert sha256_file(ITEMS) == pre["measurement_items_sha256"]
assert sha256_file(DIRECTIONS) == pre["bindings"]["directions_sha256"]
assert sha256_file(M1_PATH) == pre["bindings"]["historical_m1_runner_sha256"]
assert sha256_file(CONTRACT / "M2_BETA_SOLVING_PROCEDURE_V1.json") == \
       pre["bindings"]["solving_procedure_sha256"]
assert pre["steering_applied"] is False
print("RUNTIME_FROZEN_BINDINGS=PASS", flush=True)

m1 = load_module("historical_m1", M1_PATH)
hist = load_module("historical_hook", Path(m1.HIST_PATH))
baseline_helper = load_module("historical_baseline_helper", Path(m1.BASELINE_PATH))
helper = load_module("historical_dija_helper", Path(m1.HELPER_PATH))
compat_helper = load_module("frozen_dija_compat_helper", COMPAT_HELPER_PATH)
m1.hist = hist
if hasattr(m1, "sha_text"):
    helper.sha256_text = m1.sha_text
    compat_helper.sha256_text = m1.sha_text

rows = [json.loads(l) for l in open(ITEMS) if l.strip()]
assert len(rows) == 306
rows = [r for r in rows if int(r["shard_index"]) == SHARD_INDEX]
assert len(rows) == pre["rows_per_shard"][str(SHARD_INDEX)]

if OUT.exists():
    raise RuntimeError(f"REFUSING_TO_OVERWRITE_SHARD_OUTPUT={OUT}")
OUT.mkdir(parents=True, exist_ok=False)

# ============================================================
# Directions
# ============================================================
directions = torch.load(DIRECTIONS, map_location="cpu", weights_only=True)
assert directions["schema"] == "GENERALIZED_SAFETY_V2_DIRECTIONS_V1"
assert int(directions["layer_zero_based"]) == LAYER
assert directions["orientation"] == "safe_minus_harmful"

vecs = {"V_DIJA": directions["v_DIJA"], "V_RENELLM": directions["v_RENELLM"],
        "V_ALL": directions["v_ALL"]}
for k, v in vecs.items():
    v = v.detach().float().view(-1)
    assert v.numel() == 4096 and torch.isfinite(v).all()
    assert math.isclose(float(v.norm()), 1.0, abs_tol=1e-5), k
    vecs[k] = v
print("DIRECTION_BINDING=PASS", flush=True)

# ============================================================
# Model
# ============================================================
assert MODEL.is_dir() and torch.cuda.is_available()
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True,
                                          local_files_only=True)
assert helper.tokenize_without_specials(tokenizer, "<|mdm_mask|>") == [MASK_ID]
model = AutoModel.from_pretrained(str(MODEL), trust_remote_code=True,
                                  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                  local_files_only=True).eval().to(device)
assert model.dtype == torch.bfloat16
print("MODEL_BINDING=PASS", flush=True)

blocks, blocks_attr, _ = hist.resolve_transformer_blocks(model)
assert len(blocks) == 32

V = torch.stack([vecs[a] for a in ARM_ORDER], 0).to(device, torch.float32)  # [3, 4096]


class MeasurementProbe:
    """Measurement-only. Captures current_mask from the model input, records
    ||h||_2 and <h,v> at L16, and NEVER modifies a hidden state."""

    def __init__(self, model, blocks, layer):
        self.layer = layer
        self.current_mask = None
        self.step = 0
        self.fired = 0
        self.buf = []
        self.active = False
        self._pre = model.register_forward_pre_hook(self._make_pre())
        self._h = blocks[layer].register_forward_hook(self._make_hook())

    def reset(self):
        self.current_mask, self.step, self.fired, self.buf = None, 0, 0, []

    def _make_pre(self):
        def pre_hook(_module, args):
            if not self.active:
                return None
            x = args[0]
            self.current_mask = (x == MASK_ID)
            self.step += 1
            return None
        return pre_hook

    def _make_hook(self):
        def hook(_module, _inputs, output):
            if not self.active:
                return None
            hidden = hist.output_to_hidden(output)
            assert hidden.ndim == 3 and hidden.shape[-1] == 4096
            target = self.current_mask
            assert target is not None
            assert list(target.shape) == list(hidden.shape[:2])
            n = int(target.sum().item())
            assert n > 0, "empty current_mask at measurement step"

            sel = hidden[target].to(torch.float32)              # [n, 4096]
            norms = torch.linalg.vector_norm(sel, dim=-1)       # [n]
            proj = sel @ V.t()                                  # [n, 3]
            assert torch.isfinite(norms).all() and torch.isfinite(proj).all()

            rec = torch.cat([norms.unsqueeze(-1), proj], dim=-1)  # [n, 4]
            self.buf.append(rec.detach().cpu().numpy().astype(np.float32).copy())
            self.fired += 1
            return None                                          # <-- no modification
        return hook

    def close(self):
        self._pre.remove()
        self._h.remove()


probe = MeasurementProbe(model, blocks, LAYER)

# All historical banks stay inert (layer=None), so they never fire.
banks = {k: hist.SafetyLayerLocationHookBank(blocks, v, (LAYER,)) for k, v in vecs.items()}
print("PROBE_BINDING=PASS", flush=True)

# ============================================================
# Execute
# ============================================================
index, chunks, offset = [], [], 0
try:
    for item in rows:
        route = item["input_helper_route"]
        helper_for_item = compat_helper if route == "DIJA_COMPAT" else helper
        assert sha256_file(COMPAT_HELPER_PATH if route == "DIJA_COMPAT"
                           else Path(m1.HELPER_PATH)) == item["input_helper_sha256"]

        probe.reset()
        probe.active = True
        result = m1.generate_one(item=item, model=model, tokenizer=tokenizer,
                                 helper=helper_for_item, baseline=baseline_helper,
                                 banks=banks, device=device)
        probe.active = False

        steps = int(result["trajectory_steps"])
        # unsteered invariants
        assert result["steering_condition"] == "BASELINE"
        assert result["vector_key"] is None
        assert int(result["application_count"]) == 0
        assert result["step_audits"] == []
        # input parity with the frozen manifest
        assert result["initial_ids_sha256"] == item["expected_initial_ids_sha256"]
        if item["attack_family"] == "DIJA":
            assert steps == int(item["expected_trajectory_steps"])
        else:
            assert steps == 128
        # probe fired exactly once per denoising step
        assert probe.fired == steps, f"probe fired {probe.fired} != {steps}"
        assert probe.step == steps

        obs = np.concatenate(probe.buf, axis=0)
        assert obs.ndim == 2 and obs.shape[1] == 4
        assert np.isfinite(obs).all()

        index.append({
            "source_case_id": item["source_case_id"],
            "pair_id": item["pair_id"],
            "attack_family": item["attack_family"],
            "condition": item["condition"],
            "domain": item["domain"],
            "m2_split": item["m2_split"],
            "trajectory_steps": steps,
            "observations": int(obs.shape[0]),
            "offset": offset,
            "arm_order": ARM_ORDER,
            "columns": ["h_norm", "proj_V_DIJA", "proj_V_RENELLM", "proj_V_ALL"],
        })
        chunks.append(obs)
        offset += int(obs.shape[0])

        print(f"MEAS_CASE_PASS case={item['source_case_id']} family={item['attack_family']} "
              f"cond={item['condition']} split={item['m2_split']} steps={steps} "
              f"obs={obs.shape[0]}", flush=True)
finally:
    probe.close()
    for b in banks.values():
        b.close()

assert len(index) == len(rows)
allobs = np.concatenate(chunks, axis=0)
assert allobs.shape[0] == offset

np.save(OUT / "OBSERVATIONS.npy", allobs)
with open(OUT / "OBSERVATION_INDEX.jsonl", "w") as f:
    for r in index:
        f.write(json.dumps(r, sort_keys=True) + "\n")

summary = {
    "schema": "GENERALIZED_SAFETY_V2_M2_MEASUREMENT_SHARD_SUMMARY_V1",
    "status": "PASS",
    "shard_index": SHARD_INDEX,
    "source_cases": len(index),
    "total_observations": int(offset),
    "model_forward_calls": int(sum(r["trajectory_steps"] for r in index)),
    "layer_zero_based": LAYER,
    "scope": "current_mask",
    "arm_order": ARM_ORDER,
    "columns": ["h_norm", "proj_V_DIJA", "proj_V_RENELLM", "proj_V_ALL"],
    "steering_applied": False,
    "hidden_states_modified": False,
    "beta_used": 0.0,
    "decoded_text_recorded": False,
    "outcome_labels_read": False,
    "beta_solved": False,
    "dtype": "torch.bfloat16",
    "measurement_dtype": "float32",
    "observations_sha256": sha256_file(OUT / "OBSERVATIONS.npy"),
    "index_sha256": sha256_file(OUT / "OBSERVATION_INDEX.jsonl"),
    "items_sha256": pre["measurement_items_sha256"],
    "modeling_llada_sha256": sha256_file(Path(inspect.getfile(type(model)))),
}
with open(OUT / "SHARD_SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True); f.write("\n")

print(f"SHARD_{SHARD_INDEX:02d}_COMPLETE cases={len(index)} observations={offset}", flush=True)
