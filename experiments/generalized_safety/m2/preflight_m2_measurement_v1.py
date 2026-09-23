"""
M2 beta-calibration measurement pass - STATIC PREFLIGHT (CPU only).

Builds and freezes the 306 unsteered measurement items and verifies every binding
the GPU executor will re-assert. No model is loaded. No generation is performed.
No M2 confirmatory row is touched.
"""

import json, hashlib, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV  = ROOT / "analysis_output/rrae_development"
HERE = Path(__file__).parent
CONTRACT = DEV / "generalized_safety_v2_m2_contract_v1"

MANIFEST = DEV / "generalized_safety_v2_development_m1_full918_preflight_v2/DEVELOPMENT_M1_FULL_918_MANIFEST.jsonl"
DIRECTIONS = DEV / "generalized_safety_v2_directions_v1/GENERALIZED_SAFETY_V2_DIRECTIONS.pt"
SPLIT = CONTRACT / "M2_BETA_DEV_SPLIT_V1.jsonl"
SOLVING = CONTRACT / "M2_BETA_SOLVING_PROCEDURE_V1.json"
M1_RUNNER = ROOT / "scripts/rrae_development/run_generalized_safety_m1_eval_v1.py"

EXPECT = {
    "directions_sha256": "e43916ff3aaab143201daeeae78c66fa50830b0b4a7ff4ff6447fb5b3c05d487",
    "split_sha256": "5bac6508c7577f539b8687971df0f2fa32faba45adcfb6d6cc78d6b045768883",
    "solving_procedure_sha256": "b3a975ea591533dff0cb9e6805b94a443956d3923cc7ea5f85405aa9b688a502",
    "historical_m1_runner_sha256": "3ff21e59c7d045c42187d9880410280e92d2efcd6a585ecad1db984b63da618f",
}

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def fail(msg):
    print("PREFLIGHT_FAIL:", msg)
    sys.exit(1)

checks = {}

# ---- 1. frozen artifact bindings ----
for key, path in (("directions_sha256", DIRECTIONS),
                  ("split_sha256", SPLIT),
                  ("solving_procedure_sha256", SOLVING),
                  ("historical_m1_runner_sha256", M1_RUNNER)):
    got = sha256_file(path)
    if got != EXPECT[key]:
        fail(f"{key} mismatch: expected {EXPECT[key]} got {got} ({path})")
    checks[key] = got
print("FROZEN_BINDINGS=PASS")

# ---- 2. solving procedure must be frozen before measurements ----
sp = json.load(open(SOLVING))
if not sp.get("frozen_before_any_calibration_number_observed"):
    fail("solving procedure not marked frozen before calibration")
if sp["build_dev_combination"]["decision"] != "DEV_ONLY_SPLIT_INTO_SOLVE_AND_CHECK":
    fail("solving procedure combination rule changed")
if sp["displacement_aggregation"]["rule"] != "PER_TOKEN_THEN_AGGREGATE":
    fail("aggregation rule changed")
checks["solving_procedure_frozen"] = True
print("SOLVING_PROCEDURE=PASS")

# ---- 3. DEV manifest -> 306 source cases ----
rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]
if len(rows) != 918:
    fail(f"expected 918 manifest rows, got {len(rows)}")

by_case = {}
for r in rows:
    k = (r["pair_id"], r["attack_family"], r["condition"])
    by_case.setdefault(k, []).append(r)
if len(by_case) != 306:
    fail(f"expected 306 source cases, got {len(by_case)}")

for k, v in by_case.items():
    if len(v) != 3:
        fail(f"source case {k} has {len(v)} arms, expected 3")
    if len({x["generation_seed"] for x in v}) != 1:
        fail(f"seed drift across arms for {k}")
    if len({x["expected_initial_ids_sha256"] for x in v}) != 1:
        fail(f"initial-ids drift across arms for {k}")
    if len({x["prompt_sha256"] for x in v}) != 1:
        fail(f"prompt drift across arms for {k}")
checks["source_cases"] = 306
checks["arm_parity_verified"] = True
print("MANIFEST_PARITY=PASS")

# ---- 4. split covers exactly those cases ----
split_rows = [json.loads(l) for l in open(SPLIT) if l.strip()]
split_map = {r["source_case_id"]: r["split"] for r in split_rows}
def case_id(k):
    return f"{k[0]:04d}|{k[1]}|{k[2]}"
if set(split_map) != {case_id(k) for k in by_case}:
    fail("split manifest does not cover exactly the 306 DEV source cases")
sc = Counter(split_map.values())
if sc["DEV_SOLVE"] + sc["DEV_CHECK"] != 306:
    fail("split counts wrong")
checks["split_counts"] = dict(sc)
print("SPLIT_COVERAGE=PASS")

# ---- 5. generation settings match the frozen contract ----
contract = json.load(open(CONTRACT / "M2_CONTRACT_V1.json"))
gs = contract["item_08_generation_settings"]
if {r["gen_length"] for r in rows} != {gs["generation_length"]}:
    fail("gen_length drift")
if {r["temperature"] for r in rows} != {gs["temperature"]}:
    fail("temperature drift")
if {r["remasking"] for r in rows} != {gs["remasking"]}:
    fail("remasking drift")
if {r["layer_zero_based"] for r in rows} != {contract["item_02_layer"]["value"]}:
    fail("layer drift")
if {r["token_scope"] for r in rows} != {contract["item_03_token_scope"]["value"]}:
    fail("token scope drift")
checks["generation_settings_match_contract"] = True
print("GENERATION_SETTINGS=PASS")

# ---- 6. build the unsteered measurement items ----
items = []
for k in sorted(by_case):
    base = sorted(by_case[k], key=lambda r: r["steering_condition"])[0]
    it = dict(base)
    # unsteered: no direction, no beta, no active steps
    it["vector_key"] = None
    it["beta"] = 0.0
    it["steering_condition"] = "BASELINE"
    it["direction_key"] = None
    it["method"] = "M2_BETA_CALIBRATION_MEASUREMENT_UNSTEERED"
    it["role"] = "M2_BETA_CALIBRATION_MEASUREMENT"
    it["operation"] = "measurement_only_no_hidden_state_modification"
    it["source_case_id"] = case_id(k)
    it["m2_split"] = split_map[case_id(k)]
    it["measurement_index"] = len(items)
    it["shard_index"] = len(items) % 4
    for drop in ("beta_artifact", "beta_artifact_sha256", "direction_artifact",
                 "direction_artifact_sha256", "direction_tensor_sha256"):
        it.pop(drop, None)
    items.append(it)

if len(items) != 306:
    fail("measurement item count wrong")
if any(i["vector_key"] is not None or i["beta"] != 0.0 for i in items):
    fail("measurement item is not unsteered")

ip = HERE / "MEASUREMENT_ITEMS_306.jsonl"
with open(ip, "w") as f:
    for it in items:
        f.write(json.dumps(it, sort_keys=True) + "\n")
items_sha = sha256_file(ip)
print("MEASUREMENT_ITEMS=PASS")

# ---- 7. isolation guards ----
blob = json.dumps(items)
for forbidden in ("FSBV1_", "M2_CONFIRMATORY", "HELDOUT_CONFIRMATORY"):
    if forbidden in blob:
        fail(f"measurement items reference forbidden population marker: {forbidden}")
checks["m2_confirmatory_population_referenced"] = False
checks["m1_heldout_referenced"] = False
checks["outcome_labels_referenced"] = False
print("ISOLATION_GUARDS=PASS")

# ---- 8. freeze ----
freeze = {
    "schema": "GENERALIZED_SAFETY_V2_M2_BETA_CALIBRATION_MEASUREMENT_PREFLIGHT_FREEZE_V1",
    "status": "PASS_FROZEN",
    "contract_item": 6,
    "bindings": checks,
    "measurement_items": 306,
    "measurement_items_sha256": items_sha,
    "shards": 4,
    "rows_per_shard": dict(Counter(i["shard_index"] for i in items)),
    "family_split": dict(Counter(i["attack_family"] for i in items)),
    "condition_split": dict(Counter(i["condition"] for i in items)),
    "solve_check_split": dict(Counter(i["m2_split"] for i in items)),
    "steering_applied": False,
    "hidden_states_modified": False,
    "outcome_labels_read": False,
    "decoded_text_recorded": False,
    "generation_started": False,
    "gpu_used": False,
    "next_state": "MEASUREMENT_PREFLIGHT_FROZEN_AWAITING_EXECUTION_AUTHORIZATION",
}
fp = HERE / "PREFLIGHT_FREEZE.json"
with open(fp, "w") as f:
    json.dump(freeze, f, indent=2, sort_keys=True); f.write("\n")

print()
print("MEASUREMENT_ITEMS_306.jsonl sha256 =", items_sha)
print("PREFLIGHT_FREEZE.json       sha256 =", sha256_file(fp))
print("family:", freeze["family_split"], "| condition:", freeze["condition_split"])
print("split :", freeze["solve_check_split"], "| shards:", freeze["rows_per_shard"])
print("PREFLIGHT_STATUS=PASS")
