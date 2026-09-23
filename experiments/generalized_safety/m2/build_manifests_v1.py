#!/usr/bin/env python3
"""Step 5: CPU-only LLaDA generation manifests (confirmatory 3038 rows + non-confirmatory smoke).
No model weights are loaded; only the tokenizer and the frozen input builders.

Paired-seed rule (M1, preserved): generation_seed = 4 * pair_id + (1 if condition B else 2), shared by all
7 arms and by both attack families for the same group+condition. M1's integer pair_id is bound for the
fresh population to the frozen group ordinal = 0-based index of fresh_group_id in the sorted list of all
126 population IDs (fixed before any construction outcome existed).

DIJA input route (M1 rule, pair-level): a group's DIJA rows (B and C) use DIJA_COMPAT iff its B_DIJA
prompt raises the localization error ("Could not locate clean/harm prompt") under the baseline-preparation
source helper (SHA 9c4c2b06..., from which the compat helper was patched); otherwise DIJA_ORIGINAL.
(M1 precedent: recovered pair 479 -> both B_DIJA and C_DIJA on DIJA_COMPAT.) Both helpers must produce
byte-identical input ids (asserted). ReNeLLM: RENELLM_STANDARD."""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m2_llada_runtime_v1 as RT  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEV = RT.DEV
POP = DEV / "generalized_safety_v2_m2_confirmatory_population_freeze_v1/M2_CONFIRMATORY_126_GROUP_MANIFEST.jsonl"
COMBINED = DEV / "generalized_safety_v2_m2_confirmatory_variant_population_freeze_v1/M2_CONFIRMATORY_VARIANT_POPULATION_FREEZE.json"
DIJA_PROMPTS = DEV / "generalized_safety_v2_m2_dija_variant_freeze_v1/M2_DIJA_FINAL_PROMPTS_PRIVATE.jsonl"
DIJA_FREEZE = DEV / "generalized_safety_v2_m2_dija_variant_freeze_v1/FREEZE.json"
REN_PROMPTS = DEV / "generalized_safety_v2_m2_renellm_execution_v1/full_final_prompts.jsonl"
REN_FREEZE = DEV / "generalized_safety_v2_m2_renellm_variant_freeze_v1/FREEZE.json"
MEAS = DEV / "generalized_safety_v2_m2_beta_calibration_measurement_v1/MEASUREMENT_ITEMS_306.jsonl"
SHARDS = 14
SOURCE_CASES_PER_SHARD = 31


def H(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def jl(p): return [json.loads(x) for x in open(p) if x.strip()]


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    return RT.sha_file(p)


class Builder:
    def __init__(self):
        self.Hm = RT.load_historical()
        self.m1 = self.Hm["m1"]
        assert (self.m1.LAYER, self.m1.MASK_ID, self.m1.GEN_LENGTH, self.m1.TEMPERATURE, self.m1.REMASK) == (16, 126336, 128, 0.2, "low_confidence")
        self.betas, self.mapping = RT.load_betas()
        self.tok = AutoTokenizer.from_pretrained(str(RT.MODEL), trust_remote_code=True, local_files_only=True)
        rh = DEV / "generalized_safety_v2_llada_baseline_preparation_v2_realized2328/frozen_generation_helpers.py"
        assert RT.sha_file(rh) == "9c4c2b067b1f02ee4ed05384812ce43ac95da563585baedcc9a4a327d9a89b5b"
        self.route_helper = RT.load_module("route_source_helper", rh)
        if hasattr(self.m1, "sha_text"):
            self.route_helper.sha256_text = self.m1.sha_text

    def route_and_input(self, sc):
        base = {"attack_family": sc["family"], "condition": sc["condition"], "pair_id": sc["pair_id"],
                "prompt": sc["prompt"], "clean_semantic_request": sc.get("clean")}
        if sc["family"] == "RENELLM":
            x, _, meta = self.m1.build_input(base, self.tok, self.Hm["helper"], self.Hm["baseline"])
            return "RENELLM_STANDARD", x, meta
        # M1 route rule (pair-level), decided on the group's B_DIJA prompt with the source helper.
        bbase = {"attack_family": "DIJA", "condition": "B", "pair_id": sc["pair_id"],
                 "prompt": sc["route_B_prompt"], "clean_semantic_request": sc["route_B_clean"]}
        try:
            self.m1.build_input(bbase, self.tok, self.route_helper, self.Hm["baseline"])
            route = "DIJA_ORIGINAL"
        except RuntimeError as e:
            if "Could not locate clean/harm prompt" not in str(e):
                raise
            route = "DIJA_COMPAT"
        use = self.Hm["compat_helper"] if route == "DIJA_COMPAT" else self.Hm["helper"]
        x, _, meta = self.m1.build_input(base, self.tok, use, self.Hm["baseline"])
        # both frozen helpers must yield byte-identical inputs (denoising identical across routes)
        other = self.Hm["helper"] if route == "DIJA_COMPAT" else self.Hm["compat_helper"]
        x2, _, _ = self.m1.build_input(base, self.tok, other, self.Hm["baseline"])
        assert torch.equal(x, x2), "route helpers disagree on input ids"
        return route, x, meta

    def items_for(self, sc, arms, split):
        route, x, meta = self.route_and_input(sc)
        init_count = int((x == RT.MASK_ID).sum())
        steps = init_count if sc["family"] == "DIJA" else 128
        if sc["family"] == "RENELLM":
            assert init_count == 128
        helper_path = RT.P["compat_helper"] if route == "DIJA_COMPAT" else Path(self.m1.HELPER_PATH)
        out = []
        for arm in arms:
            method, d = RT.arm_parts(arm)
            if method == "BASELINE":
                beta, bsrc = 0.0, None
            else:
                key = self.mapping["mapping"][d]["m1_key" if method == "M1_ADD" else "m2_key"]
                f = self.mapping["m1_beta_file" if method == "M1_ADD" else "m2_beta_file"]
                beta = self.betas[method][d]
                bsrc = {"file": f, "key": key, "file_sha256": RT.sha_file(f), "mapping_sha256": RT.EXPECT["beta_mapping"]}
            out.append({
                "schema": "GENERALIZED_SAFETY_V2_M2_LLADA_GENERATION_ROW_V1",
                "item_id": f"{sc['id_prefix']}_{sc['group_key']}_{sc['v2_condition']}_{arm}",
                "split": split, "arm": arm, "method": method, "direction": d,
                "steering_condition": arm, "vector_key": d, "beta": beta, "beta_source": beta_src_or_none(bsrc),
                "attack_family": sc["family"], "condition": sc["condition"], "v2_condition": sc["v2_condition"],
                "group_key": sc["group_key"], "group_ordinal": sc["pair_id"], "pair_id": sc["pair_id"],
                "source_case_id": sc["source_case_id"], "domain": sc["domain"],
                "prompt": sc["prompt"], "prompt_sha256": H(sc["prompt"]),
                "clean_semantic_request": sc.get("clean"),
                "generation_seed": sc["seed"],
                "input_helper_route": route, "input_helper_sha256": RT.sha_file(helper_path),
                "runtime_mode": meta["generation_mode"],
                "expected_initial_ids_sha256": self.m1.tensor_sha256(x), "expected_initial_fillable_mask_count": init_count,
                "expected_trajectory_steps": steps,
                "layer_zero_based": 16, "token_scope": "current_mask", "scope": "current_mask", "schedule": "persistent",
                "gen_length": 128, "temperature": 0.2, "remasking": "low_confidence",
                "model_path": str(RT.MODEL), "outcome_based_selection": False,
            })
        return out


def beta_src_or_none(b):
    return b


def seed_rule(ordinal, condition):
    return 4 * ordinal + (1 if condition == "B" else 2)


def confirmatory_source_cases():
    comb = json.loads(COMBINED.read_text())
    assert RT.sha_file(COMBINED) == "e605c53b5efbef32a8671f3225f7facf85e5bf512d1b31951b42c95a841c235d"
    assert RT.sha_file(DIJA_FREEZE) == "c52731b7f3a2386d290e5d81cf40924d4955cd15f8da57a58ba7c462cf2c7309"
    rf = json.loads(REN_FREEZE.read_text())
    assert rf["status"] == "PASS_FROZEN" and rf["final_prompts_sha256"] == RT.sha_file(REN_PROMPTS)
    dfz = json.loads(DIJA_FREEZE.read_text())
    assert dfz["final_prompts_sha256"] == RT.sha_file(DIJA_PROMPTS)
    pop = jl(POP)
    assert len(pop) == 126
    ordinal = {g: i for i, g in enumerate(sorted(r["fresh_group_id"] for r in pop))}
    domain = {r["fresh_group_id"]: r["domain"] for r in pop}
    dija_acc = (DEV / "generalized_safety_v2_m2_confirmatory_variant_population_freeze_v1/DIJA_ACCEPTED_GROUP_IDS.txt").read_text().split()
    ren_acc = (DEV / "generalized_safety_v2_m2_confirmatory_variant_population_freeze_v1/RENELLM_ACCEPTED_GROUP_IDS.txt").read_text().split()
    assert len(dija_acc) == comb["DIJA"]["population_size"] == 115 and len(ren_acc) == comb["RENELLM"]["population_size"] == 102
    assert H("\n".join(dija_acc)) == comb["DIJA"]["accepted_group_ids_sha256"]
    assert H("\n".join(ren_acc)) == comb["RENELLM"]["accepted_group_ids_sha256"]
    scs = []
    for r in jl(DIJA_PROMPTS):
        assert r["fresh_group_id"] in dija_acc and r["prompt_sha256"] == H(r["prompt"])
        scs.append({"family": "DIJA", "condition": r["condition"], "v2_condition": f"{r['condition']}_DIJA",
                    "group_key": r["fresh_group_id"], "prompt": r["prompt"], "clean": r["clean_prompt"]})
    for r in jl(REN_PROMPTS):
        c = r["condition"].split("_")[0]
        assert r["fresh_group_id"] in ren_acc and r["prompt_sha256"] == H(r["prompt"]) and r["condition"] in ("B_RENELLM", "C_RENELLM")
        scs.append({"family": "RENELLM", "condition": c, "v2_condition": r["condition"],
                    "group_key": r["fresh_group_id"], "prompt": r["prompt"], "clean": None})
    bprompt = {s["group_key"]: s for s in scs if s["family"] == "DIJA" and s["condition"] == "B"}
    for s in scs:
        if s["family"] == "DIJA":
            s["route_B_prompt"], s["route_B_clean"] = bprompt[s["group_key"]]["prompt"], bprompt[s["group_key"]]["clean"]
        s["pair_id"] = ordinal[s["group_key"]]
        s["seed"] = seed_rule(s["pair_id"], s["condition"])
        s["domain"] = domain[s["group_key"]]
        s["source_case_id"] = f"{s['group_key']}|{s['family']}|{s['condition']}"
        s["id_prefix"] = "m2conf"
    scs.sort(key=lambda s: (s["family"], s["group_key"], s["condition"]))
    cnt = Counter((s["family"], s["condition"]) for s in scs)
    assert cnt == {("DIJA", "B"): 115, ("DIJA", "C"): 115, ("RENELLM", "B"): 102, ("RENELLM", "C"): 102}, cnt
    assert {s["group_key"] for s in scs if s["family"] == "DIJA"} == set(dija_acc)
    assert {s["group_key"] for s in scs if s["family"] == "RENELLM"} == set(ren_acc)
    assert len({s["source_case_id"] for s in scs}) == 434
    return scs, ordinal


def smoke_source_cases():
    meas = jl(MEAS)
    pick = [(5, "B_DIJA"), (5, "C_DIJA"), (5, "B_RENELLM"), (5, "C_RENELLM"), (479, "C_DIJA")]
    scs = []
    for pid, v2 in pick:
        r = next(x for x in meas if x["pair_id"] == pid and x["v2_condition"] == v2)
        assert r["m2_split"] in ("DEV_CHECK", "DEV_SOLVE") and r["generation_seed"] == seed_rule(pid, r["condition"])
        scs.append({"family": r["attack_family"], "condition": r["condition"], "v2_condition": v2,
                    "group_key": f"DEVPAIR_{pid:04d}", "pair_id": pid, "seed": r["generation_seed"],
                    "prompt": r["prompt"], "clean": r.get("clean_semantic_request"), "domain": r["domain"],
                    "source_case_id": f"DEVPAIR_{pid:04d}|{r['attack_family']}|{r['condition']}", "id_prefix": "m2smoke",
                    "_meas": r})
        if r["attack_family"] == "DIJA":
            rb = next(x for x in meas if x["pair_id"] == pid and x["v2_condition"] == "B_DIJA")
            scs[-1]["route_B_prompt"], scs[-1]["route_B_clean"] = rb["prompt"], rb["clean_semantic_request"]
    return scs


def main():
    B = Builder()
    # ------------------------------------------------ confirmatory
    scs, ordinal = confirmatory_source_cases()
    rows = []
    for i, sc in enumerate(scs):
        for it in B.items_for(sc, RT.ARMS, "M2_HELDOUT_CONFIRMATORY_EVALUATION"):
            it["shard_index"] = i // SOURCE_CASES_PER_SHARD
            rows.append(it)
    assert len(rows) == 3038 and len({r["item_id"] for r in rows}) == 3038
    assert max(r["shard_index"] for r in rows) == SHARDS - 1
    fam = Counter(r["attack_family"] for r in rows)
    assert fam == {"DIJA": 1610, "RENELLM": 1428}
    # paired seeds: one seed per source case across all 7 arms; same seed across families for same group+condition
    per_sc = {}
    for r in rows:
        per_sc.setdefault(r["source_case_id"], set()).add(r["generation_seed"])
    assert all(len(v) == 1 for v in per_sc.values())
    # baseline reuse check (section 8): search every existing result file for these exact prompt bytes
    prompt_shas = {r["prompt_sha256"] for r in rows}
    reuse_hits = []
    for f in RT.ROOT.glob("analysis_output/**/*.jsonl"):
        if "generalized_safety_v2_m2_llada_executor_v1" in str(f) or (
                "generalized_safety_v2_m2_" in str(f) and ("renellm" in str(f) or "dija" in str(f))):
            continue  # construction artifacts themselves (not LLaDA outputs)
        try:
            with open(f, errors="ignore") as fh:
                for line in fh:
                    if '"prompt_sha256"' in line and any(s in line for s in prompt_shas):
                        reuse_hits.append(str(f)); break
        except OSError:
            pass
    man = HERE / "M2_LLADA_CONFIRMATORY_MANIFEST_PRIVATE_3038.jsonl"
    man_sha = write_jsonl(man, rows)
    audit = {
        "schema": "GENERALIZED_SAFETY_V2_M2_LLADA_MANIFEST_AUDIT_V1",
        "rows": len(rows), "source_cases": len(per_sc), "arms": list(RT.ARMS), "shards": SHARDS,
        "rows_per_shard": dict(sorted(Counter(r["shard_index"] for r in rows).items())),
        "family_rows": dict(fam), "family_source_cases": dict(Counter(s["family"] for s in scs)),
        "route_counts": dict(Counter(r["input_helper_route"] for r in rows if r["arm"] == "BASELINE")),
        "seed_rule": "generation_seed = 4*group_ordinal + (1 if B else 2); group_ordinal = index of fresh_group_id in sorted(126 population ids)",
        "seed_shared_across_7_arms": True,
        "seed_shared_across_families_for_same_group_condition": True,
        "group_ordinal_sha256": H(json.dumps(ordinal, sort_keys=True)),
        "betas_loaded_via_mapping": {m: B.betas[m] for m in B.betas},
        "baseline_reuse": {"existing_llada_outputs_with_identical_prompt_bytes": reuse_hits,
                           "decision": "GENERATE_BASELINE_ONCE_PER_SOURCE_CASE" if not reuse_hits else "REVIEW_REQUIRED"},
        "manifest_sha256": man_sha, "model_calls_made": 0, "llada_generation_started": False,
    }
    (HERE / "M2_LLADA_MANIFEST_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print("confirmatory rows", len(rows), "sha", man_sha, "routes", audit["route_counts"], "reuse hits", reuse_hits)
    # ------------------------------------------------ smoke (non-confirmatory BUILD/DEV)
    srows = []
    for sc in smoke_source_cases():
        arms = ["BASELINE", "M1_ADD_V_ALL", "M2_DMNP_V_ALL"]
        if sc["v2_condition"] == "B_DIJA" and sc["pair_id"] == 5:
            arms += ["M1_ADD_V_DIJA", "M2_DMNP_V_DIJA"]
        if sc["v2_condition"] == "B_RENELLM" and sc["pair_id"] == 5:
            arms += ["M1_ADD_V_RENELLM", "M2_DMNP_V_RENELLM"]
        items = B.items_for(sc, arms, "NON_CONFIRMATORY_BUILD_DEV_SMOKE")
        for it in items:
            m = sc["_meas"]
            assert it["expected_initial_ids_sha256"] == m["expected_initial_ids_sha256"], "smoke CPU input parity vs historical"
            assert it["input_helper_route"] == m["input_helper_route"]
            it["historical_dev_m1_item_id"] = (f"v2_dev_m1_{sc['pair_id']:04d}_{sc['v2_condition']}_{it['direction']}"
                                               if it["method"] == "M1_ADD" else None)
            variants = ["WRAPPED"]
            if it["method"] in ("BASELINE", "M1_ADD"):
                variants.append("UNWRAPPED")
            if it["method"] == "M1_ADD":
                variants.append("COPY_LOOP_M1")
            if it["method"] == "M2_DMNP" and sc["pair_id"] == 5 and sc["condition"] == "B" and it["direction"] == "V_ALL":
                variants.append("REPEAT")
            for v in variants:
                r = dict(it); r["smoke_variant"] = v
                r["capture"] = (v == "WRAPPED" and sc["pair_id"] == 5 and sc["condition"] == "B" and it["direction"] == "V_ALL")
                srows.append(r)
    sm = HERE / "M2_LLADA_SMOKE_MANIFEST_DEV.jsonl"
    print("smoke rows", len(srows), "sha", write_jsonl(sm, srows),
          dict(Counter(r["smoke_variant"] for r in srows)), dict(Counter(r["arm"] for r in srows)))


if __name__ == "__main__":
    main()
