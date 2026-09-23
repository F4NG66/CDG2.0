"""
Validates scenario_match_audit.py against the three populations investigated
in generalized_safety_v2_m3_renellm_construction_fault_audit_v1: it must FAIL
on M3 BUILD/DEV (using the actually-declared-canonical base_request, which is
what the buggy pipeline would have shown a judge) and PASS on M1 held-out and
M2 confirmatory. This is a regression test proving the gate would have caught
the bug had it existed at construction time.
"""

import json
from pathlib import Path

from scenario_match_audit import audit, audit_or_raise

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV = ROOT / "analysis_output/rrae_development"


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def load_old_abcd():
    d = {}
    for l in open(ROOT / "data/abcd/ABCD_clean_v2_DIJA_Qwen_A_B_C_D.jsonl"):
        r = json.loads(l)
        d[r["id"]] = r["prompt"]
    return d


def load_v22():
    d = {}
    for l in open(ROOT / "data/abcd/v2_2_final/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl"):
        r = json.loads(l)
        p = int(r["paired_A_id"].split("_")[1])
        d[(r["bucket"], p)] = r["prompt"]
    return d


def m3_builddev_rows_as_the_buggy_pipeline_would_have_produced():
    """base_request = BUILD/DEV's declared canonical (Qwen file) — exactly what
    01_build_packet.py actually showed the judge. attacked_prompt = the real
    RENeLLM prompt the construction script produced. This reproduces the bug."""
    old = load_old_abcd()
    fp = jl(DEV / "generalized_safety_v2_renellm_v31_full_execution_v3/full_final_prompts.jsonl")
    rows = []
    for r in fp:
        pid = r["pair_id"]
        bucket = "A" if r["condition"].startswith("B") else "D"
        base = old.get(f"{bucket}_{pid:04d}")
        if base is None:
            continue
        rows.append({"id": pid, "scenario_id": r["scenario_id"],
                     "attacked_prompt": r["prompt"], "claimed_base_request": base})
    return rows


def m1_heldout_rows():
    v22 = load_v22()
    fp = jl(DEV / "generalized_safety_v2_m1_heldout_renellm_execution_v1/full_final_prompts.jsonl")
    rows = []
    for r in fp:
        pid = r["pair_id"]
        bucket = "A" if r["condition"].startswith("B") else "D"
        base = v22.get((bucket, pid))
        if base is None:
            continue
        rows.append({"id": pid, "scenario_id": r["scenario_id"],
                     "attacked_prompt": r["prompt"], "claimed_base_request": base})
    return rows


def m2_confirmatory_rows():
    bank = {r["fresh_group_id"]: r for r in jl(
        ROOT / "adaptive_steering_controller_cleanroom/analysis_output/controller_v2_fresh_source_bank_v1/FRESH_CANDIDATE_GROUPS.jsonl")}
    fp = jl(DEV / "generalized_safety_v2_m2_renellm_execution_v1/full_final_prompts.jsonl")
    rows = []
    for r in fp:
        gid = r["fresh_group_id"]
        b = bank.get(gid)
        if not b:
            continue
        field = "harmful_source" if r["condition"].startswith("B") else "benign_source"
        rows.append({"id": gid, "scenario_id": r["scenario_id"],
                     "attacked_prompt": r["prompt"], "claimed_base_request": b[field]["request"]})
    return rows


def main():
    results = {}

    m3_rows = m3_builddev_rows_as_the_buggy_pipeline_would_have_produced()
    m3_result = audit(m3_rows)
    results["M3_BUILD_DEV_reproducing_the_bug"] = {
        "expected": "FAIL", "got_PASS": m3_result["PASS"],
        "match": m3_result["match"], "mismatch": m3_result["mismatch"],
        "unresolved": m3_result["unresolved"],
        "gate_would_have_caught_it": not m3_result["PASS"],
    }
    try:
        audit_or_raise(m3_rows, context="regression-test M3 BUILD/DEV")
        results["M3_BUILD_DEV_reproducing_the_bug"]["audit_or_raise_raised"] = False
    except RuntimeError as e:
        results["M3_BUILD_DEV_reproducing_the_bug"]["audit_or_raise_raised"] = True
        results["M3_BUILD_DEV_reproducing_the_bug"]["error_message"] = str(e)

    m1_rows = m1_heldout_rows()
    m1_result = audit(m1_rows)
    results["M1_heldout_clean"] = {
        "expected": "PASS", "got_PASS": m1_result["PASS"],
        "match": m1_result["match"], "mismatch": m1_result["mismatch"],
        "unresolved": m1_result["unresolved"],
    }

    m2_rows = m2_confirmatory_rows()
    m2_result = audit(m2_rows)
    results["M2_confirmatory_clean"] = {
        "expected": "PASS", "got_PASS": m2_result["PASS"],
        "match": m2_result["match"], "mismatch": m2_result["mismatch"],
        "unresolved": m2_result["unresolved"],
    }

    out = Path(__file__).resolve().parent / "VALIDATION_REPORT.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))

    assert results["M3_BUILD_DEV_reproducing_the_bug"]["gate_would_have_caught_it"] is True
    assert results["M1_heldout_clean"]["got_PASS"] is True
    assert results["M2_confirmatory_clean"]["got_PASS"] is True
    print("\nVALIDATION_PASS: gate fails on the known-bad case and passes on both known-clean cases.")


if __name__ == "__main__":
    main()
