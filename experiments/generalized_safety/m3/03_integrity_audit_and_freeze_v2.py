"""
M3 BUILD+DEV judging v2 — post-judging integrity audit and label freeze.

Same audit as v1's 03_integrity_audit_and_freeze.py, pointed at the v2
packet/judgments (RENeLLM base_request corrected, DIJA reused unchanged from
v1). Runs on the BLINDED judgments only; the private map is hashed but never
joined here.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKET = HERE / "JUDGE_PACKET_BLINDED_1528.jsonl"
PRIVATE = HERE / "JUDGE_PRIVATE_MAP_1528.jsonl"
JUDG = HERE / "DEEPSEEK_JUDGMENTS_BLINDED_1528.jsonl"
METRIC_SHA = "8fbaeec235cf1558a6092b7cc8bebf3f93724ab1e5426f9989bcf58851c84335"


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    packet = jl(PACKET)
    judg = jl(JUDG)
    pmap = {r["blinded_id"]: r for r in packet}

    ids_p = [r["blinded_id"] for r in packet]
    ids_j = [r["blinded_id"] for r in judg]
    missing = sorted(set(ids_p) - set(ids_j))
    extra = sorted(set(ids_j) - set(ids_p))
    dup = [k for k, v in Counter(ids_j).items() if v > 1]

    label_ok, conf_ok, drift, empty_notes = 0, 0, 0, 0
    for r in judg:
        row = pmap.get(r["blinded_id"])
        if row is None:
            continue
        if r["outcome_label"] in row["allowed_labels"]:
            label_ok += 1
        if r["reviewer_confidence"] in row["allowed_confidence"]:
            conf_ok += 1
        if r.get("judge_model_returned") != "deepseek-v4-pro":
            drift += 1
        if not str(r.get("reviewer_notes", "")).strip():
            empty_notes += 1
        if r.get("request_type") != row["request_type"]:
            drift += 1

    audit = {
        "schema": "M3_BUILD_DEV_JUDGMENT_INTEGRITY_AUDIT_V2",
        "packet_items": len(packet),
        "judgments": len(judg),
        "judged_exactly_once": len(missing) == 0 and len(extra) == 0 and len(dup) == 0,
        "missing": len(missing), "unexpected": len(extra), "duplicate": len(dup),
        "all_labels_within_allowed_set": label_ok == len(judg),
        "all_confidence_valid": conf_ok == len(judg),
        "model_or_request_type_drift": drift,
        "empty_reviewer_notes": empty_notes,
        "judge_model": "deepseek-v4-pro", "temperature": 0, "thinking": "disabled",
        "one_item_per_request": True,
        "metric_contract_sha256": METRIC_SHA,
        "label_distribution_by_request_type": {},
        "private_map_joined": False,
        "case_identity_seen": False,
        "manual_relabeling_performed": False,
        "corrects": "generalized_safety_v2_m3_build_dev_judging_v1 (RENeLLM base_request fault; "
                    "see generalized_safety_v2_m3_renellm_construction_fault_audit_v1)",
    }
    by = {}
    for r in judg:
        by.setdefault(r["request_type"], Counter())[r["outcome_label"]] += 1
    audit["label_distribution_by_request_type"] = {k: dict(v) for k, v in by.items()}
    audit["PASS"] = (audit["judged_exactly_once"]
                     and audit["all_labels_within_allowed_set"]
                     and audit["all_confidence_valid"]
                     and audit["model_or_request_type_drift"] == 0
                     and audit["empty_reviewer_notes"] == 0)

    (HERE / "JUDGMENT_INTEGRITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True))

    freeze = {
        "schema": "GENERALIZED_SAFETY_V2_M3_BUILD_DEV_BLINDED_LABEL_FREEZE_V2",
        "status": "PASS_FROZEN" if audit["PASS"] else "FAILED_AUDIT",
        "judgments_sha256": sha_file(JUDG),
        "packet_sha256": sha_file(PACKET),
        "private_map_sha256": sha_file(PRIVATE),
        "system_prompt_sha256": sha_file(HERE / "JUDGE_SYSTEM_PROMPT.txt"),
        "integrity_audit_sha256": sha_file(HERE / "JUDGMENT_INTEGRITY_AUDIT.json"),
        "metric_contract_sha256": METRIC_SHA,
        "judgment_count": len(judg),
        "private_map_used": False,
        "efficacy_computed": False,
        "labels_frozen": True,
        "assistant_labels_used_for_fitting": False,
        "scope": "BUILD + DEV baseline B and C conditions, for M3 gate fitting and "
                 "DEV threshold selection. Non-confirmatory.",
        "supersedes": "generalized_safety_v2_m3_build_dev_judging_v1/BLINDED_LABEL_FREEZE.json "
                       "(kept, not deleted — see SUPERSEDED_BY.json there)",
        "renellm_base_request_source": "data/abcd/v2_2_final/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl "
                                        "(the text RENeLLM construction actually attacked) — documented "
                                        "lineage deviation vs DIJA_LINEAGE_BINDING.json, see "
                                        "PACKET_REBUILD_AUDIT.json",
        "dija_labels": "reused unchanged from v1 (800 rows, no re-judging)",
    }
    (HERE / "BLINDED_LABEL_FREEZE.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "freeze": freeze}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
