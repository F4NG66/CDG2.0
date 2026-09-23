"""
M3' step 6c — post-judging integrity audit and blinded label freeze (NEW_DEV baselines).

Same audit as generalized_safety_v2_m3_build_dev_judging_v2/03_integrity_audit_and_freeze_v2.py, on the BLINDED
judgments only; the private map is hashed but never joined here. DEVIATION FROM M3: no label distribution is
computed. In M3' every row is NEW_DEV, so a by-request_type distribution would equal the pooled step-7 support
counts (plus categories step 7 forbids inspecting) before the support check runs.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKET = HERE / "JUDGE_PACKET_BLINDED_622.jsonl"
PRIVATE = HERE / "JUDGE_PRIVATE_MAP_622.jsonl"
JUDG = HERE / "DEEPSEEK_JUDGMENTS_BLINDED_622.jsonl"
METRIC_SHA = "8fbaeec235cf1558a6092b7cc8bebf3f93724ab1e5426f9989bcf58851c84335"
PROMPT_SHA = "1249c06f067c22c14ad5a2a6fd87b74378b61a117785444cc7808fe62b95d0b9"


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pa = json.loads((HERE / "PACKET_AUDIT.json").read_text())
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
        "schema": "M3PRIME_NEW_DEV_JUDGMENT_INTEGRITY_AUDIT_V1",
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
        "packet_sha_matches_packet_audit": sha_file(PACKET) == pa["packet_sha256"],
        "private_map_sha_matches_packet_audit": sha_file(PRIVATE) == pa["private_map_sha256"],
        "system_prompt_sha_ok": sha_file(HERE / "JUDGE_SYSTEM_PROMPT.txt") == PROMPT_SHA,
        "metric_contract_sha256": METRIC_SHA,
        "label_distribution_computed": False,
        "private_map_joined": False,
        "case_identity_seen": False,
        "manual_relabeling_performed": False,
    }
    audit["PASS"] = (audit["judged_exactly_once"]
                     and audit["all_labels_within_allowed_set"]
                     and audit["all_confidence_valid"]
                     and audit["model_or_request_type_drift"] == 0
                     and audit["empty_reviewer_notes"] == 0
                     and audit["packet_sha_matches_packet_audit"]
                     and audit["private_map_sha_matches_packet_audit"]
                     and audit["system_prompt_sha_ok"])

    (HERE / "JUDGMENT_INTEGRITY_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    freeze = {
        "schema": "GENERALIZED_SAFETY_V2_M3PRIME_NEW_DEV_BLINDED_LABEL_FREEZE_V1",
        "contract_item": "M3prime checklist item 7 (NEW_DEV blinded judgments + integrity audit PASS_FROZEN)",
        "status": "PASS_FROZEN" if audit["PASS"] else "FAILED_AUDIT",
        "judgments_sha256": sha_file(JUDG),
        "packet_sha256": sha_file(PACKET),
        "private_map_sha256": sha_file(PRIVATE),
        "system_prompt_sha256": sha_file(HERE / "JUDGE_SYSTEM_PROMPT.txt"),
        "integrity_audit_sha256": sha_file(HERE / "JUDGMENT_INTEGRITY_AUDIT.json"),
        "call_journal_sha256": sha_file(HERE / "judging_execution/call_journal.jsonl"),
        "metric_contract_sha256": METRIC_SHA,
        "generation_freeze_results_sha256": pa["generation_freeze_results_sha256"],
        "judgment_count": len(judg),
        "private_map_used": False,
        "efficacy_computed": False,
        "labels_frozen": True,
        "scope": "M3prime NEW_DEV baseline B and C conditions, for the step-7 pooled support check and (only if "
                 "it passes) tau selection. Non-confirmatory.",
    }
    (HERE / "BLINDED_LABEL_FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "freeze": freeze}, indent=2, sort_keys=True))
    assert "outcome_label" not in json.dumps(audit)


if __name__ == "__main__":
    main()
