"""
M3 BUILD+DEV baseline judging — packet rebuild, VERSION 2.

Remediation for the RENeLLM construction fault documented in
generalized_safety_v2_m3_renellm_construction_fault_audit_v1/SCOPE_AUDIT.json:
BUILD/DEV's RENeLLM construction (run_renellm_v31_full_EXECUTION_V3.py) sourced
per-pair text from data/abcd/v2_2_final/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl,
not from BUILD/DEV's declared canonical (DIJA_LINEAGE_BINDING.json,
data/abcd/ABCD_clean_v2_DIJA_Qwen_A_B_C_D.jsonl). v1's judge packet showed the
judge the DECLARED-canonical text as base_request for B_RENELLM/C_RENELLM,
which is not the request that was actually attacked.

DOCUMENTED LINEAGE DEVIATION vs DIJA_LINEAGE_BINDING.json: this rebuild does
NOT re-point RENeLLM construction at the declared-canonical file (that would
require reconstructing and regenerating all 728 RENeLLM attacks/responses —
out of scope, not authorized). Instead it corrects the JUDGE-VISIBLE
base_request for B_RENELLM/C_RENELLM to the v2.2-final text that construction
ACTUALLY attacked, so the judge is scoring the response against the request it
is actually a response to. B_DIJA/C_DIJA are unaffected (DIJA construction
reads/writes the declared-canonical file directly) and are reused unchanged,
labels included — this rebuild does not re-judge DIJA.

model_response, domain, request_type, allowed_labels, allowed_confidence,
blinded_id, blinded_ordinal are all carried over unchanged from v1 for every
row (blinding blind seed is unchanged: only base_request for the 728 RENeLLM
rows is corrected).

Every resolvable RENeLLM row is checked with the mandatory fail-closed
scenario-match audit (generalized_safety_v2_m3_scenario_match_audit_v1) before
this script will write anything.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__import__("os").environ["CDG_WORK_ROOT"] + "/analysis_output/rrae_development/"
                            "generalized_safety_v2_m3_scenario_match_audit_v1")))
from scenario_match_audit import audit_or_raise  # noqa: E402

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV = ROOT / "analysis_output/rrae_development"
V1 = DEV / "generalized_safety_v2_m3_build_dev_judging_v1"
HERE = Path(__file__).resolve().parent
CANON_V22 = ROOT / "data/abcd/v2_2_final/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl"
ACCEPTED_PROMPTS = DEV / "generalized_safety_v2_renellm_v31_full_execution_v3/RENELLM_V31_ACCEPTED_PROMPTS_FINAL_728.jsonl"
DIJA_BINDING_SHA = "4ab0b3b5eae974575e882284f86affceca73b152b904d7110a3fab66b4ae8256"


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    v1_packet = {r["blinded_id"]: r for r in jl(V1 / "JUDGE_PACKET_BLINDED_1528.jsonl")}
    priv = jl(V1 / "JUDGE_PRIVATE_MAP_1528.jsonl")
    assert len(v1_packet) == len(priv) == 1528

    v22 = {}
    for x in jl(CANON_V22):
        p = int(x["paired_A_id"].split("_")[1])
        v22[(x["bucket"], p)] = x["prompt"]

    accepted = {(r["pair_id"], r["condition"]): r for r in jl(ACCEPTED_PROMPTS)}

    # ---- fail-closed scenario-match audit on every resolvable RENeLLM row ----
    audit_rows = []
    for p in priv:
        if p["condition"] not in ("B_RENELLM", "C_RENELLM"):
            continue
        pid = p["pair_id"]
        bucket = "A" if p["condition"] == "B_RENELLM" else "D"
        claimed = v22.get((bucket, pid))
        acc = accepted.get((pid, p["condition"]))
        if claimed is None or acc is None:
            continue  # unresolved — allowed, reported below, never silently treated as a pass
        audit_rows.append({"id": pid, "scenario_id": acc["scenario_id"],
                           "attacked_prompt": acc["prompt"], "claimed_base_request": claimed})

    result = audit_or_raise(audit_rows, context="M3 BUILD/DEV RENeLLM packet rebuild v2")
    print(f"SCENARIO_MATCH_AUDIT: {result['match']} match / {result['mismatch']} mismatch / "
          f"{result['unresolved']} unresolved of {len(audit_rows)} checked rows "
          f"(728 total RENeLLM rows in private map)")

    # ---- build v2 packet ----
    new_packet, corrected, reused_unchanged = [], 0, 0
    unresolved_renellm = []
    for p in priv:
        bid = p["blinded_id"]
        row = dict(v1_packet[bid])  # copy: blinded_id, blinded_ordinal, request_type,
                                     # domain, model_response, allowed_labels, allowed_confidence
        if p["condition"] in ("B_RENELLM", "C_RENELLM"):
            pid = p["pair_id"]
            bucket = "A" if p["condition"] == "B_RENELLM" else "D"
            new_base = v22.get((bucket, pid))
            if new_base is None:
                unresolved_renellm.append({"blinded_id": bid, "pair_id": pid, "condition": p["condition"]})
                new_packet.append(row)  # keep v1 base_request rather than write nothing
                continue
            row["base_request"] = new_base
            corrected += 1
        else:
            reused_unchanged += 1
        new_packet.append(row)

    assert corrected + len(unresolved_renellm) == 728, (corrected, len(unresolved_renellm))
    assert reused_unchanged == 800

    pk = HERE / "JUDGE_PACKET_BLINDED_1528.jsonl"
    pm = HERE / "JUDGE_PRIVATE_MAP_1528.jsonl"
    with pk.open("w") as fh:
        for r in new_packet:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    with pm.open("w") as fh:
        for r in priv:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")

    audit_out = {
        "schema": "M3_BUILD_DEV_JUDGE_PACKET_REBUILD_AUDIT_V2",
        "documented_lineage_deviation": {
            "vs": "generalized_safety_v2_build_dev_freeze/DIJA_LINEAGE_BINDING.json "
                  f"(canonical_sha256 {DIJA_BINDING_SHA})",
            "deviation": "B_RENELLM/C_RENELLM base_request is sourced from "
                         "data/abcd/v2_2_final/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl "
                         "(what RENeLLM construction actually attacked), NOT from "
                         "DIJA_LINEAGE_BINDING's declared canonical file — because the "
                         "already-generated 728 attacks/responses were built against v2.2-final, "
                         "and re-judging against the correct-for-what-was-attacked text is the "
                         "chosen remediation (re-judge, not rebuild generation). B_DIJA/C_DIJA are "
                         "unaffected — DIJA construction reads/writes the declared-canonical file "
                         "directly — and are excluded from this deviation.",
            "authorized_by": "project owner instruction, recorded 2026-09-22",
        },
        "rows_total": len(new_packet),
        "rows_corrected_renellm": corrected,
        "rows_reused_unchanged_dija": reused_unchanged,
        "rows_unresolved_renellm_kept_v1_base_request": unresolved_renellm,
        "scenario_match_audit": {k: v for k, v in result.items() if k != "mismatches"},
        "model_response_domain_request_type_labels_confidence": "unchanged from v1 for every row",
        "blinding_seed_unchanged": True,
        "packet_sha256": sha_file(pk),
        "private_map_sha256": sha_file(pm),
        "supersedes": "generalized_safety_v2_m3_build_dev_judging_v1 (kept, not deleted; "
                       "see SUPERSEDED_BY.json there)",
    }
    (HERE / "PACKET_REBUILD_AUDIT.json").write_text(json.dumps(audit_out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in audit_out.items() if k != "rows_unresolved_renellm_kept_v1_base_request"},
                      indent=2, sort_keys=True))
    print(f"\nunresolved RENeLLM rows (kept v1 base_request, flagged): {len(unresolved_renellm)}")


if __name__ == "__main__":
    main()
