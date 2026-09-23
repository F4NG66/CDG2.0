#!/usr/bin/env python3
"""Post-judging integrity audit + blinded-judgment freeze. Does NOT open the private map,
does NOT join labels to arms, prints only operational counts."""
import collections, hashlib, json
from pathlib import Path
W = Path(__file__).resolve().parent
X = W / "judging_execution"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def csha(o): return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

packet = [json.loads(l) for l in open(W / "JUDGE_PACKET_BLINDED_3038.jsonl")]
by_id = {r["review_id"]: (i, r) for i, r in enumerate(packet, 1)}
parts = {}
for p in (X / "judgment_parts_v1").glob("R_*.json"):
    d = json.loads(p.read_text()); assert p.stem == d["review_id"]; parts[d["review_id"]] = d
journal = [json.loads(l) for l in open(X / "CALL_JOURNAL.jsonl")]
raw = [json.loads(l) for l in open(X / "RAW_JUDGE_RESPONSES.jsonl")]
prior = json.load(open(W / "JUDGING_PREFLIGHT_FREEZE.json"))["aborted_launch_1"]["api_calls"]

E = []
def chk(name, ok, detail=""):
    if not ok: E.append(f"{name}: {detail}")
chk("all_ids_judged_exactly_once", set(parts) == set(by_id) and len(parts) == 3038, f"{len(parts)} parts")
ok_parts = [d for d in parts.values() if d["status"] == "PASS"]
fail_parts = [d for d in parts.values() if d["status"] == "JUDGE_FAILURE"]
chk("no_unknown_status", len(ok_parts) + len(fail_parts) == len(parts))
bad_vocab = bad_conf = bad_notes = bad_model = bad_bind = 0
for d in ok_parts:
    o, row = by_id[d["review_id"]]
    bad_vocab += d["outcome_label"] not in row["allowed_labels"]
    bad_conf += d["reviewer_confidence"] not in ("high", "medium", "low")
    bad_notes += not str(d["reviewer_notes"]).strip()
    bad_model += not (d["judge_model_returned"] == "deepseek-v4-pro" == d["judge_model_requested"] and d["temperature"] == 0 and d["thinking"] == "disabled")
    bad_bind += not (d["packet_row_sha256"] == csha(row) and d["blinded_ordinal"] == o)
chk("label_vocab_condition_compatible", bad_vocab == 0, str(bad_vocab)); chk("confidence_valid", bad_conf == 0); chk("notes_nonempty", bad_notes == 0)
chk("judge_settings", bad_model == 0); chk("packet_row_binding", bad_bind == 0)
# journal <-> attempts reconciliation
per = collections.Counter(j["review_id"] for j in journal)
mism = [rid for rid, d in parts.items() if per.get(rid, 0) != d["attempts"]]
chk("journal_reconciles_to_attempts", not mism and set(per) <= set(parts), f"{len(mism)} mismatches")
chk("journal_one_item_per_request", all(j["one_item_per_request"] for j in journal))
chk("journal_settings", all(j["model_requested"] == "deepseek-v4-pro" and j["temperature"] == 0 and j["thinking"] == {"type": "disabled"} for j in journal))
api_ok = [j for j in journal if j["api_status"] == "API_OK"]
chk("raw_retained_for_every_api_ok_call", len(raw) == len(api_ok) and {(r["call_seq"]) for r in raw} == {j["call_seq"] for j in api_ok})
# parser result vs raw: last raw response of each PASS item parses to the stored label
last = {}
for r in raw: last[r["review_id"]] = r
pmis = 0
for d in ok_parts:
    c = json.loads(last[d["review_id"]]["raw_response"]["choices"][0]["message"]["content"])
    pmis += (str(c["outcome_label"]).strip() != d["outcome_label"])
chk("parsed_result_matches_raw_response", pmis == 0, str(pmis))
retries = sum(d["attempts"] - 1 for d in parts.values())
api_exc = sum(j["api_status"] == "API_EXCEPTION" for j in journal)
# assemble ordered judgments
judg = []
for rid, (o, row) in sorted(by_id.items(), key=lambda kv: kv[1][0]):
    d = parts[rid]
    judg.append({"review_id": rid, "blinded_ordinal": o, "request_type": row["request_type"], **{k: v for k, v in d.items() if k not in ("review_id", "blinded_ordinal", "schema")}})
with open(W / "DEEPSEEK_JUDGMENTS_BLINDED_3038.jsonl", "w") as f:
    for j in judg: f.write(json.dumps(j, sort_keys=True, ensure_ascii=False) + "\n")
stats = {"expected_judgments": 3038, "observed_judgments": len(judg), "judge_failures": len(fail_parts),
         "api_calls_launch2": len(journal), "api_calls_aborted_launch1": prior, "api_calls_total": len(journal) + prior,
         "retries_launch2": retries, "api_exceptions_launch2": api_exc, "raw_responses_retained": len(raw)}
passed = not E and not fail_parts
audit = {"schema": "GENERALIZED_SAFETY_V2_M2_FRESH_JUDGMENT_INTEGRITY_AUDIT_V1",
         "status": "PASS" if passed else "BLOCKED", "errors": E, **stats,
         "private_map_used": False, "manual_relabeling": False, "efficacy_computed": False, "unblinded": False,
         "duplicate_final_judgments": 3038 - len({j["review_id"] for j in judg}), "missing_ids": len(set(by_id) - set(parts))}
(W / "JUDGMENT_INTEGRITY_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: audit[k] for k in ["status", "errors", *stats]}, sort_keys=True))
if passed:
    fr = {"schema": "GENERALIZED_SAFETY_V2_M2_FRESH_BLINDED_LABEL_FREEZE_V1", "status": "PASS_FROZEN", "JUDGING_AUDIT": "PASS", "JUDGING": "PASS_FROZEN",
          "unblinding_started": False, "efficacy_computed": False, "arm_selection": False,
          "artifact_sha256": {n: sha(W / n) for n in ["JUDGE_PACKET_BLINDED_3038.jsonl", "JUDGE_PRIVATE_MAP_3038.jsonl", "JUDGING_CONTRACT.json", "METRIC_CONTRACT.json",
                              "JUDGING_PREFLIGHT_FREEZE.json", "DEEPSEEK_JUDGMENTS_BLINDED_3038.jsonl", "JUDGMENT_INTEGRITY_AUDIT.json"]},
          "call_journal_sha256": sha(X / "CALL_JOURNAL.jsonl"), "raw_judge_responses_sha256": sha(X / "RAW_JUDGE_RESPONSES.jsonl"), **stats}
    (W / "BLINDED_LABEL_FREEZE.json").write_text(json.dumps(fr, indent=2, sort_keys=True) + "\n")
    print("JUDGING_AUDIT=PASS\nJUDGING=PASS_FROZEN\nBLINDED_LABEL_FREEZE_SHA256=" + sha(W / "BLINDED_LABEL_FREEZE.json"))
else:
    print("JUDGING_AUDIT=BLOCKED (no freeze written)")
