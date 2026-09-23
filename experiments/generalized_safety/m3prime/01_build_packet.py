"""
M3' step 6a — blinded judge packet for the NEW_DEV baseline generations (622 rows).

Judge contract: M1 HELD-OUT (metric contract 8fbaeec2...), deepseek-v4-pro, same system prompt as M1 held-out /
M2 fresh / M3 BUILD-DEV (byte-identical, sha 1249c06f...). Structure follows
generalized_safety_v2_m3_build_dev_judging_v1/01_build_packet.py.

Base request (never the attacked prompt), verified three ways for every NEW_DEV group x {B, C}:
  1. ReNeLLM construction clean text (M3PRIME_RENELLM_CLEAN_TEXT_PRIVATE.jsonl) ==
     DIJA construction source text (M3PRIME_DIJA_REQUEST_MANIFEST_PRIVATE_330.jsonl);
  2. frozen-normalised hash == NEW_DEV identity manifest's normalized_request_sha256;
  3. DIJA generation rows: manifest clean_semantic_request == base request.
RENELLM rows additionally pass the mandatory fail-closed scenario-match audit (attacked prompt vs base request),
which must reproduce the step-4 frozen audit counts exactly (M3PRIME_RENELLM_VARIANT_AUDIT.json).

Refuses to run unless the generation freeze is PASS_FROZEN and the results file matches its SHA.
Blinding: packet carries no item/group id, condition, family, split or arm; private map written separately and not
joined until the judgment freeze.
"""
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV = ROOT / "analysis_output/rrae_development"
HERE = Path(__file__).resolve().parent
GEN = DEV / "generalized_safety_v2_m3prime_llada_baseline_v1"
CON = DEV / "generalized_safety_v2_m3prime_attack_construction_v1"
NEWDEV = DEV / "generalized_safety_v2_m3prime_new_dev_freeze_v1/M3PRIME_NEW_DEV_GROUP_MANIFEST.jsonl"
METRIC = DEV / "generalized_safety_v2_m1_heldout_judging_contract_v1/HELDOUT_METRIC_CONTRACT.json"
METRIC_SHA = "8fbaeec235cf1558a6092b7cc8bebf3f93724ab1e5426f9989bcf58851c84335"
PROMPT_SRC = DEV / "generalized_safety_v2_m3_build_dev_judging_v2/JUDGE_SYSTEM_PROMPT.txt"
PROMPT_SHA = "1249c06f067c22c14ad5a2a6fd87b74378b61a117785444cc7808fe62b95d0b9"
BLIND_SEED = "M3PRIME_NEW_DEV_JUDGING_BLIND_20260922"

sys.path.insert(0, str(DEV / "generalized_safety_v2_m3_scenario_match_audit_v1"))
from scenario_match_audit import audit_or_raise, extract_embedded, jaccard, MATCH_THRESHOLD  # noqa: E402


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def norm(t):
    """Frozen normalisation — byte-identical to the fresh-bank builder."""
    t = str(t or "").lower().replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"<\|mdm_mask\|>|<mask:\d+>|<<\s*tpl\s*>>|<</\s*tpl\s*>>", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    gf = json.loads((GEN / "GENERATION_FREEZE.json").read_text())
    assert gf["status"] == "PASS_FROZEN", "generation not PASS_FROZEN — refusing to build packet"
    assert sha_file(GEN / "GENERATION_RESULTS.jsonl") == gf["results_sha256"], "results changed since freeze"
    assert sha_file(METRIC) == METRIC_SHA, "M1 held-out metric contract SHA mismatch"
    assert sha_file(PROMPT_SRC) == PROMPT_SHA, "judge system prompt SHA mismatch"
    LABELS = json.load(open(METRIC))["judge_labels"]
    CONF = LABELS["confidence"]

    newdev = {r["group_id"]: r for r in jl(NEWDEV)}
    ren_txt = {(r["fresh_group_id"], r["condition"]): r["text"] for r in jl(CON / "M3PRIME_RENELLM_CLEAN_TEXT_PRIVATE.jsonl")}
    dija_txt = {(r["source_id"], "B" if r["branch"] == "HARMFUL_B" else "C"): r["source_text"]
                for r in jl(CON / "M3PRIME_DIJA_REQUEST_MANIFEST_PRIVATE_330.jsonl")}
    assert len(ren_txt) == len(dija_txt) == 330
    base = {}
    for k, t in ren_txt.items():
        g, c = k
        assert dija_txt[k] == t, f"base request disagrees between constructions at {k}"
        src = newdev[g]["harmful_source" if c == "B" else "benign_source"]
        assert sha_text(norm(t)) == src["normalized_request_sha256"], f"NEW_DEV identity hash mismatch at {k}"
        base[k] = t

    man = {r["item_id"]: r for r in jl(GEN / "M3PRIME_LLADA_BASELINE_MANIFEST_622.jsonl")}
    res = jl(GEN / "GENERATION_RESULTS.jsonl")
    assert len(res) == 622 and set(r["item_id"] for r in res) == set(man)

    ren_final = {(r["fresh_group_id"], r["condition"]): r for r in jl(CON / "renellm_final_prompts.jsonl")}
    audit_rows, items, stats, dija_clean_checked = [], [], Counter(), 0
    for r in res:
        m = man[r["item_id"]]
        g, c, fam = m["group_key"], m["condition"], m["attack_family"]
        br = base[(g, c)]
        if fam == "DIJA":
            assert m["clean_semantic_request"].strip() == br.strip(), f"DIJA clean_semantic_request != base at {g}|{c}"
            dija_clean_checked += 1
        else:
            fp = ren_final[(g, f"{c}_RENELLM")]
            assert fp["prompt"] == m["prompt"], f"ReNeLLM attacked prompt != generation prompt at {g}|{c}"
            audit_rows.append({"id": f"{g}|{c}_RENELLM", "scenario_id": fp["scenario_id"], "attacked_prompt": fp["prompt"],
                               "claimed_base_request": br})
        rtype = "harmful_request_safety" if c == "B" else "benign_request_utility"
        items.append({"_item_id": r["item_id"], "_group_id": g, "_condition": f"{c}_{fam}", "_family": fam,
                      "_response_sha256": r["decoded_output_sha256"], "_domain": m["domain"],
                      "base_request": br, "model_response": r["decoded_output"], "request_type": rtype,
                      "allowed_labels": LABELS[rtype], "allowed_confidence": CONF})
        stats[f"{c}_{fam}"] += 1

    sm = audit_or_raise(audit_rows, context="M3prime NEW_DEV judge packet")
    # must reproduce the step-4 frozen audit exactly (224 match / 0 mismatch / 80 unresolved-by-extractor);
    # unresolved rows are still covered by the three-way base-request verification above
    frozen_sm = json.load(open(CON / "M3PRIME_RENELLM_VARIANT_AUDIT.json"))["scenario_match_audit"]
    assert len(audit_rows) == 304 and sm["mismatch"] == 0, sm
    assert all(sm[k] == frozen_sm[k] for k in ("total", "match", "mismatch", "unresolved")), (sm, frozen_sm)

    # ---- supplementary check for the extractor-unresolved rows (all table_filling, \multicolumn layout) ----
    TABLE_MC = re.compile(r"\\multicolumn\{2\}\{\|c\|\}\{(.*?)\}\s*\\\\", re.S)
    unresolved = [x for x in audit_rows if extract_embedded(x["scenario_id"], x["attacked_prompt"]) is None]
    sup = []
    for x in unresolved:
        mm = TABLE_MC.search(x["attacked_prompt"]) if x["scenario_id"] == "table_filling" else None
        emb = mm.group(1).strip() if mm else None
        sup.append({"id": x["id"], "extracted": emb is not None,
                    "jaccard": None if emb is None else jaccard(emb, x["claimed_base_request"]), "emb": emb})
    ctrl = [jaccard(s["emb"], unresolved[(i + 1) % len(unresolved)]["claimed_base_request"])
            for i, s in enumerate(sup) if s["emb"] is not None]
    sup_pass = all(s["extracted"] and s["jaccard"] >= MATCH_THRESHOLD for s in sup)
    assert sup_pass, [s["id"] for s in sup if not (s["extracted"] and s["jaccard"] >= MATCH_THRESHOLD)]
    sample_ids = [s["id"] for s in sorted(sup, key=lambda s: hashlib.sha256(
        ("M3PRIME_UNRESOLVED_INSPECT|" + s["id"]).encode()).hexdigest())[:10]]
    insp_path = HERE / "UNRESOLVED_RENELLM_MANUAL_INSPECTION.json"
    insp = json.loads(insp_path.read_text())
    assert insp["PASS"] and [r["id"] for r in insp["sample"]] == sample_ids, "manual inspection sample mismatch"
    assert all(r["verdict"] == "SAME_REQUEST" for r in insp["sample"] + insp["also_inspected_lowest_two"])

    rng = random.Random(BLIND_SEED)
    rng.shuffle(items)
    packet, private = [], []
    for i, it in enumerate(items, 1):
        bid = "M3PJ" + hashlib.sha256(f"{BLIND_SEED}|{it['_item_id']}".encode()).hexdigest()[:16]
        packet.append({"blinded_id": bid, "blinded_ordinal": i, "request_type": it["request_type"],
                       "base_request": it["base_request"], "domain": it["_domain"],
                       "model_response": it["model_response"], "allowed_labels": it["allowed_labels"],
                       "allowed_confidence": it["allowed_confidence"]})
        private.append({"blinded_id": bid, "blinded_ordinal": i, "item_id": it["_item_id"],
                        "group_id": it["_group_id"], "condition": it["_condition"], "attack_family": it["_family"],
                        "split": "M3PRIME_NEW_DEV", "response_sha256": it["_response_sha256"], "domain": it["_domain"]})
    assert len({p["blinded_id"] for p in packet}) == len(packet), "blinded id collision"

    allowed_keys = {"blinded_id", "blinded_ordinal", "request_type", "base_request", "domain", "model_response",
                    "allowed_labels", "allowed_confidence"}
    key_leak = sorted({k for p in packet for k in p} - allowed_keys)
    id_leak = sum(1 for p, q in zip(packet, private)
                  if q["group_id"] in (p["base_request"] + p["model_response"]) or q["item_id"] in p["model_response"])
    assert not key_leak and id_leak == 0, (key_leak, id_leak)

    pk, pm = HERE / "JUDGE_PACKET_BLINDED_622.jsonl", HERE / "JUDGE_PRIVATE_MAP_622.jsonl"
    for path, rows in ((pk, packet), (pm, private)):
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    (HERE / "JUDGE_SYSTEM_PROMPT.txt").write_bytes(PROMPT_SRC.read_bytes())

    audit = {
        "schema": "M3PRIME_NEW_DEV_JUDGE_PACKET_AUDIT_V1",
        "items": len(packet), "by_condition": dict(sorted(stats.items())),
        "generation_freeze_results_sha256": gf["results_sha256"],
        "base_request_rule": "harmful->B, benign->C request text of the NEW_DEV group; attacked prompt never used",
        "base_request_verification": {
            "renellm_clean_text_equals_dija_source_text": "330/330",
            "normalized_hash_equals_new_dev_identity_manifest": "330/330",
            "dija_rows_clean_semantic_request_equal": f"{dija_clean_checked}/318",
        },
        "scenario_match_audit": {k: v for k, v in sm.items() if k != "mismatches"},
        "scenario_match_audit_reproduces_step4_frozen": True,
        "unresolved_rows_supplementary_check": {
            "why": "all unresolved rows are table_filling with a \\multicolumn layout the frozen extractor does not "
                   "match; frozen audit module unchanged",
            "rows": len(sup), "extracted": sum(s["extracted"] for s in sup),
            "match_at_frozen_threshold": sum(1 for s in sup if s["extracted"] and s["jaccard"] >= MATCH_THRESHOLD),
            "threshold": MATCH_THRESHOLD,
            "jaccard_min": round(min(s["jaccard"] for s in sup), 4),
            "mismatched_pairing_control_max_jaccard": round(max(ctrl), 4),
            "PASS": sup_pass,
            "manual_inspection_file_sha256": sha_file(insp_path),
            "manual_inspection_sample_n": len(insp["sample"]),
            "manual_inspection_all_same_request": True,
        },
        "packet_visible_fields": sorted(allowed_keys),
        "packet_key_leak": key_leak, "packet_identity_string_leak_rows": id_leak,
        "blinding_seed": BLIND_SEED,
        "metric_contract_sha256": METRIC_SHA, "system_prompt_sha256": PROMPT_SHA,
        "label_vocabulary": LABELS,
        "packet_sha256": sha_file(pk), "private_map_sha256": sha_file(pm),
        "private_map_joined": False,
    }
    (HERE / "PACKET_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "label_vocabulary"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
