"""M3' step 8b — BUILD feature-capture byte-identity check and freeze.

SPAN RULE (tooling note, from the confirmed hashing-span diagnosis, DIAGNOSIS_RESULT.json):
the two harnesses hash DIFFERENT SPANS of the same generation.
  * stored BUILD DIJA rows hash only the tokens at the initial masked positions
    (adapter: ids = x[0, initial_mask_positions]; response = decode(ids); response_sha256 = sha256(response));
  * the M3' runtime's decoded_output comes from hist.decode_final, which decodes everything after the prefix and
    therefore includes surrounding template text.
The 40-case GPU diagnostic showed the TOKENS are identical at those positions for 20/20 DIJA rows (and 20/20
ReNeLLM rows matched directly), with identical inputs, seeds, sampler and environment. So the comparison below
uses each family's own stored span:
  DIJA    -> sha256(decode(tokens at initial masked positions))  vs stored response_sha256
  RENELLM -> sha256(decoded standard output region)              vs stored response_sha256
Comparing the M3' decoded_output directly for DIJA would flag every DIJA row as a false mismatch.

Any real mismatch is NOT repaired here: it is reported, and the contract's rule (re-judge -> versioned label set ->
recompute the three pooled support counts -> stop if any < 20) applies, which needs an explicit go.
"""
import hashlib, importlib.util, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV = ROOT / "analysis_output/rrae_development"
HERE = Path(__file__).resolve().parent
CASES = DEV / "generalized_safety_v2_m3_trajectory_features_v1/CASES.jsonl"
NEWDEV = DEV / "generalized_safety_v2_m3prime_llada_baseline_v1"
NSHARDS = 4
FEATURE_KEYS = ("f1_proj_ALL", "f2_proj_DIJA", "f3_proj_REN", "f4_dproj_ALL", "f5_dproj_DIJA", "f6_dproj_REN",
                "f7_norm", "f8_denoise_frac", "f9_mask_frac")


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    fails, rows = [], []
    for s in range(NSHARDS):
        jp = HERE / f"BUILD_CAPTURE_JOURNAL_shard{s:02d}.jsonl"
        rp = HERE / f"BUILD_CAPTURE_RESULTS_shard{s:02d}.jsonl"
        J = jl(jp)
        ev = Counter(e["event"] for e in J)
        if ev.get("GENERATION_COMPLETE", 0) != 1:
            fails.append(f"shard {s}: no single GENERATION_COMPLETE")
        for bad in ("CATEGORY_C", "CATEGORY_D_OOM"):
            if ev.get(bad, 0):
                fails.append(f"shard {s}: {ev[bad]} {bad} events")
        rows += jl(rp)

    # input_helper_route is not carried on the runtime result, so it is recomputed here from the SAME
    # deterministic item builder the capture used (CPU only, no model, no generation).
    spec = importlib.util.spec_from_file_location("m3p_capture", HERE / "06_build_capture.py")
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    route_by_item = {it["item_id"]: it["input_helper_route"] for it in cap.build_items(cap.G.Builder())}
    route_stats = dict(cap.ROUTE_STATS)

    build = {c["generation_id"]: c for c in jl(CASES) if c["split"] == "BUILD"}
    if len(rows) != 1222 or len({r["item_id"] for r in rows}) != 1222:
        fails.append(f"expected 1222 unique captured rows, got {len(rows)} / {len({r['item_id'] for r in rows})}")
    if {r["build_generation_id"] for r in rows} != set(build):
        fails.append("captured generation_id set != BUILD case set")

    mism, checks = [], Counter()
    for r in rows:
        c = build.get(r["build_generation_id"])
        if c is None:
            continue
        fam = r["attack_family"]
        checks[f"span_{r['span_rule']}"] += 1
        ok_span = r["span_text_sha256"] == c["response_sha256"] == r["stored_response_sha256"]
        ok_steps = int(r["trajectory_steps"]) == int(c["trajectory_step_count"])
        ok_seed = int(r["generation_seed"]) == int(c["generation_seed"])
        ok_feats = len(r["trajectory_features"]) == int(r["trajectory_steps"])
        ok_base = r["vector_key"] is None and int(r["application_count"]) == 0
        nonempty = [f for f in r["trajectory_features"] if not f.get("empty_mask")]
        ok_keys = all(all(k in f for k in FEATURE_KEYS) for f in nonempty) and len(nonempty) >= 1
        ok_first = sum(1 for f in nonempty if f.get("first_eligible_step")) == 1
        for name, ok in (("span_text", ok_span), ("steps", ok_steps), ("seed", ok_seed),
                         ("features_one_per_step", ok_feats), ("baseline_no_steering", ok_base),
                         ("feature_keys", ok_keys), ("single_first_eligible_step", ok_first)):
            if ok:
                checks[f"pass_{name}"] += 1
            else:
                checks[f"FAIL_{name}"] += 1
                if name == "span_text":
                    mism.append({"generation_id": r["build_generation_id"], "family": fam,
                                 "v2_condition": r["v2_condition"], "route": r.get("input_helper_route")})
                else:
                    fails.append(f"{r['build_generation_id']}: {name} failed")

    n = len(rows) or 1
    report = {
        "schema": "GENERALIZED_SAFETY_V2_M3PRIME_BUILD_FEATURE_BYTE_IDENTITY_V1",
        "contract_item": "M3prime checklist item 9 (BUILD trajectory features frozen; byte-identity report)",
        "observer": "M3' NewDevFeatureObserver (same code path that produced the NEW_DEV features; the M3 "
                    "extract_trajectory_features_v1 path was NOT used)",
        "span_rule": {
            "DIJA": "sha256(decode(tokens at initial masked positions)) vs stored response_sha256",
            "RENELLM": "sha256(decoded standard output region) vs stored response_sha256",
            "why": "stored BUILD rows hash only the masked-position span; the M3' decoded_output field spans more "
                   "text (hist.decode_final decodes everything after the prefix). Same tokens, different span — "
                   "confirmed token-identical on the 40-case GPU diagnostic (20/20 DIJA, 20/20 ReNeLLM).",
            "tooling_note": "Comparing M3' decoded_output directly against stored response_sha256 for DIJA rows "
                            "would report a false mismatch on every DIJA row. Any future comparison must use the "
                            "per-family span above.",
        },
        "dija_route_rule": "M1 pair-level rule; the 10 BUILD pairs whose B_DIJA was M1-recovered use the frozen "
                           "compat helper (20 rows). The both-helpers-agree cross-check is unevaluable on those 10 "
                           "B rows (the source helper raises on exactly them) and is recorded as skipped.",
        "rows_captured": len(rows),
        "mismatch_count": len(mism),
        "mismatch_rate": round(len(mism) / n, 6),
        "mismatches": mism[:50],
        "checks": dict(checks),
        "by_condition": dict(Counter(r["v2_condition"] for r in rows)),
        "by_route": dict(Counter(route_by_item[r["item_id"]] for r in rows)),
        "route_resolution": dict(route_stats,
                                 source="recomputed at freeze time by 06_build_capture.build_items (deterministic, "
                                        "CPU only); the runtime result dict does not carry input_helper_route"),
        "total_feature_steps": sum(len(r["trajectory_features"]) for r in rows),
        "shard_results_sha256": {f"shard{s:02d}": sha_file(HERE / f"BUILD_CAPTURE_RESULTS_shard{s:02d}.jsonl")
                                 for s in range(NSHARDS)},
        "shard_journal_sha256": {f"shard{s:02d}": sha_file(HERE / f"BUILD_CAPTURE_JOURNAL_shard{s:02d}.jsonl")
                                 for s in range(NSHARDS)},
        "capture_script_sha256": sha_file(HERE / "06_build_capture.py"),
        "newdev_generation_freeze_sha256": sha_file(NEWDEV / "GENERATION_FREEZE.json"),
        "structural_failures": fails,
        "supersedes": {"sha256": "b148756d1f069fed2e52f5ae79c7077747157f43cf4755c2f74a227e0b7e8178",
                       "why": "first freeze reported by_route as null for every row (missing field, not a missing "
                              "check). Only that reporting field changed; every check, count and shard SHA is "
                              "identical. BUILD_GATE_FIT.json cites the superseded SHA because it was produced "
                              "before this patch; the fit consumed the shard result files, whose SHAs are "
                              "unchanged here, so the lineage is intact."},
    }
    report["status"] = "PASS_FROZEN" if not fails and not mism else (
        "FAILED_STRUCTURAL" if fails else "MISMATCH_REQUIRES_CONTRACT_REJUDGE_RULE")
    (HERE / "BUILD_FEATURE_FREEZE.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("mismatches", "span_rule")},
                     indent=2, sort_keys=True))
    if report["status"] != "PASS_FROZEN":
        print("\nSTOP: not PASS_FROZEN — no re-judging is performed here; the contract rule needs an explicit go.",
              file=sys.stderr)
    sys.exit(0 if report["status"] == "PASS_FROZEN" else 8)


if __name__ == "__main__":
    main()
