#!/usr/bin/env python3
"""M3' step 5 freeze: NEW_DEV baseline generation outputs + trajectory features, frozen together.

Validates GENERATION_RESULTS.jsonl against the frozen 622-row manifest and the generation journal, and records
the execution plan actually used (single job or sequential time-budgeted segments), the generation script SHA,
the non-perturbation test result SHA, and (for a split run) the journal-resume exactness test result SHA.

Reads outputs only as hashes/structural fields; no label, no judgment, no outcome is computed here.
"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "M3PRIME_LLADA_BASELINE_MANIFEST_622.jsonl"
MANIFEST_SHA = "449e2887b19e79e0a56019f055c83cb23fe559b31160cb8bc2be2f9d6a421d6a"
RESULTS = HERE / "GENERATION_RESULTS.jsonl"
JOURNAL = HERE / "GENERATION_JOURNAL.jsonl"
SCRIPT = HERE / "m3prime_llada_generate.py"
NONPERT = HERE / "NONPERT_TEST_RESULT.json"
RESUME = HERE / "resume_test/RESUME_TEST_RESULT.json"


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execution-plan", choices=["SPLIT_2x45MIN_SEQUENTIAL", "SINGLE_1H30_JOB"], required=True)
    ap.add_argument("--expect-script-sha", required=True)
    ap.add_argument("--expect-nonpert-sha", required=True)
    a = ap.parse_args()

    fails = []

    def req(ok, msg):
        if not ok:
            fails.append(msg)

    req(sha_file(MANIFEST) == MANIFEST_SHA, "manifest SHA mismatch")
    script_sha = sha_file(SCRIPT)
    req(script_sha.startswith(a.expect_script_sha), f"generation script SHA {script_sha} != expected")
    nonpert_sha = sha_file(NONPERT)
    req(nonpert_sha.startswith(a.expect_nonpert_sha), f"nonpert result SHA {nonpert_sha} != expected")
    req(json.loads(NONPERT.read_text())["PASS"] is True, "nonpert result not PASS")
    resume = None
    if a.execution_plan.startswith("SPLIT"):
        req(RESUME.exists(), "split plan requires the resume-exactness test result")
        if RESUME.exists():
            rr = json.loads(RESUME.read_text())
            req(rr["PASS"] is True, "resume-exactness test not PASS")
            resume = {"path": str(RESUME.relative_to(HERE)), "sha256": sha_file(RESUME), "PASS": rr["PASS"],
                      "n_rows": rr["n_rows"], "killed_mid_row": rr["killed_mid_row"],
                      "rows_generated_after_restart": rr["rows_generated_after_restart"]}

    man = jl(MANIFEST)
    res = jl(RESULTS)
    J = jl(JOURNAL)
    mids = [r["item_id"] for r in man]
    rids = [r["item_id"] for r in res]
    req(len(res) == 622, f"results rows {len(res)} != 622")
    req(len(set(rids)) == len(rids), "duplicate item_id in results")
    req(set(rids) == set(mids), "results item_id set != manifest item_id set")

    ev = Counter(e["event"] for e in J)
    req(ev.get("GENERATION_COMPLETE", 0) == 1, "journal lacks exactly one GENERATION_COMPLETE")
    for bad in ("CATEGORY_C", "CATEGORY_D_OOM"):
        req(ev.get(bad, 0) == 0, f"journal has {bad} events")
    ok_ids = [e["item_id"] for e in J if e["event"] == "OK"]
    req(len(ok_ids) == len(set(ok_ids)) == 622 and set(ok_ids) == set(mids), "journal OK set != manifest")
    ok_sha = {e["item_id"]: e["final_ids_sha256"] for e in J if e["event"] == "OK"}

    M = {r["item_id"]: r for r in man}
    per_row_fail = Counter()
    for r in res:
        m = M.get(r["item_id"])
        if m is None:
            continue
        checks = {
            "seed": int(r["generation_seed"]) == int(m["generation_seed"]),
            "initial_ids": r["initial_ids_sha256"] == m["expected_initial_ids_sha256"],
            "steps": int(r["trajectory_steps"]) == int(m["expected_trajectory_steps"]),
            "prompt_sha": r["prompt_sha256"] == m["prompt_sha256"],
            "baseline_no_steering": r["vector_key"] is None and int(r["application_count"]) == 0,
            "decoded_sha": hashlib.sha256(r["decoded_output"].encode("utf-8")).hexdigest()
                           == r["decoded_output_sha256"],
            "journal_final_ids": ok_sha.get(r["item_id"]) == r["final_ids_sha256"],
            "features_one_per_step": len(r["trajectory_features"]) == int(r["trajectory_steps"]),
            "features_nonempty": r["trajectory_features_nonempty_steps"] >= 1,
        }
        for k, v in checks.items():
            if not v:
                per_row_fail[k] += 1
    req(not per_row_fail, f"per-row check failures: {dict(per_row_fail)}")

    segments = []
    for e in J:
        if e["event"] == "SEGMENT_START":
            segments.append({"slurm_job_id": e["slurm_job_id"], "done_at_start": e["done_at_start"],
                             "orphans_at_start": e["orphans_at_start"], "max_seconds": e["max_seconds"]})
        elif e["event"] == "SEGMENT_END" and segments:
            segments[-1]["ended"] = "SEGMENT_END_TIME_BUDGET"
            segments[-1]["done_at_end"] = e["done"]
    if segments:
        segments[-1].setdefault("ended", "GENERATION_COMPLETE")
    attempts = Counter(e["attempt"] for e in J if e["event"] == "OK")
    reconciled = sum(1 for e in J if e["event"] == "OK" and e.get("reconciled_orphan"))

    status = "PASS_FROZEN" if not fails else "FAILED"
    freeze = {
        "schema": "GENERALIZED_SAFETY_V2_M3PRIME_NEW_DEV_BASELINE_GENERATION_FREEZE_V1",
        "contract_item": "M3prime checklist item 6 (nonpert PASS; NEW_DEV baseline generation + trajectory "
                         "features frozen together)",
        "status": status,
        "failures": fails,
        "rows": len(res),
        "by_condition": dict(Counter(r["v2_condition"] for r in res)),
        "manifest_sha256": MANIFEST_SHA,
        "results_sha256": sha_file(RESULTS),
        "journal_sha256": sha_file(JOURNAL),
        "generation_script_sha256": script_sha,
        "nonpert_test_result_sha256": nonpert_sha,
        "execution_detail": {
            "plan": a.execution_plan,
            "max_seconds_per_segment": 2400 if a.execution_plan.startswith("SPLIT") else None,
            "segments_observed": segments,
            "ok_attempt_histogram": dict(attempts),
            "orphan_rows_reconciled": reconciled,
            "resume_exactness_test": resume,
            "note": ("Execution detail only: each row's output depends only on its frozen seed and inputs "
                     "(demonstrated byte-exact across a process restart by the resume test); segmenting "
                     "changes no generation setting." if a.execution_plan.startswith("SPLIT")
                     else "Single continuous job."),
        },
        "judging_started": False,
        "labels_seen": False,
    }
    (HERE / "GENERATION_FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(freeze, indent=2, sort_keys=True))
    sys.exit(0 if status == "PASS_FROZEN" else 1)


if __name__ == "__main__":
    main()
