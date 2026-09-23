#!/usr/bin/env python3
"""M2 confirmatory LLaDA generation - one shard (SLURM_ARRAY_TASK_ID). PREPARED ONLY; not submitted
without explicit authorization.

Hard-failure policy (frozen, efe2e4a8...; terminology clarification a6ee0d64...):
  A infrastructure/transient : a row whose ATTEMPT_START has no matching OK is retried with the exact same
                               case/arm/seed/settings on the next launch; max 2 retries (3 attempts), then
                               escalated to category C. Every attempt is journaled.
  B weird/empty text          : retained as generated; never regenerated.
  C scientific/runtime        : NaN/Inf, hook invariant violation, M2_NUMERICAL_DENOMINATOR_INVALID,
                               M2_NORM_ANOMALY_SENTINEL -> journal CATEGORY_C, stop the shard (exit 3).
                               A shard with any CATEGORY_C in its journal refuses to run.
  D reproducible OOM          : not handled automatically - stops; infrastructure amendment required.
No seed reroll, no arm-specific removal, no outcome inspection."""
import json, os, sys, time, traceback
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m2_llada_runtime_v1 as RT  # noqa: E402

MAN = HERE / "M2_LLADA_CONFIRMATORY_MANIFEST_PRIVATE_3038.jsonl"
EXPECT_MAN_SHA = "840eb0778e58f6d634c5bbfe650b6b5b8974aa4340ed0af241c34ff5384f9ac7"
OUT_ROOT = RT.DEV / "generalized_safety_v2_m2_llada_confirmatory_generation_v1"
GATE = HERE / "CONFIRMATORY_SUBMISSION_GATE.json"
MAX_ATTEMPTS = 3


def jl(p):
    return [json.loads(x) for x in open(p)] if Path(p).exists() else []


def append(p, rec):
    with open(p, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())


def main():
    shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    gate = json.loads(GATE.read_text())
    if gate.get("status") != "PASS" or gate.get("confirmatory_generation_authorized") is not True:
        raise SystemExit("BLOCKED: confirmatory submission gate not PASS/authorized")
    assert RT.sha_file(MAN) == EXPECT_MAN_SHA == gate["manifest_sha256"]
    rows = [r for r in jl(MAN) if int(r["shard_index"]) == shard]
    assert rows and all(r["split"] == "M2_HELDOUT_CONFIRMATORY_EVALUATION" for r in rows)
    out = OUT_ROOT / f"shard_{shard:02d}"
    out.mkdir(parents=True, exist_ok=True)
    journal, results = out / "JOURNAL.jsonl", out / "SHARD_RESULTS_PRIVATE.jsonl"
    J = jl(journal)
    if any(e["event"] == "CATEGORY_C" for e in J):
        raise SystemExit("BLOCKED: shard has a CATEGORY_C event; investigation + authorization required")
    done = {e["item_id"] for e in J if e["event"] == "OK"}
    starts = {}
    for e in J:
        if e["event"] == "ATTEMPT_START":
            starts[e["item_id"]] = starts.get(e["item_id"], 0) + 1
    job = os.environ.get("SLURM_JOB_ID")
    append(journal, {"event": "LAUNCH", "shard": shard, "slurm_job_id": job, "t": time.time(), "done": len(done)})
    rt = RT.build_runtime()
    for r in rows:
        if r["method"] != "BASELINE" and r["beta"] != rt["betas"][r["method"]][r["direction"]]:
            raise SystemExit("BLOCKED: manifest beta != beta loaded through frozen mapping")
    for r in rows:
        iid = r["item_id"]
        if iid in done:
            continue
        attempt = starts.get(iid, 0) + 1
        if attempt > MAX_ATTEMPTS:
            append(journal, {"event": "CATEGORY_C", "item_id": iid, "label": "INFRASTRUCTURE_RETRIES_EXHAUSTED",
                             "attempts": attempt - 1, "slurm_job_id": job})
            sys.exit(3)
        append(journal, {"event": "ATTEMPT_START", "item_id": iid, "attempt": attempt, "arm": r["arm"],
                         "seed": r["generation_seed"], "slurm_job_id": job, "retry_category": "A" if attempt > 1 else None})
        try:
            res = RT.run_item(rt, r, variant="WRAPPED")
        except RT.CategoryCFailure as e:
            append(journal, {"event": "CATEGORY_C", "item_id": iid, "label": e.label, "detail": e.detail,
                             "arm": r["arm"], "slurm_job_id": job})
            print("CATEGORY_C_FAIL_CLOSED", iid, e.label, e.detail, flush=True)
            sys.exit(3)
        except RT.torch.cuda.OutOfMemoryError as e:  # category D: stop; relaunch retries same row; reproducible -> amendment
            append(journal, {"event": "CATEGORY_D_OOM", "item_id": iid, "attempt": attempt, "detail": str(e)[:500],
                             "arm": r["arm"], "slurm_job_id": job})
            sys.exit(4)
        except Exception as e:  # any other in-generation exception is treated as scientific (fail closed)
            append(journal, {"event": "CATEGORY_C", "item_id": iid, "label": "UNCLASSIFIED_RUNTIME_EXCEPTION",
                             "detail": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:],
                             "arm": r["arm"], "slurm_job_id": job})
            sys.exit(3)
        res["attempt"] = attempt
        res["slurm_job_id"] = job
        append(results, res)
        append(journal, {"event": "OK", "item_id": iid, "attempt": attempt, "final_ids_sha256": res["final_ids_sha256"],
                         "slurm_job_id": job})
        print(f"OK {iid} steps={res['trajectory_steps']}", flush=True)
    append(journal, {"event": "SHARD_COMPLETE", "shard": shard, "slurm_job_id": job, "t": time.time()})


if __name__ == "__main__":
    main()
