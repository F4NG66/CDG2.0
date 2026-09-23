#!/usr/bin/env python3
"""M3' step 5b/c: (mode=nonpert) 40-case BUILD non-perturbation test for the
NEW trajectory-feature observer, then (mode=full) NEW_DEV BASELINE generation
with that observer attached, trajectory features + decoded output frozen
together per item.

Reuses UNCHANGED: m2_llada_runtime_v1.build_runtime/run_item/reset_all_banks
(model load, hook banks, DIJA/RENELLM input routing, seed policy, hard-failure
categories A/B/C/D pattern from run_m2_llada_confirmatory_shard_v1.py);
build_manifests_v1.Builder.route_and_input (DIJA route resolution).

NEW (this script only): NewDevFeatureObserver -- composes the SAME
self-sufficient hook pattern as m2_llada_runtime_v1.L16Observer (its own
model-level forward-pre-hook for step/mask, so it needs no cooperation from
the black-box generation loop) with the SAME per-step feature math as
generalized_safety_v2_m3_trajectory_features_v1.L16FeatureObserver (9 gate
features, byte-identical formulas). Registered as an ADDITIONAL read-only
forward hook on blocks[LAYER], alongside (not replacing) the runtime's own
L16Observer -- both return None, so activations are provably unaltered by
either (the runtime's own observer already fail-closed-asserts this for the
BASELINE arm).
"""
import argparse, hashlib, json, math, os, sys, time, traceback
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV = ROOT / "analysis_output/rrae_development"
M2EXE = DEV / "generalized_safety_v2_m2_llada_executor_v1"
sys.path.insert(0, str(M2EXE))
import m2_llada_runtime_v1 as RT  # noqa: E402
from build_manifests_v1 import Builder, seed_rule, H  # noqa: E402
import torch  # noqa: E402

OUT = DEV / "generalized_safety_v2_m3prime_llada_baseline_v1"
CASES = DEV / "generalized_safety_v2_m3_trajectory_features_v1/CASES.jsonl"
CANON = ROOT / "data/abcd/ABCD_clean_v2_DIJA_Qwen_A_B_C_D.jsonl"
MANIFEST = OUT / "M3PRIME_LLADA_BASELINE_MANIFEST_622.jsonl"

MASK_ID = RT.MASK_ID
LAYER = RT.LAYER


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def append(p, rec):
    with open(p, "a") as f:
        f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def output_to_hidden(o):
    h = o[0] if isinstance(o, (tuple, list)) else o
    assert torch.is_tensor(h)
    return h


class NewDevFeatureObserver:
    """Self-sufficient (own model-level pre-hook for step/mask), 9-feature
    math identical to L16FeatureObserver. enabled=False -> both hooks no-op."""

    def __init__(self, model, block, v_all, v_dija, v_ren):
        self.v = {"ALL": v_all, "DIJA": v_dija, "REN": v_ren}
        self.enabled = True
        self._h_pre = model.register_forward_pre_hook(self._model_pre)
        self._h_blk = block.register_forward_hook(self._block_hook)
        self.reset(0)

    def reset(self, trajectory_steps):
        self.T = int(trajectory_steps)
        self.step = 0
        self.mask = None
        self.total_tokens = None
        self.prev = None
        self.rows = []

    def close(self):
        self._h_pre.remove()
        self._h_blk.remove()

    def _model_pre(self, _module, args):
        if not self.enabled:
            return None
        self.step += 1
        self.mask = (args[0] == MASK_ID).detach().clone()
        self.total_tokens = int(args[0].numel())
        return None

    def _block_hook(self, _module, _inputs, output):
        if not self.enabled:
            return None
        hidden = output_to_hidden(output)
        tgt = self.mask.to(hidden.device, torch.bool)
        n = int(tgt.sum().item())
        if n == 0:
            self.rows.append({"step": self.step, "empty_mask": True})
            return None
        m_t = hidden.float()[tgt].mean(dim=0)
        dev = m_t.device
        p = {k: float(torch.dot(m_t, v.to(dev))) for k, v in self.v.items()}
        first = self.prev is None
        d = {k: (0.0 if first else p[k] - self.prev[k]) for k in p}
        self.prev = p
        self.rows.append({
            "step": self.step, "empty_mask": False,
            "f1_proj_ALL": p["ALL"], "f2_proj_DIJA": p["DIJA"], "f3_proj_REN": p["REN"],
            "f4_dproj_ALL": d["ALL"], "f5_dproj_DIJA": d["DIJA"], "f6_dproj_REN": d["REN"],
            "f7_norm": float(torch.linalg.vector_norm(m_t)),
            "f8_denoise_frac": self.step / self.T if self.T else None,
            "f9_mask_frac": n / self.total_tokens,
            "first_eligible_step": first, "n_target": n,
        })
        return None


def load_feature_directions():
    dirs = torch.load(DEV / "generalized_safety_v2_directions_v1/GENERALIZED_SAFETY_V2_DIRECTIONS.pt",
                      map_location="cpu", weights_only=True)
    sha = {
        "v_ALL": "937a78750761e63ebc1362bd43e7b3adfa20790b6cec9ae15df01a7befe177e7",
        "v_DIJA": "2a52b08871548bdcb951ea5a32595fc30d7fbc4d057e51a0f3f9c39d8140c6b5",
        "v_RENELLM": "27cd930d1633cd84d1fb8c33b98f16a6dd86220cadf4d5fcd90c404cb9f7be33",
    }
    for k, want in sha.items():
        got = hashlib.sha256(dirs[k].float().contiguous().numpy().tobytes()).hexdigest()
        assert got == want, f"direction {k} SHA mismatch"
    return dirs["v_ALL"].float().view(-1), dirs["v_DIJA"].float().view(-1), dirs["v_RENELLM"].float().view(-1)


def build_case_item(B, family, cond, prompt, pair_id, route_B_prompt=None, route_B_clean=None, clean=None,
                    id_prefix="nonpert"):
    sc = {"family": family, "condition": cond, "v2_condition": f"{cond}_{family}", "group_key": f"PAIR_{pair_id:04d}",
         "prompt": prompt, "clean": clean, "pair_id": pair_id, "seed": seed_rule(pair_id, cond),
         "domain": None, "source_case_id": f"PAIR_{pair_id:04d}|{family}|{cond}", "id_prefix": id_prefix}
    if family == "DIJA":
        sc["route_B_prompt"], sc["route_B_clean"] = route_B_prompt, route_B_clean
    route, x, meta = B.route_and_input(sc)
    init_count = int((x == RT.MASK_ID).sum())
    steps = init_count if family == "DIJA" else 128
    return {
        "item_id": f"{id_prefix}_{sc['group_key']}_{sc['v2_condition']}_BASELINE", "split": "NONPERT_BUILD_SAMPLE",
        "arm": "BASELINE", "method": "BASELINE", "direction": None, "beta": 0.0, "beta_source": None,
        "attack_family": family, "condition": cond, "v2_condition": sc["v2_condition"], "group_key": sc["group_key"],
        "group_ordinal": pair_id, "pair_id": pair_id, "source_case_id": sc["source_case_id"], "domain": None,
        "prompt": prompt, "clean_semantic_request": sc.get("clean"), "generation_seed": sc["seed"],
        "input_helper_route": route,
        "expected_initial_ids_sha256": B.m1.tensor_sha256(x), "expected_initial_fillable_mask_count": init_count,
        "expected_trajectory_steps": steps,
        # metadata fields m1.generate_one echoes into its result; same values as build_manifests_v1 BASELINE rows
        "steering_condition": "BASELINE", "vector_key": None, "prompt_sha256": H(prompt),
        "runtime_mode": meta["generation_mode"],
        "layer_zero_based": 16, "token_scope": "current_mask", "scope": "current_mask", "schedule": "persistent",
        "gen_length": 128, "temperature": 0.2, "remasking": "low_confidence",
    }


def select_nonpert_cases(B):
    cases = jl(CASES)
    build = [c for c in cases if c["split"] == "BUILD"]
    canon = {r["id"]: r for r in jl(CANON)}
    strata = {"B_DIJA": [], "C_DIJA": [], "B_RENELLM": [], "C_RENELLM": []}
    for c in build:
        strata[c["condition"]].append(c)
    for k in strata:
        strata[k].sort(key=lambda c: c["pair_id"])
    items = []
    for cond_key, family, cond in (("B_DIJA", "DIJA", "B"), ("C_DIJA", "DIJA", "C"),
                                   ("B_RENELLM", "RENELLM", "B"), ("C_RENELLM", "RENELLM", "C")):
        picked = strata[cond_key][:10]
        assert len(picked) == 10, f"only {len(picked)} BUILD cases for {cond_key}"
        for c in picked:
            pid = c["pair_id"]
            if family == "DIJA":
                brow = canon[f"B_{pid:04d}"]
                own_clean = canon[f"{cond}_{pid:04d}"]["clean_prompt"]
                item = build_case_item(B, family, cond, c["prompt"], pid,
                                       route_B_prompt=brow["prompt"], route_B_clean=brow["clean_prompt"],
                                       clean=own_clean)
            else:
                item = build_case_item(B, family, cond, c["prompt"], pid)
            items.append(item)
    assert len(items) == 40
    return items


def run_one(rt, feat_obs, item, feat_enabled):
    feat_obs.enabled = feat_enabled
    feat_obs.reset(item["expected_trajectory_steps"])
    res = RT.run_item(rt, item, variant="WRAPPED")
    return res, list(feat_obs.rows)


def do_nonpert(rt, feat_obs):
    B = Builder()
    items = select_nonpert_cases(B)
    out_path = OUT / "NONPERT_TEST_RESULT.json"
    mismatches = []
    per_stratum = {}
    per_item = []
    for it in items:
        res_on, rows_on = run_one(rt, feat_obs, it, True)
        res_off, rows_off = run_one(rt, feat_obs, it, False)
        # byte identity on final token ids AND decoded text; seed/initial ids/steps must also agree across arms
        cmp_keys = ("final_ids_sha256", "decoded_output_sha256", "initial_ids_sha256", "trajectory_steps",
                    "generation_seed")
        diff = [k for k in cmp_keys if res_on[k] != res_off[k]]
        match = not diff
        nonempty_on = sum(1 for r in rows_on if not r.get("empty_mask"))
        key = it["v2_condition"]
        per_stratum.setdefault(key, {"n": 0, "match": 0})
        per_stratum[key]["n"] += 1
        per_stratum[key]["match"] += int(match)
        per_item.append({"item_id": it["item_id"], "match": match, "differing_fields": diff,
                         "generation_seed": it["generation_seed"], "prompt_sha256": it["prompt_sha256"],
                         "on": {k: res_on[k] for k in cmp_keys}, "off": {k: res_off[k] for k in cmp_keys},
                         "feature_rows_on": nonempty_on, "feature_rows_off": len(rows_off)})
        if not match:
            mismatches.append({"item_id": it["item_id"], "differing_fields": diff})
        print(f"{it['item_id']}: on={res_on['final_ids_sha256'][:12]} off={res_off['final_ids_sha256'][:12]} "
             f"match={match} feature_rows_on={nonempty_on} feature_rows_off={len(rows_off)}", flush=True)
    # the observer must actually have been live in the on-arm and silent in the off-arm
    observer_live = all(p["feature_rows_on"] > 0 for p in per_item) and all(p["feature_rows_off"] == 0 for p in per_item)
    m1 = rt["H"]["m1"]
    result = {"schema": "GENERALIZED_SAFETY_V2_M3PRIME_OBSERVER_NONPERT_TEST_V1",
             "n_cases": len(items), "per_stratum": per_stratum, "mismatches": mismatches,
             "observer_live_on_silent_off": observer_live,
             "generation_constants_both_arms": {"GEN_LENGTH": m1.GEN_LENGTH, "TEMPERATURE": m1.TEMPERATURE,
                                                "REMASK": m1.REMASK, "runner_sha256": RT.sha_file(RT.P["runner"])},
             "arm_order": "per item: observer-on then observer-off, same process, same loaded model",
             "per_item": per_item,
             "PASS": len(mismatches) == 0 and observer_live,
             "seed": "deterministic: BUILD cases sorted by pair_id, first 10/stratum"}
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return result["PASS"]


RESUME_CMP_KEYS = ("final_ids_sha256", "decoded_output_sha256", "initial_ids_sha256", "trajectory_steps",
                   "generation_seed")


def do_full(rt, feat_obs, rows=None, journal=None, results=None, max_seconds=None, _kill_after_rows=None):
    """Resumable generation. Journal OK events define `done`. max_seconds: stop cleanly BETWEEN rows (event
    SEGMENT_END, exit 0) once too little budget remains for another row, so a time-limited segment never dies
    mid-row. A result row written without its OK (kill between the two appends) is an orphan: on resume it is
    regenerated, required to be byte-identical to the orphan, and journaled OK without a duplicate append.
    _kill_after_rows: resume_test only -- after N rows, journal ATTEMPT_START for the next row and hard-exit,
    imitating a kill mid-row."""
    if rows is None:
        rows = jl(MANIFEST)
        assert len(rows) == 622
    journal = journal or OUT / "GENERATION_JOURNAL.jsonl"
    results = results or OUT / "GENERATION_RESULTS.jsonl"
    J = jl(journal) if journal.exists() else []
    if any(e["event"] == "CATEGORY_C" for e in J):
        raise SystemExit("BLOCKED: a CATEGORY_C event exists; investigation required")
    done = {e["item_id"] for e in J if e["event"] == "OK"}
    written = jl(results) if results.exists() else []
    ids_written = [r["item_id"] for r in written]
    if len(ids_written) != len(set(ids_written)):
        raise SystemExit("BLOCKED: duplicate item_id in results file")
    orphans = {r["item_id"]: r for r in written if r["item_id"] not in done}
    if len(orphans) > 1:
        raise SystemExit(f"BLOCKED: {len(orphans)} orphan result rows (expected at most 1); investigation required")
    if set(done) - set(ids_written):
        raise SystemExit("BLOCKED: journal OK without a result row")
    starts = {}
    for e in J:
        if e["event"] == "ATTEMPT_START":
            starts[e["item_id"]] = starts.get(e["item_id"], 0) + 1
    job = os.environ.get("SLURM_JOB_ID")
    append(journal, {"event": "SEGMENT_START", "slurm_job_id": job, "done_at_start": len(done),
                     "orphans_at_start": sorted(orphans), "max_seconds": max_seconds, "t": time.time()})
    feat_obs.enabled = True
    t0 = time.time()
    n_done_this_run = 0
    slowest = 0.0
    for r in rows:
        iid = r["item_id"]
        if iid in done:
            continue
        if max_seconds is not None:
            need = max(60.0, 3 * slowest)
            if time.time() - t0 + need > max_seconds:
                append(journal, {"event": "SEGMENT_END", "reason": "TIME_BUDGET", "slurm_job_id": job,
                                 "done": len(done) + n_done_this_run, "t": time.time()})
                print(f"SEGMENT_END: budget reached after {n_done_this_run} rows this segment", flush=True)
                return "SEGMENT_END"
        attempt = starts.get(iid, 0) + 1
        if attempt > 3:
            append(journal, {"event": "CATEGORY_C", "item_id": iid, "label": "INFRASTRUCTURE_RETRIES_EXHAUSTED",
                             "attempts": attempt - 1, "slurm_job_id": job})
            sys.exit(3)
        append(journal, {"event": "ATTEMPT_START", "item_id": iid, "attempt": attempt, "seed": r["generation_seed"],
                         "slurm_job_id": job})
        if _kill_after_rows is not None and n_done_this_run >= _kill_after_rows:
            print(f"RESUME_TEST: simulated kill mid-row at {iid}", flush=True)
            os._exit(9)
        t_row = time.time()
        try:
            feat_obs.reset(r["expected_trajectory_steps"])
            res = RT.run_item(rt, r, variant="WRAPPED")
            feat_rows = list(feat_obs.rows)
        except RT.CategoryCFailure as e:
            append(journal, {"event": "CATEGORY_C", "item_id": iid, "label": e.label, "detail": e.detail,
                             "slurm_job_id": job})
            print("CATEGORY_C_FAIL_CLOSED", iid, e.label, e.detail, flush=True)
            sys.exit(3)
        except RT.torch.cuda.OutOfMemoryError as e:
            append(journal, {"event": "CATEGORY_D_OOM", "item_id": iid, "attempt": attempt, "detail": str(e)[:500],
                             "slurm_job_id": job})
            sys.exit(4)
        except Exception as e:
            append(journal, {"event": "CATEGORY_C", "item_id": iid, "label": "UNCLASSIFIED_RUNTIME_EXCEPTION",
                             "detail": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:],
                             "slurm_job_id": job})
            sys.exit(3)
        slowest = max(slowest, time.time() - t_row)
        res["attempt"] = attempt
        res["slurm_job_id"] = job
        res["trajectory_features"] = feat_rows
        res["trajectory_features_nonempty_steps"] = sum(1 for x in feat_rows if not x.get("empty_mask"))
        if iid in orphans:
            o = orphans[iid]
            if any(o[k] != res[k] for k in RESUME_CMP_KEYS) or o["trajectory_features"] != feat_rows:
                append(journal, {"event": "CATEGORY_C", "item_id": iid, "label": "ORPHAN_REGENERATION_MISMATCH",
                                 "slurm_job_id": job})
                sys.exit(3)
            append(journal, {"event": "OK", "item_id": iid, "attempt": attempt, "reconciled_orphan": True,
                             "final_ids_sha256": res["final_ids_sha256"], "slurm_job_id": job})
        else:
            append(results, res)
            append(journal, {"event": "OK", "item_id": iid, "attempt": attempt,
                             "final_ids_sha256": res["final_ids_sha256"], "slurm_job_id": job})
        n_done_this_run += 1
        if n_done_this_run % 25 == 0:
            el = time.time() - t0
            print(f"progress: {len(done)+n_done_this_run}/{len(rows)}  ({el/n_done_this_run:.2f}s/item this run)",
                  flush=True)
    append(journal, {"event": "GENERATION_COMPLETE", "slurm_job_id": job, "t": time.time()})
    print("GENERATION_COMPLETE", flush=True)
    return "COMPLETE"


# ------------------------------------------------------------ resume test (BUILD cases only; never NEW_DEV rows)
RT_DIR = OUT / "resume_test"


def resume_test_rows(B):
    """12 BUILD cases from the nonpert sample, 3 per stratum, in family order so the simulated kill (after 5
    rows, i.e. mid-row on the 6th = C_DIJA) is followed by a resume that crosses the DIJA -> RENELLM boundary."""
    items = select_nonpert_cases(B)
    rows = []
    for cond in ("B_DIJA", "C_DIJA", "B_RENELLM", "C_RENELLM"):
        rows += [dict(it, item_id=it["item_id"].replace("nonpert_", "resumetest_"), split="RESUME_TEST_BUILD_SAMPLE")
                 for it in items if it["v2_condition"] == cond][:3]
    assert len(rows) == 12
    return rows


def compare_resume_test():
    A = {r["item_id"]: r for r in jl(RT_DIR / "A_continuous_results.jsonl")}
    Bres = {r["item_id"]: r for r in jl(RT_DIR / "B_resumed_results.jsonl")}
    JB = jl(RT_DIR / "B_resumed_journal.jsonl")
    per_row = []
    for iid, a in A.items():
        b = Bres.get(iid)
        diff = ["MISSING_IN_B"] if b is None else [k for k in RESUME_CMP_KEYS if a[k] != b[k]]
        feat_equal = b is not None and a["trajectory_features"] == b["trajectory_features"]
        per_row.append({"item_id": iid, "segment_job_B": b and b["slurm_job_id"], "attempt_B": b and b["attempt"],
                        "differing_fields": diff, "trajectory_features_identical": feat_equal,
                        "final_ids_sha256": a["final_ids_sha256"]})
    killed = [e["item_id"] for e in JB if e["event"] == "ATTEMPT_START" and e["attempt"] == 1
              and not any(x["event"] == "OK" and x["item_id"] == e["item_id"] and x["attempt"] == 1 for x in JB)]
    segs = sum(1 for e in JB if e["event"] == "SEGMENT_START")
    seg2 = [i for i, e in enumerate(JB) if e["event"] == "SEGMENT_START"][1:2]
    after_restart = sorted({e["item_id"] for e in JB[seg2[0]:] if e["event"] == "OK"}) if seg2 else []
    for p in per_row:
        p["generated_after_restart"] = p["item_id"] in after_restart
    ok = (len(A) == len(Bres) == 12 and all(not p["differing_fields"] and p["trajectory_features_identical"]
                                            for p in per_row) and len(killed) == 1 and segs == 2
          and killed[0] in after_restart and len(after_restart) == 7)
    res = {"schema": "GENERALIZED_SAFETY_V2_M3PRIME_JOURNAL_RESUME_EXACTNESS_TEST_V1",
           "design": "A: one process generates 12 BUILD rows continuously. B: process 1 generates 5 rows, journals "
                     "ATTEMPT_START for row 6 and hard-exits (simulated kill mid-row); process 2 (fresh model load) "
                     "resumes from the journal. Rows compared byte-for-byte on final ids, decoded text, initial ids, "
                     "steps, seed, and the full per-step trajectory-feature record (exact float equality).",
           "n_rows": len(A), "rows_generated_after_restart": after_restart,
           "killed_mid_row": killed, "segments_B": segs, "per_row": per_row, "PASS": ok}
    (RT_DIR / "RESUME_TEST_RESULT.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("PASS", "n_rows", "killed_mid_row", "segments_B")}, indent=2))
    for p in per_row:
        print(p["item_id"], "job", p["segment_job_B"], "attempt", p["attempt_B"], "diff", p["differing_fields"],
              "features_identical", p["trajectory_features_identical"])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nonpert", "full", "resume_test_A", "resume_test_B1", "resume_test_B2",
                                       "resume_test_compare"], required=True)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="full mode: stop cleanly between rows before this many seconds of generation")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "resume_test_compare":
        sys.exit(0 if compare_resume_test() else 6)

    rt = RT.build_runtime()
    v_all, v_dija, v_ren = load_feature_directions()
    feat_obs = NewDevFeatureObserver(rt["model"], rt["blocks_attr"] if False else
                                     rt["model"].model.transformer.blocks[LAYER], v_all, v_dija, v_ren)

    if a.mode == "nonpert":
        ok = do_nonpert(rt, feat_obs)
        feat_obs.close()
        if not ok:
            print("NONPERT_TEST_FAILED -- DO NOT PROCEED TO FULL RUN", file=sys.stderr)
            sys.exit(5)
        print("NONPERT_TEST_PASSED", flush=True)
    elif a.mode.startswith("resume_test_"):
        RT_DIR.mkdir(exist_ok=True)
        rows = resume_test_rows(Builder())
        if a.mode == "resume_test_A":
            do_full(rt, feat_obs, rows, RT_DIR / "A_continuous_journal.jsonl", RT_DIR / "A_continuous_results.jsonl")
        else:
            do_full(rt, feat_obs, rows, RT_DIR / "B_resumed_journal.jsonl", RT_DIR / "B_resumed_results.jsonl",
                    _kill_after_rows=5 if a.mode == "resume_test_B1" else None)
        feat_obs.close()
    else:
        npt = OUT / "NONPERT_TEST_RESULT.json"
        if not npt.exists() or not json.loads(npt.read_text())["PASS"]:
            raise SystemExit("BLOCKED: non-perturbation test has not PASSed; run --mode nonpert first")
        do_full(rt, feat_obs, max_seconds=a.max_seconds)
        feat_obs.close()


if __name__ == "__main__":
    main()
