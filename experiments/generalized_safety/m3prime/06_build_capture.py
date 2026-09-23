"""M3' step 8a — BUILD baseline trajectory-feature capture with the M3' observer (the one that passed the
non-perturbation test and produced the NEW_DEV features; NOT the M3 extractor, so BUILD and NEW_DEV features
come from one code path).

Each BUILD case is regenerated under its frozen seed with the read-only observer attached. Sharded by index;
each shard keeps its own journal/results and is independently resumable (do_full, unchanged).

run_item is wrapped (additively) to record, per row, the hashes needed for the step-8b byte-identity check in the
SAME SPAN the stored BUILD rows used:
  DIJA    -> tokens at the initial masked positions   (stored: generated_token_ids / response_sha256)
  RENELLM -> the standard decoded output span          (stored: response_sha256)
The wrapper computes hashes only; it never alters generation. No response text is written to disk, only hashes.
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV = ROOT / "analysis_output/rrae_development"
sys.path.insert(0, str(DEV / "generalized_safety_v2_m3prime_llada_baseline_v1"))
import m3prime_llada_generate as G  # noqa: E402
RT = G.RT
HERE = Path(__file__).resolve().parent
CASES = DEV / "generalized_safety_v2_m3_trajectory_features_v1/CASES.jsonl"
STORED = DEV / "generalized_safety_v2_llada_baseline_full_execution_v1/outputs"
MASK_ID = 126336


ROUTE_STATS = {"DIJA_ORIGINAL": 0, "DIJA_COMPAT": 0, "cross_check_done": 0, "cross_check_skipped": 0}


def route_and_input_local(B, sc):
    """M1's pair-level DIJA route rule, applied exactly as build_manifests_v1.Builder.route_and_input applies it:
    decide on the group's B_DIJA prompt with the source helper (realized2328), use the compat helper if it cannot
    locate the clean/harm span, and use that one helper for both B and C of the pair.

    One documented difference, forced by the data: Builder additionally asserts that BOTH frozen helpers yield
    byte-identical ids. For the 11 M1-recovered B_DIJA rows that assertion cannot be evaluated at all -- the source
    helper raises on exactly those rows, which is why they were recovered with the compat helper in the first place
    (localization_compat_v1: the patch only replaces the raise with harm_span=None; nothing else differs). So the
    cross-check runs whenever the other helper CAN build the row, and is recorded as skipped when it cannot.
    No generation setting changes: the helper selected here is the one M1's own rule selects.
    """
    if sc["family"] == "RENELLM":
        return B.route_and_input(sc)
    bbase = {"attack_family": "DIJA", "condition": "B", "pair_id": sc["pair_id"],
             "prompt": sc["route_B_prompt"], "clean_semantic_request": sc["route_B_clean"]}
    try:
        B.m1.build_input(bbase, B.tok, B.route_helper, B.Hm["baseline"])
        route = "DIJA_ORIGINAL"
    except RuntimeError as e:
        if "Could not locate clean/harm prompt" not in str(e):
            raise
        route = "DIJA_COMPAT"
    base = {"attack_family": "DIJA", "condition": sc["condition"], "pair_id": sc["pair_id"],
            "prompt": sc["prompt"], "clean_semantic_request": sc.get("clean")}
    use = B.Hm["compat_helper"] if route == "DIJA_COMPAT" else B.Hm["helper"]
    other = B.Hm["helper"] if route == "DIJA_COMPAT" else B.Hm["compat_helper"]
    x, _, meta = B.m1.build_input(base, B.tok, use, B.Hm["baseline"])
    try:
        x2, _, _ = B.m1.build_input(base, B.tok, other, B.Hm["baseline"])
        assert G.torch.equal(x, x2), "route helpers disagree on input ids"
        ROUTE_STATS["cross_check_done"] += 1
    except RuntimeError as e:
        if "Could not locate clean/harm prompt" not in str(e):
            raise
        ROUTE_STATS["cross_check_skipped"] += 1
    ROUTE_STATS[route] += 1
    return route, x, meta


def build_items(B):
    """One item per BUILD baseline case, with the same fields the nonpert/NEW_DEV items carry (same seed_rule,
    same expected_* fields), for every BUILD case instead of 10 per stratum."""
    canon = {r["id"]: r for r in G.jl(G.CANON)}
    cases = [c for c in G.jl(CASES) if c["split"] == "BUILD"]
    cases.sort(key=lambda c: (c["pair_id"], c["condition"]))
    stored_masks = {}
    for pid in sorted({c["pair_id"] for c in cases}):
        for r in map(json.loads, (STORED / f"pair_{pid:06d}.jsonl").open()):
            stored_masks[(r["pair_id"], r["condition"])] = r
    items = []
    for c in cases:
        pid, cond_key = c["pair_id"], c["condition"]
        cond, family = cond_key.split("_")[0], c["attack_family"]
        sc = {"family": family, "condition": cond, "v2_condition": cond_key, "group_key": f"PAIR_{pid:04d}",
              "prompt": c["prompt"], "clean": canon[f"{cond}_{pid:04d}"]["clean_prompt"] if family == "DIJA" else None,
              "pair_id": pid, "seed": G.seed_rule(pid, cond), "domain": None,
              "source_case_id": f"PAIR_{pid:04d}|{family}|{cond}", "id_prefix": "m3pbuild"}
        if family == "DIJA":
            sc["route_B_prompt"] = canon[f"B_{pid:04d}"]["prompt"]
            sc["route_B_clean"] = canon[f"B_{pid:04d}"]["clean_prompt"]
        route, x, meta = route_and_input_local(B, sc)
        init_count = int((x == MASK_ID).sum())
        steps = init_count if family == "DIJA" else 128
        st = stored_masks[(pid, cond_key)]
        assert init_count == int(st["initial_fillable_mask_count"]), f"mask count drift {c['generation_id']}"
        assert steps == int(st["trajectory_step_count"]) == int(c["trajectory_step_count"]), \
            f"step count drift {c['generation_id']}"
        assert int(sc["seed"]) == int(c["generation_seed"]) == int(st["seed"]), f"seed drift {c['generation_id']}"
        it = {"item_id": f"m3pbuild_PAIR_{pid:04d}_{cond_key}_BASELINE", "split": "BUILD_FEATURE_CAPTURE",
              "arm": "BASELINE", "method": "BASELINE", "direction": None, "beta": 0.0, "beta_source": None,
              "attack_family": family, "condition": cond, "v2_condition": cond_key,
              "group_key": f"PAIR_{pid:04d}", "group_ordinal": pid, "pair_id": pid,
              "source_case_id": sc["source_case_id"], "domain": None, "prompt": c["prompt"],
              "clean_semantic_request": sc["clean"], "generation_seed": int(sc["seed"]),
              "input_helper_route": route, "expected_initial_ids_sha256": B.m1.tensor_sha256(x),
              "expected_initial_fillable_mask_count": init_count, "expected_trajectory_steps": steps,
              "steering_condition": "BASELINE", "vector_key": None, "prompt_sha256": G.H(c["prompt"]),
              "runtime_mode": meta["generation_mode"], "layer_zero_based": 16, "token_scope": "current_mask",
              "scope": "current_mask", "schedule": "persistent", "gen_length": 128, "temperature": 0.2,
              "remasking": "low_confidence",
              "build_generation_id": c["generation_id"], "stored_response_sha256": c["response_sha256"],
              "stored_generated_token_ids_sha256": st.get("generated_token_ids_sha256")}
        if family == "DIJA":
            it["_maskpos"] = (x[0] == MASK_ID).nonzero().flatten().tolist()
            assert len(it["_maskpos"]) == steps
        items.append(it)
    return items


def install_span_wrapper(rt):
    """Additive: capture the final tensor via hist.decode_final, then record span hashes on the result."""
    hist, tok = rt["H"]["hist"], rt["tok"]
    cap = {}
    orig_decode = hist.decode_final

    def decode_final(tokenizer, x, regions, metadata):
        cap["x"] = x.detach().cpu()
        return orig_decode(tokenizer, x, regions, metadata)

    hist.decode_final = decode_final
    orig_run = RT.run_item

    def run_item(rt_, item, *, variant="WRAPPED"):
        cap.clear()
        res = orig_run(rt_, item, variant=variant)
        if item["attack_family"] == "DIJA":
            xf = cap["x"][0]
            ids = [int(xf[p]) for p in item["_maskpos"]]
            text = tok.decode(ids, skip_special_tokens=True)
            res["span_rule"] = "DIJA_INITIAL_MASK_POSITIONS"
            res["span_token_ids_sha256"] = hashlib.sha256(
                json.dumps(ids, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            res["span_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        else:
            res["span_rule"] = "RENELLM_STANDARD_OUTPUT_REGION"
            res["span_token_ids_sha256"] = None
            res["span_text_sha256"] = res["decoded_output_sha256"]
        res["build_generation_id"] = item["build_generation_id"]
        res["stored_response_sha256"] = item["stored_response_sha256"]
        res.pop("decoded_output", None)  # hashes only; no generated text on disk
        return res

    RT.run_item = run_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()
    npt = json.loads((G.OUT / "NONPERT_TEST_RESULT.json").read_text())
    assert npt["PASS"] is True, "BLOCKED: observer non-perturbation test is not PASS"
    rt = RT.build_runtime()
    v_all, v_dija, v_ren = G.load_feature_directions()
    feat_obs = G.NewDevFeatureObserver(rt["model"], rt["model"].model.transformer.blocks[G.LAYER],
                                       v_all, v_dija, v_ren)
    install_span_wrapper(rt)
    items = build_items(G.Builder())
    assert len(items) == 1222, len(items)
    rows = [it for i, it in enumerate(items) if i % a.nshards == a.shard]
    print("ROUTE_STATS", json.dumps(ROUTE_STATS), flush=True)
    print(f"shard {a.shard}/{a.nshards}: {len(rows)} of {len(items)} BUILD cases", flush=True)
    st = G.do_full(rt, feat_obs, rows,
                   HERE / f"BUILD_CAPTURE_JOURNAL_shard{a.shard:02d}.jsonl",
                   HERE / f"BUILD_CAPTURE_RESULTS_shard{a.shard:02d}.jsonl")
    feat_obs.close()
    print("SHARD_STATUS", a.shard, st, flush=True)
    sys.exit(0 if st == "COMPLETE" else 7)


if __name__ == "__main__":
    main()
