#!/usr/bin/env python3
"""PHASE 0 - verify the CDG injection data and inventory what exists for the
harm-vs-safe direction.

Read-only. Writes its findings to harm_dir/data/phase0_inventory.json and prints
a human-readable report. Touches nothing outside harm_dir/.
"""
import collections
import glob
import json
import os
import statistics
import sys

PROMPTS = "/home/ore99/serverFiles/prompts/cdg_injection"
BASE_MANIFEST = "/home/ore99/serverFiles/outputs/manifest.jsonl"
BASE_STATES = "/home/ore99/serverFiles/outputs/llada"
PREFILL_RUNS = "/home/ore99/serverFiles/dijawithprefill/runs/full"
OUT = "/home/ore99/experement/harm_dir/data/phase0_inventory.json"

VARIANTS = ["A_harmful_clean", "B_harmful_injected", "C_neutral_injected", "D_neutral_clean"]


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def wc(text):
    return len((text or "").split())


def lenstats(vals):
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": min(vals),
        "median": statistics.median(vals),
        "mean": round(statistics.mean(vals), 1),
        "max": max(vals),
    }


def main():
    inv = {}

    # ---------- 1. the four prompt folders ----------
    print("=" * 72)
    print("1. PROMPT FOLDERS -", PROMPTS)
    print("=" * 72)
    folders = {}
    for v in VARIANTS:
        p = os.path.join(PROMPTS, v, "cases.json")
        if not os.path.exists(p):
            print(f"  MISSING: {p}")
            folders[v] = {"exists": False}
            continue
        cases = json.load(open(p))
        keys = sorted(cases[0].keys())
        folders[v] = {
            "exists": True,
            "n_cases": len(cases),
            "fields": keys,
            "id_range": [cases[0]["id"], cases[-1]["id"]],
            "has_response_field": any("response" in k.lower() for k in keys),
            "attack_methods": dict(collections.Counter(c.get("attack_method", "none") for c in cases)),
        }
        print(f"  {v:22s} n={len(cases):4d}  ids {cases[0]['id']}..{cases[-1]['id']}  fields={keys}")
    inv["prompt_folders"] = folders

    B_cases = json.load(open(os.path.join(PROMPTS, "B_harmful_injected", "cases.json")))
    print("\n  --- format of ONE B_harmful_injected item (B000) ---")
    print(json.dumps(B_cases[0], indent=2))
    print("  >>> VERDICT: PROMPT ONLY. No response field. `user_content` is the DIJA")
    print("      scaffold with <mask:k> slots; `behavior` is the bare harmful request.")
    inv["B_example"] = B_cases[0]

    # ---------- 2. base run: responses + states ----------
    print()
    print("=" * 72)
    print("2. BASE RUN -", os.path.dirname(BASE_MANIFEST))
    print("=" * 72)
    base = jsonl(BASE_MANIFEST)
    baseB = [r for r in base if r["variant"] == "B_harmful_injected"]
    labels = collections.Counter(r["judge"]["label"] for r in baseB if r.get("judge"))
    n_states = {v: len(glob.glob(f"{BASE_STATES}/{v}/*.pt")) for v in VARIANTS}
    comply_lens = [wc(r["response_text"]) for r in baseB if r.get("judge") and r["judge"]["label"] == "comply"]
    print(f"  manifest rows: {len(base)}  ({dict(collections.Counter(r['variant'] for r in base))})")
    print(f"  B rows with response_text: {sum(1 for r in baseB if r.get('response_text'))}/{len(baseB)}")
    print(f"  B ASR-judge labels: {dict(labels)}   (judge_model={baseB[0]['judge']['judge_model']})")
    print(f"  NOTE: base-run judge is ASR only (comply/partial/refuse). No valence_category.")
    print(f"  .pt state files: {n_states}")
    print(f"  base 'comply' response length (words): {lenstats(comply_lens)}")
    inv["base_run"] = {
        "manifest": BASE_MANIFEST,
        "n_B": len(baseB),
        "n_B_with_response": sum(1 for r in baseB if r.get("response_text")),
        "asr_labels": dict(labels),
        "judge_kind": "ASR only (comply/partial/refuse) - NOT valence_category",
        "n_state_files": n_states,
        "comply_len_words": lenstats(comply_lens),
    }

    # what is actually inside a .pt
    print("\n  --- contents of B000__seed0.pt ---")
    try:
        import torch

        d = torch.load(f"{BASE_STATES}/B_harmful_injected/B000__seed0.pt", map_location="cpu", weights_only=False)
        regions = list(d["hidden"].keys())
        fracs = sorted(d["hidden"][regions[0]].keys())
        layers = sorted(d["hidden"][regions[0]][fracs[0]].keys())
        dim = tuple(d["hidden"][regions[0]][fracs[0]][layers[0]].shape)
        print(f"  top-level keys : {list(d.keys())}")
        print(f"  hidden regions : {regions}")
        print(f"  denoise fracs  : {fracs}")
        print(f"  layers         : {layers}")
        print(f"  vector shape   : {dim}  (mean-pooled over region tokens, per frac, per layer)")
        print(f"  sae            : {d['meta']['saes']} -> 16384-d features, same fracs/layers")
        print("  >>> These are GENERATION-TIME pooled states (during denoising), layers 11/16/26.")
        print("  >>> They are NOT teacher-forced [prompt+response] states, and layers 25/27 are absent.")
        print("  >>> Phase 1 therefore needs a fresh GPU capture regardless.")
        inv["base_states"] = {
            "top_keys": list(d.keys()),
            "regions": regions,
            "fracs": [float(f) for f in fracs],
            "layers": [int(l) for l in layers],
            "dim": dim[0],
            "pooling": "mean over region tokens at each denoising fraction",
            "teacher_forced": False,
        }
    except ImportError:
        print("  (torch unavailable in this interpreter - skipping .pt introspection)")
        inv["base_states"] = {"error": "torch unavailable"}

    # ---------- 3. prefill arms: graded valence ----------
    print()
    print("=" * 72)
    print("3. PREFILL ARMS -", PREFILL_RUNS)
    print("=" * 72)
    arms = sorted(d for d in glob.glob(PREFILL_RUNS + "/*/") if os.path.exists(d + "graded_judge.jsonl"))
    per_case = collections.defaultdict(dict)   # case_id -> arm -> record
    arm_info = {}
    for a in arms:
        name = a.rstrip("/").split("/")[-1]
        graded = {r["case_id"]: r for r in jsonl(a + "graded_judge.jsonl") if r.get("group") == "B"}
        man = {r["case_id"]: r for r in jsonl(a + "manifest.jsonl") if r["variant"] == "B_harmful_injected"}
        n_pt = len(glob.glob(a + "llada/B_harmful_injected/*.pt"))
        cats = collections.Counter(r["graded"]["valence_category"] for r in graded.values() if r.get("graded"))
        pos = collections.Counter((r["position"], r["dose"]) for r in graded.values())
        arm_info[name] = {
            "position_dose": list(pos)[0] if len(pos) == 1 else list(pos),
            "n_B_graded": len(graded),
            "n_B_states": n_pt,
            "valence_categories": dict(cats),
        }
        for cid, r in graded.items():
            per_case[cid][name] = {
                "category": r["graded"]["valence_category"],
                "valence_score": r["graded"].get("valence_score"),
                "specificity": r["graded"].get("specificity_score"),
                "harm_delivered": r.get("harm_delivered"),
                "response": man[cid]["response_text"],
                "words": wc(man[cid]["response_text"]),
            }
        print(f"  {name:9s} pos/dose={str(arm_info[name]['position_dose']):14s} graded={len(graded):3d} "
              f"states={n_pt:3d}  {dict(cats)}")
    inv["prefill_arms"] = arm_info

    # ---------- 4. harm / safe pool for Phase 1 ----------
    print()
    print("=" * 72)
    print("4. HARM / SAFE POOL FOR PHASE 1  (base-request id = pairing unit)")
    print("=" * 72)
    HARM_CAT, SAFE_CAT = "unchanged_harmful", "substantive_safe_substitution"
    harm_ids, safe_ids = set(), set()
    harm_rows, safe_rows = [], []
    for cid, arms_d in per_case.items():
        for arm, r in arms_d.items():
            if r["category"] == HARM_CAT:
                harm_ids.add(cid)
                harm_rows.append((cid, arm, r["words"]))
            elif r["category"] == SAFE_CAT:
                safe_ids.add(cid)
                safe_rows.append((cid, arm, r["words"]))

    base_comply_ids = {r["case_id"] for r in baseB if r.get("judge") and r["judge"]["label"] == "comply"}
    both = harm_ids & safe_ids
    harm_any = harm_ids | base_comply_ids

    print(f"  distinct B base-request ids                        : {len(per_case)}")
    print(f"  ids with >=1 graded '{HARM_CAT}'        : {len(harm_ids)}  ({len(harm_rows)} responses total)")
    print(f"  ids with >=1 graded '{SAFE_CAT}' : {len(safe_ids)}  ({len(safe_rows)} responses total)")
    print(f"  ids with BOTH -> free same-prompt harm/safe pair    : {len(both)}")
    print(f"  ids with a harmful response from ANY source        : {len(harm_any)}")
    print(f"    (graded unchanged_harmful {len(harm_ids)}  U  base-run ASR 'comply' {len(base_comply_ids)})")
    print(f"  ids needing a GENERATED safe response              : {len(harm_any - safe_ids)}")

    hl = lenstats([w for _, _, w in harm_rows])
    sl = lenstats([w for _, _, w in safe_rows])
    print(f"\n  length (words) unchanged_harmful      : {hl}")
    print(f"  length (words) substantive_safe_sub   : {sl}")
    print(f"  >>> The two pools are already length-matched (median {hl['median']} vs {sl['median']}).")

    harm_by_arm = collections.Counter(a for _, a, _ in harm_rows)
    safe_by_arm = collections.Counter(a for _, a, _ in safe_rows)
    print("\n  arm composition (CONFOUND TO WATCH - prefill text lives inside the response):")
    print(f"    {'arm':10s} {'harm':>6s} {'safe':>6s}")
    for a in sorted(arm_info):
        print(f"    {a:10s} {harm_by_arm.get(a,0):6d} {safe_by_arm.get(a,0):6d}")

    inv["phase1_pool"] = {
        "pairing_unit": "B base-request id (case_id)",
        "n_ids": len(per_case),
        "n_ids_harm_graded": len(harm_ids),
        "n_ids_safe_graded": len(safe_ids),
        "n_ids_both": len(both),
        "n_ids_harm_any_source": len(harm_any),
        "n_ids_need_generated_safe": len(harm_any - safe_ids),
        "n_harm_responses": len(harm_rows),
        "n_safe_responses": len(safe_rows),
        "harm_len_words": hl,
        "safe_len_words": sl,
        "harm_by_arm": dict(harm_by_arm),
        "safe_by_arm": dict(safe_by_arm),
        "ids_both": sorted(both),
        "ids_harm_any": sorted(harm_any),
        "ids_safe": sorted(safe_ids),
    }

    # ---------- 5. probes for Gate C ----------
    print()
    print("=" * 72)
    print("5. EXISTING DIRECTIONS (Gate C reference)")
    print("=" * 72)
    for p in ["/scratch/ore99/clockv2/probes/v_injection_svd.pt",
              "/scratch/ore99/clockv2/probes/v_refusal.pt"]:
        ok = os.path.exists(p)
        print(f"  {'OK ' if ok else 'MISSING'} {p}")
        if ok:
            try:
                import torch
                d = torch.load(p, map_location="cpu", weights_only=False)
                print(f"      layers={d.get('layers')}  extraction={str(d.get('extraction_point'))[:70]}")
            except ImportError:
                pass
    inv["probes"] = {
        "v_injection_svd": "/scratch/ore99/clockv2/probes/v_injection_svd.pt",
        "v_refusal": "/scratch/ore99/clockv2/probes/v_refusal.pt",
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(inv, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
