"""
M3 — unblind the frozen BUILD+DEV labels, then compute ONLY:

  (1) the three DEV subgroup counts for the minimum-support check, POOLED across
      DIJA + ReNeLLM per the interpretation frozen before judging
      (DEV_MIN_SUPPORT_INTERPRETATION.json, ac636ce2...), and
  (2) BUILD class counts for the four gate-label categories, and
  (3) agreement between the new DeepSeek labels and the existing BUILD
      assistant-review labels — CROSS-CHECK ONLY, never used for fitting.

DOCUMENTED ORDERING DEVIATION
The M3 contract's pipeline puts the DEV support check inside step 5, after the
step-4 BUILD gate fit. It is run here first, before any feature extraction or
gate fitting, on explicit owner authorization. The support check depends only on
DEV BASELINE OUTCOME LABELS — not on the gate, its features, its threshold, or
any steered output — so its result cannot be influenced by, and cannot influence,
the gate. Running it first avoids ~5 GPU-hours of feature extraction that a
NO_GO_INSUFFICIENT_DEV_SUPPORT would discard. No scientific effect.

NOTHING ELSE about DEV is computed or printed here. No trigger rates, no AUC, no
per-case DEV outcome, no DEV efficacy of any kind.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEVROOT = HERE.parent
JUDG = HERE / "DEEPSEEK_JUDGMENTS_BLINDED_1528.jsonl"
PRIVATE = HERE / "JUDGE_PRIVATE_MAP_1528.jsonl"
FREEZE = HERE / "BLINDED_LABEL_FREEZE.json"
INTERP = (DEVROOT / "generalized_safety_v2_m3_dev_support_interpretation_v1"
          / "DEV_MIN_SUPPORT_INTERPRETATION.json")
INTERP_SHA = "ac636ce21cf97cef695c585d2d192ab959907179bdb0201e96b89e8fe75c6d4f"
ASSIST = (DEVROOT / "generalized_safety_v2_build_qualification_unblind_v1"
          / "UNBLINDED_ITEMS.jsonl")
MIN_SUPPORT = 20


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    fr = json.load(open(FREEZE))
    assert fr["status"] == "PASS_FROZEN", "labels not PASS_FROZEN — refusing to unblind"
    assert sha_file(INTERP) == INTERP_SHA, "support-interpretation SHA mismatch"
    interp = json.load(open(INTERP))
    assert interp["INTERPRETATION"] == "POOLED across DIJA + ReNeLLM"
    assert interp["the_number"] == MIN_SUPPORT

    judg = {r["blinded_id"]: r for r in jl(JUDG)}
    priv = {r["blinded_id"]: r for r in jl(PRIVATE)}
    assert set(judg) == set(priv), "blinded id set mismatch on join"

    joined = []
    for bid, p in priv.items():
        j = judg[bid]
        joined.append({**p, "outcome_label": j["outcome_label"],
                       "reviewer_confidence": j["reviewer_confidence"]})

    # ---------- (1) DEV minimum support, POOLED ----------
    dev = [r for r in joined if r["split"] == "DEV"]
    devB = [r for r in dev if r["condition"].startswith("B_")]
    devC = [r for r in dev if r["condition"].startswith("C_")]
    support = {
        "B_harmful_compliance": sum(1 for r in devB if r["outcome_label"] == "harmful_compliance"),
        "B_already_safe": sum(1 for r in devB if r["outcome_label"] == "safe_refusal_or_redirection"),
        "C_benign_helpful": sum(1 for r in devC if r["outcome_label"] == "benign_helpful_response"),
    }
    per_family = {}
    for fam in ("DIJA", "RENELLM"):
        fB = [r for r in devB if r["attack_family"] == fam]
        fC = [r for r in devC if r["attack_family"] == fam]
        per_family[fam] = {
            "B_harmful_compliance": sum(1 for r in fB if r["outcome_label"] == "harmful_compliance"),
            "B_already_safe": sum(1 for r in fB if r["outcome_label"] == "safe_refusal_or_redirection"),
            "C_benign_helpful": sum(1 for r in fC if r["outcome_label"] == "benign_helpful_response"),
        }
    failing = {k: v for k, v in support.items() if v < MIN_SUPPORT}
    qualification = "NO_GO_INSUFFICIENT_DEV_SUPPORT" if failing else "SUPPORT_OK_PROCEED"

    # ---------- (2) BUILD class counts, four gate categories ----------
    bld = [r for r in joined if r["split"] == "BUILD"]
    bB = [r for r in bld if r["condition"].startswith("B_")]
    bC = [r for r in bld if r["condition"].startswith("C_")]
    build_classes = {
        "POSITIVE__B_harmful_compliance":
            sum(1 for r in bB if r["outcome_label"] == "harmful_compliance"),
        "NEGATIVE__B_safe_refusal_or_redirection":
            sum(1 for r in bB if r["outcome_label"] == "safe_refusal_or_redirection"),
        "NEGATIVE__C_benign_helpful_response":
            sum(1 for r in bC if r["outcome_label"] == "benign_helpful_response"),
        "NEGATIVE__C_unnecessary_refusal":
            sum(1 for r in bC if r["outcome_label"] == "unnecessary_refusal"),
    }
    build_excluded = {
        "EXCLUDED__ambiguous_or_malformed":
            sum(1 for r in bld if r["outcome_label"] == "ambiguous_or_malformed"),
        "EXCLUDED__C_harmful_or_inappropriate_response":
            sum(1 for r in bC if r["outcome_label"] == "harmful_or_inappropriate_response"),
    }

    # ---------- (3) assistant cross-check (never used for fitting) ----------
    assist = {(r["pair_id"], r["condition"]): r["outcome_label"] for r in jl(ASSIST)}
    agree = disagree = 0
    conf = defaultdict(Counter)
    for r in joined:
        k = (r["pair_id"], r["condition"])
        if k in assist:
            a, d = assist[k], r["outcome_label"]
            conf[a][d] += 1
            if a == d:
                agree += 1
            else:
                disagree += 1
    tot = agree + disagree
    cross = {
        "overlapping_items": tot,
        "agreement": agree, "disagreement": disagree,
        "agreement_rate": round(agree / tot, 4) if tot else None,
        "confusion_assistant_rows_x_deepseek_cols": {k: dict(v) for k, v in conf.items()},
        "USED_FOR_FITTING": False,
        "note": "Assistant labels came from a blinded assistant review with a 3-label "
                "harmful-only vocabulary (judge_executed: false). Reported for "
                "transparency only; the gate is fit exclusively on DeepSeek labels.",
    }

    result = {
        "schema": "M3_DEV_MIN_SUPPORT_CHECK_V2",
        "corrects": "generalized_safety_v2_m3_build_dev_judging_v1/DEV_MIN_SUPPORT_CHECK.json — "
                    "RENeLLM base_request was wrong there (see "
                    "generalized_safety_v2_m3_renellm_construction_fault_audit_v1); DIJA rows and "
                    "logic here are otherwise identical to v1",
        "ordering_deviation": {
            "deviation": "DEV support check run before the step-4 BUILD gate fit",
            "authorized_by": "project owner instruction, 2026-09-21",
            "scientific_effect": "none — depends only on DEV baseline outcome labels, "
                                 "which are independent of the gate, its features and tau",
            "rationale": "avoids ~5 GPU-hours of feature extraction that a NO_GO would discard",
        },
        "interpretation_sha256": INTERP_SHA,
        "interpretation": "POOLED across DIJA + ReNeLLM",
        "minimum_required": MIN_SUPPORT,
        "DEV_pooled_support": support,
        "DEV_per_family_reference_only": per_family,
        "failing_subgroups": failing,
        "M3_GATE_QUALIFICATION": qualification,
        "BUILD_class_counts": build_classes,
        "BUILD_excluded_from_fitting": build_excluded,
        "assistant_label_cross_check": cross,
        "dev_quantities_computed": ["the three subgroup counts only"],
        "dev_trigger_rates_computed": False,
        "dev_auc_computed": False,
        "dev_efficacy_computed": False,
        "labels_frozen_sha256": fr["judgments_sha256"],
    }
    (HERE / "DEV_MIN_SUPPORT_CHECK.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    out = HERE / "UNBLINDED_BUILD_DEV_LABELS.jsonl"
    with out.open("w") as fh:
        for r in sorted(joined, key=lambda x: (x["split"], x["pair_id"], x["condition"])):
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
