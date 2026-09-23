"""M3' step 9 — BUILD gate fit (frozen hyperparameters), fit on BUILD baseline generations only.

Model, exactly as frozen in the parent contract:
  StandardScaler fit on BUILD only; LogisticRegression(penalty=l2, C=1.0, solver=lbfgs, fit_intercept=True,
  class_weight=None, max_iter=10000, tol=1e-8); case-equal weighting (each case contributes equal TOTAL weight,
  so long cases do not dominate: each of a case's step rows gets weight 1/n_steps_of_that_case).

Labels (case-level supervision; every step of a case carries its case's label):
  positive: B baseline judged harmful_compliance
  negative: B safe_refusal_or_redirection, C benign_helpful_response, C unnecessary_refusal
  excluded: ambiguous_or_malformed (any), C harmful_or_inappropriate_response

Rows: non-empty-mask steps only (the frozen empty-mask rule is "no decision", so those steps carry no features).
BUILD class counts for all four label categories are reported BEFORE the fit is frozen, per the contract.
"""
import hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV = ROOT / "analysis_output/rrae_development"
HERE = Path(__file__).resolve().parent
CASES = DEV / "generalized_safety_v2_m3_trajectory_features_v1/CASES.jsonl"
NSHARDS = 4
F = ("f1_proj_ALL", "f2_proj_DIJA", "f3_proj_REN", "f4_dproj_ALL", "f5_dproj_DIJA", "f6_dproj_REN",
     "f7_norm", "f8_denoise_frac", "f9_mask_frac")
POS = {("B", "harmful_compliance")}
NEG = {("B", "safe_refusal_or_redirection"), ("C", "benign_helpful_response"), ("C", "unnecessary_refusal")}


def jl(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    fz = json.loads((HERE / "BUILD_FEATURE_FREEZE.json").read_text())
    assert fz["status"] == "PASS_FROZEN", "BUILD features not PASS_FROZEN — refusing to fit"
    build = {c["generation_id"]: c for c in jl(CASES) if c["split"] == "BUILD"}
    rows = []
    for s in range(NSHARDS):
        p = HERE / f"BUILD_CAPTURE_RESULTS_shard{s:02d}.jsonl"
        assert sha_file(p) == fz["shard_results_sha256"][f"shard{s:02d}"], f"shard {s} changed since freeze"
        rows += jl(p)
    assert len(rows) == 1222

    # ---- BUILD class counts for ALL FOUR categories, reported before freezing the fit
    cls = Counter((build[r["build_generation_id"]]["condition"][0], build[r["build_generation_id"]]["outcome_label"])
                  for r in rows)
    four = {
        "B_harmful_compliance": cls[("B", "harmful_compliance")],
        "B_safe_refusal_or_redirection": cls[("B", "safe_refusal_or_redirection")],
        "C_benign_helpful_response": cls[("C", "benign_helpful_response")],
        "C_unnecessary_refusal": cls[("C", "unnecessary_refusal")],
    }
    excluded = {f"{k[0]}_{k[1]}": v for k, v in cls.items()
                if (k[0], k[1]) not in POS | NEG}
    print("BUILD class counts (four fitting categories):", json.dumps(four, indent=2))
    print("BUILD excluded from fitting:", json.dumps(excluded, indent=2))

    X, y, w, groups = [], [], [], []
    per_case = Counter()
    for r in rows:
        c = build[r["build_generation_id"]]
        key = (c["condition"][0], c["outcome_label"])
        if key in POS:
            lab = 1
        elif key in NEG:
            lab = 0
        else:
            per_case["excluded_cases"] += 1
            continue
        feats = [f for f in r["trajectory_features"] if not f.get("empty_mask")]
        if not feats:
            per_case["cases_without_usable_steps"] += 1
            continue
        per_case["fit_cases"] += 1
        per_case["fit_cases_pos" if lab else "fit_cases_neg"] += 1
        wt = 1.0 / len(feats)          # case-equal weighting
        for f in feats:
            X.append([float(f[k]) for k in F]); y.append(lab); w.append(wt)
            groups.append(r["build_generation_id"])
    X, y, w = np.asarray(X, dtype=np.float64), np.asarray(y), np.asarray(w)
    assert np.isfinite(X).all(), "non-finite features"

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", fit_intercept=True, class_weight=None,
                             max_iter=10000, tol=1e-8).fit(scaler.transform(X), y, sample_weight=w)

    sparse_note = None
    rare = {k: v for k, v in four.items() if k in ("B_safe_refusal_or_redirection", "C_unnecessary_refusal")}
    if min(rare.values()) < 20:
        sparse_note = (f"RARE NEGATIVE CLASS SPARSE: {rare}. Per the contract this is recorded explicitly: with a "
                       "rare class this thin the gate is closer to a B-vs-C classifier than a true "
                       "intervention-need classifier. Documented, not worked around; no rule is changed.")
        print("\n" + sparse_note)

    fit = {
        "schema": "GENERALIZED_SAFETY_V2_M3PRIME_BUILD_GATE_FIT_V1",
        "contract_item": "M3prime checklist item 10 (BUILD gate fit; BUILD class counts reported before freeze)",
        "build_class_counts_four_categories": four,
        "build_excluded_from_fitting": excluded,
        "rare_negative_class_note": sparse_note,
        "supervision": "case-level, not step-level: every step of a case carries the case's endpoint label; there "
                       "is no step-level ground truth for when intervention was actually needed",
        "features": list(F),
        "empty_mask_rows_excluded": True,
        "case_equal_weighting": "each case's step rows share total weight 1.0 (weight 1/n_steps per row)",
        "hyperparameters": {"scaler": "StandardScaler fit on BUILD only", "penalty": "l2", "C": 1.0,
                            "solver": "lbfgs", "fit_intercept": True, "class_weight": None,
                            "max_iter": 10000, "tol": 1e-8},
        "fit_rows": int(X.shape[0]),
        "fit_cases": per_case["fit_cases"],
        "fit_cases_positive": per_case["fit_cases_pos"],
        "fit_cases_negative": per_case["fit_cases_neg"],
        "excluded_cases": per_case["excluded_cases"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "n_iter": int(clf.n_iter_[0]),
        "converged": int(clf.n_iter_[0]) < 10000,
        "build_feature_freeze_sha256": sha_file(HERE / "BUILD_FEATURE_FREEZE.json"),
        "fit_on": "BUILD only (no NEW_DEV row, no final-pool row, no old-DEV row)",
    }
    assert fit["converged"], "lbfgs hit max_iter — not freezing a non-converged fit"
    (HERE / "BUILD_GATE_FIT.json").write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in fit.items() if k not in ("scaler_mean", "scaler_scale", "coef")},
                     indent=2, sort_keys=True))
    print("GATE_FIT_FROZEN", sha_file(HERE / "BUILD_GATE_FIT.json"))


if __name__ == "__main__":
    main()
