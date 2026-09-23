"""
M2 beta solve - implements M2_BETA_SOLVING_PROCEDURE_V1.json literally.

Reads the frozen measurement observations, solves beta_M2 per direction on
DEV_SOLVE, re-solves independently on DEV_CHECK as a declared agreement check,
and writes the item-7 beta freeze.

Runs only after all 4 measurement shards are complete. Reads no outcome label.
"""

import json, hashlib, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])
DEV  = ROOT / "analysis_output/rrae_development"
HERE = Path(__file__).parent
CONTRACT = DEV / "generalized_safety_v2_m2_contract_v1"
OUT = DEV / "generalized_safety_v2_m2_beta_freeze_v1"

SP_PATH = CONTRACT / "M2_BETA_SOLVING_PROCEDURE_V1.json"
EXPECT_SP_SHA = "b3a975ea591533dff0cb9e6805b94a443956d3923cc7ea5f85405aa9b688a502"

ARMS = ["V_DIJA", "V_RENELLM", "V_ALL"]
COL = {"V_DIJA": 1, "V_RENELLM": 2, "V_ALL": 3}   # columns in OBSERVATIONS.npy


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fail(msg):
    print("SOLVE_FAIL:", msg)
    sys.exit(1)


# ---------- bind the frozen procedure ----------
if sha256_file(SP_PATH) != EXPECT_SP_SHA:
    fail("solving procedure SHA changed since it was frozen")
sp = json.load(open(SP_PATH))
if not sp["frozen_before_any_calibration_number_observed"]:
    fail("solving procedure not frozen in advance")

EPS = float(sp["solver"]["eps"])
LO, HI = sp["solver"]["bracket"]
ITERS = int(sp["solver"]["iterations"])
TOL = float(sp["build_dev_combination"]["agreement_check"]["tolerance_band_declared_in_advance"])
ABORT_SAT = float(sp["fail_closed_guards"]["displacement_ceiling"]["abort_threshold"])
M1 = sp["solver"]["beta_M1_targets"]
FW = sp["per_direction_population"]["V_ALL"]["family_weights"]
print("SOLVING_PROCEDURE_BOUND=PASS")

# ---------- load measurements ----------
cases = []
for s in range(4):
    d = HERE / f"shard_{s:02d}"
    summ = json.load(open(d / "SHARD_SUMMARY.json"))
    if summ["status"] != "PASS" or summ["steering_applied"] or summ["hidden_states_modified"]:
        fail(f"shard {s} summary invalid")
    if sha256_file(d / "OBSERVATIONS.npy") != summ["observations_sha256"]:
        fail(f"shard {s} observations SHA mismatch")
    obs = np.load(d / "OBSERVATIONS.npy").astype(np.float64)
    for r in (json.loads(l) for l in open(d / "OBSERVATION_INDEX.jsonl") if l.strip()):
        blk = obs[r["offset"]: r["offset"] + r["observations"]]
        if blk.shape[0] != r["observations"]:
            fail(f"index/observation mismatch in shard {s}")
        if not np.isfinite(blk).all():
            fail("non-finite observation")
        cases.append({**r, "obs": blk})

if len(cases) != 306:
    fail(f"expected 306 source cases, got {len(cases)}")
print(f"MEASUREMENTS_LOADED=PASS cases={len(cases)} "
      f"observations={sum(c['observations'] for c in cases)}")


# ---------- the displacement functional (per observation) ----------
def delta(H, p, beta):
    """Delta = s*(p+beta) - p ; s = H / (sqrt(H^2 + 2*beta*p + beta^2) + eps)"""
    s = H / (np.sqrt(H * H + 2.0 * beta * p + beta * beta) + EPS)
    return s * (p + beta) - p


def case_median_delta(case, arm, beta):
    """stage 1: median over all current_mask observations of this source case"""
    H = case["obs"][:, 0]
    p = case["obs"][:, COL[arm]]
    return float(np.median(delta(H, p, beta)))


def stage2(subset, arm, beta):
    """stage 2: median across source cases; equal-family-weighted for V_ALL"""
    if arm != "V_ALL":
        fam = "DIJA" if arm == "V_DIJA" else "RENELLM"
        vals = [case_median_delta(c, arm, beta) for c in subset if c["attack_family"] == fam]
        if not vals:
            fail(f"no cases for {arm}")
        return float(np.median(vals))
    # equal-family-weighted median across per-case values
    pairs = []
    for fam, w in FW.items():
        vals = [case_median_delta(c, arm, beta) for c in subset if c["attack_family"] == fam]
        if not vals:
            fail(f"no {fam} cases for V_ALL")
        pairs.extend((v, w / len(vals)) for v in vals)
    pairs.sort(key=lambda t: t[0])
    tot = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2.0:
            return float(v)
    return float(pairs[-1][0])


def ceiling_fraction(subset, arm, target):
    """fraction of source cases whose per-case median ceiling (H - p) < beta_M1"""
    fam = {"V_DIJA": "DIJA", "V_RENELLM": "RENELLM"}.get(arm)
    sel = [c for c in subset if fam is None or c["attack_family"] == fam]
    ceils = [float(np.median(c["obs"][:, 0] - c["obs"][:, COL[arm]])) for c in sel]
    return float(np.mean([c < target for c in ceils])), len(sel)


def solve(subset, arm):
    target = float(M1[arm])

    # guard: displacement ceiling
    sat, n = ceiling_fraction(subset, arm, target)
    if sat > ABORT_SAT:
        fail(f"{arm}: {sat:.3f} of cases have ceiling < beta_M1 "
             f"(abort threshold {ABORT_SAT}) - displacement-matching infeasible")

    # guard: monotonicity over the bracket
    grid = np.linspace(LO, HI, 60)
    vals = [stage2(subset, arm, float(b)) for b in grid]
    if not all(b > a for a, b in zip(vals, vals[1:])):
        fail(f"{arm}: Delta(beta) is not strictly increasing over the bracket")

    if not (vals[0] <= target <= vals[-1]):
        fail(f"{arm}: target {target} outside achievable range "
             f"[{vals[0]:.3f}, {vals[-1]:.3f}]")

    lo, hi = float(LO), float(HI)
    for _ in range(ITERS):
        mid = (lo + hi) / 2.0
        if stage2(subset, arm, mid) < target:
            lo = mid
        else:
            hi = mid
    beta2 = (lo + hi) / 2.0
    return {
        "beta_M1_target": target,
        "beta_M2": beta2,
        "beta_ratio_M2_over_M1": beta2 / target,
        "achieved_stage2_displacement": stage2(subset, arm, beta2),
        "displacement_at_beta_M1_under_M2": stage2(subset, arm, target),
        "shortfall_if_beta_reused": target - stage2(subset, arm, target),
        "cases_used": n,
        "ceiling_below_target_fraction": sat,
    }


solve_set = [c for c in cases if c["m2_split"] == "DEV_SOLVE"]
check_set = [c for c in cases if c["m2_split"] == "DEV_CHECK"]
print(f"SPLIT solve={len(solve_set)} check={len(check_set)}")

results, flags = {}, []
for arm in ARMS:
    s = solve(solve_set, arm)
    c = solve(check_set, arm)
    rel = abs(c["beta_M2"] - s["beta_M2"]) / s["beta_M2"]
    flagged = rel > TOL
    if flagged:
        flags.append({"arm": arm, "relative_discrepancy": rel, "tolerance": TOL})
    results[arm] = {
        "DEV_SOLVE": s,
        "DEV_CHECK_agreement_only": c,
        "relative_discrepancy": rel,
        "tolerance_band": TOL,
        "stability_flag": flagged,
        "frozen_beta_M2": s["beta_M2"],
        "frozen_from": "DEV_SOLVE",
        "check_changed_frozen_value": False,
    }
    print(f"{arm:10s} beta_M1={s['beta_M1_target']:8.4f}  beta_M2={s['beta_M2']:9.4f}  "
          f"ratio={s['beta_ratio_M2_over_M1']:.4f}  check={c['beta_M2']:9.4f}  "
          f"rel_disc={rel:.4f}{'  [FLAG]' if flagged else ''}")

OUT.mkdir(parents=True, exist_ok=True)
betas = {
    "schema": "GENERALIZED_SAFETY_V2_M2_BETAS_V1",
    "beta_V_DIJA": results["V_DIJA"]["frozen_beta_M2"],
    "beta_V_RENELLM": results["V_RENELLM"]["frozen_beta_M2"],
    "beta_V_ALL": results["V_ALL"]["frozen_beta_M2"],
    "derived_by": "displacement-matching on DEV_SOLVE per M2_BETA_SOLVING_PROCEDURE_V1",
    "m1_betas_reused": False,
    "sweep_used": False,
    "heldout_used": False,
    "m2_confirmatory_used": False,
    "outcome_labels_used": False,
}
bp = OUT / "M2_BETAS.json"
with open(bp, "w") as f:
    json.dump(betas, f, indent=2, sort_keys=True); f.write("\n")

ap = OUT / "M2_BETA_CALIBRATION_AUDIT.json"
with open(ap, "w") as f:
    json.dump({
        "schema": "GENERALIZED_SAFETY_V2_M2_BETA_CALIBRATION_AUDIT_V1",
        "status": "PASS",
        "solving_procedure_sha256": EXPECT_SP_SHA,
        "solving_procedure_frozen_before_measurements": True,
        "procedure_amended_after_seeing_numbers": False,
        "per_arm": results,
        "stability_flags": flags,
        "stability_flags_changed_any_frozen_beta": False,
        "source_cases": len(cases),
        "total_observations": int(sum(c["observations"] for c in cases)),
        "eps": EPS,
        "solver": sp["solver"],
    }, f, indent=2, sort_keys=True); f.write("\n")

fp = OUT / "M2_BETA_FREEZE.json"
with open(fp, "w") as f:
    json.dump({
        "schema": "GENERALIZED_SAFETY_V2_M2_BETA_FREEZE_V1",
        "status": "PASS_FROZEN",
        "contract_item": 7,
        "betas_sha256": sha256_file(bp),
        "audit_sha256": sha256_file(ap),
        "beta_values": {k: betas[f"beta_{k}"] for k in ARMS},
        "betas_frozen": True,
        "generation_authorized": False,
        "next_state": "M2_BETAS_FROZEN_AWAITING_CONFIRMATORY_VARIANT_CONSTRUCTION",
    }, f, indent=2, sort_keys=True); f.write("\n")

print()
print("M2_BETAS.json                 ", sha256_file(bp))
print("M2_BETA_CALIBRATION_AUDIT.json", sha256_file(ap))
print("M2_BETA_FREEZE.json           ", sha256_file(fp))
print("SOLVE_STATUS=PASS", "FLAGS=" + (str(len(flags)) if flags else "0"))
