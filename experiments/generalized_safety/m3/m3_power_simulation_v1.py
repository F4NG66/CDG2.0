"""
M3 pipeline step 2 — pre-registered power simulation.

Powers the ACTUAL decision rule, not one nominal-alpha test in isolation:

  4 pre-registered p-values = 2 co-primary comparisons x 2 attack families
  Holm correction applied across all 4 jointly
  A family counts as a success only if BOTH of its comparisons
    (a) show a positive point improvement, AND
    (b) pass their Holm-adjusted threshold

Because Holm is joint across all four tests, a family's power depends on the
other family's p-values too. That coupling is why this is simulated rather than
taken from a closed-form single-test formula.

Inputs are M1/M2 AGGREGATE results only (handoff §33 / §39B and the frozen M2
paired contrast). No case-level M1 or M2 outcome is read anywhere in this file.

Test: exact McNemar, two-sided, as frozen in the stats plan.
"""

import argparse
import json

import numpy as np
from scipy.stats import binom

ALPHA = 0.05
N_TESTS = 4
COMPARISONS = ("A_vs_M1_ADD", "B_vs_MATCHED_STATIC")
FAMILIES = ("DIJA", "RENELLM")


# --------------------------------------------------------------- exact McNemar
def build_pvalue_table(max_d):
    """p-value for every reachable (d, b): exact two-sided McNemar.

    b = TGAS success & comparator failure; c = d - b the reverse.
    Under H0, b ~ Bin(d, 0.5), so p = 2 * CDF(min(b, c); d, 0.5), capped at 1.
    d = 0 carries no directional information, so p = 1.

    Vectorised over b for each d; the previous exact-comb version was O(d^3)
    overall and did not finish.
    """
    table = {0: np.array([1.0])}
    for d in range(1, max_d + 1):
        b = np.arange(d + 1)
        table[d] = np.minimum(1.0, 2.0 * binom.cdf(np.minimum(b, d - b), d, 0.5))
    return table


def holm_pass_matrix(pvals, alpha=ALPHA):
    """Holm-Bonferroni across the m tests of each row, jointly. Vectorised.

    Step-down: sort ascending, compare the i-th smallest against alpha/(m-i);
    the first failure stops all further rejections in that row.
    """
    n, m = pvals.shape
    order = np.argsort(pvals, axis=1)
    ordered = np.take_along_axis(pvals, order, axis=1)
    thresh = alpha / (m - np.arange(m))
    ok_sorted = ordered <= thresh
    # a test is rejected only if it and every smaller-p test passed
    ok_sorted = np.logical_and.accumulate(ok_sorted, axis=1)
    out = np.zeros_like(ok_sorted)
    np.put_along_axis(out, order, ok_sorted, axis=1)
    return out


# --------------------------------------------------------------- simulation
def simulate(n_by_family, d_rate, effect_pp, n_sims, rng, ptab):
    """Return per-family compound power and the both-families power.

    d_rate[family][comparison]  assumed discordant-pair rate
    effect_pp[comparison]       minimum meaningful improvement, percentage points

    Given a discordant rate r and a target improvement delta (as a fraction),
    the probability that a discordant pair favours TGAS is
        pi = 0.5 * (1 + delta / r)
    since  delta = r * (2*pi - 1).  pi > 1 means the requested effect is
    arithmetically unreachable at that discordant rate.
    """
    keys = [(f, c) for f in FAMILIES for c in COMPARISONS]
    pi = {}
    for f, c in keys:
        r = d_rate[f][c]
        delta = effect_pp[c] / 100.0
        val = 0.5 * (1.0 + delta / r)
        if val > 1.0:
            return None, None, {"unreachable": f"{f}/{c}", "required_pi": val, "d_rate": r}
        pi[(f, c)] = val

    pvals = np.empty((n_sims, N_TESTS))
    positive = np.empty((n_sims, N_TESTS), dtype=bool)

    for j, (f, c) in enumerate(keys):
        n = n_by_family[f]
        d = rng.binomial(n, d_rate[f][c], size=n_sims)
        b = rng.binomial(d, pi[(f, c)])
        cc = d - b
        positive[:, j] = (b - cc) > 0
        col = np.empty(n_sims)
        for dv in np.unique(d):
            m = d == dv
            col[m] = ptab[int(dv)][b[m]]
        pvals[:, j] = col

    passed = holm_pass_matrix(pvals)
    ok_test = passed & positive

    fam_ok = {}
    for f in FAMILIES:
        idx = [keys.index((f, c)) for c in COMPARISONS]
        fam_ok[f] = ok_test[:, idx].all(axis=1)

    both = np.logical_and.reduce([fam_ok[f] for f in FAMILIES])
    return ({f: float(fam_ok[f].mean()) for f in FAMILIES},
            float(both.mean()),
            None)


def required_raw_groups(n_strict, yields):
    return {f: int(np.ceil(n_strict[f] / yields[f])) for f in FAMILIES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--power-target", type=float, default=0.80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- frozen assumptions (aggregate-derived; see ASSUMPTIONS.md) ----
    # Discordant-pair rates for strict_paired_success, observed in the frozen M2
    # paired contrast between two steering arms on the SAME population.
    # V_ALL cells are the anchors because M3 steers V_ALL only.
    d_rate = {
        "DIJA":    {"A_vs_M1_ADD": 0.1111, "B_vs_MATCHED_STATIC": 0.1111},
        "RENELLM": {"A_vs_M1_ADD": 0.2703, "B_vs_MATCHED_STATIC": 0.2703},
    }
    # Attrition chain: raw groups -> constructed groups -> strict-eligible pairs.
    #
    #   M1 held-out (100 groups, 80/20 health):
    #     construction  DIJA 100/100 = 1.000 ; ReNeLLM 98/100 = 0.980 (pairs 324,424 exhausted)
    #     baseline-qual DIJA  83/100 = 0.830 ; ReNeLLM 61/98  = 0.622
    #   M2 confirmatory (126 groups, 35/65 health):
    #     construction  DIJA 115/126 = 0.913 ; ReNeLLM 102/126 = 0.810
    #     baseline-qual DIJA  54/115 = 0.470 ; ReNeLLM  37/102 = 0.363
    #
    # Baseline qualification is strongly composition-dependent (that is the stated
    # reason M2's is so much lower), and M3 restores M1's 80/20 composition — so M1's
    # qualification rates are the composition-matched estimate. Construction attrition
    # is tooling, not composition, so M2's more recent experience is the better
    # estimate there. PRIMARY = that blend. The two pure scenarios bracket it.
    yields_m1 = {"DIJA": 1.000 * 0.830, "RENELLM": 0.980 * 0.622}
    yields_m2 = {"DIJA": 0.913 * 0.470, "RENELLM": 0.810 * 0.363}
    yields_blend = {"DIJA": 0.913 * 0.830, "RENELLM": 0.810 * 0.622}

    results = {
        "schema": "M3_POWER_SIMULATION_V1",
        "alpha": ALPHA,
        "test": "exact McNemar, two-sided",
        "multiple_comparisons": "Holm across all 4 pre-registered p-values jointly",
        "power_target_definition": (
            "per-family COMPOUND criterion: both co-primary comparisons show a "
            "positive point improvement AND both pass their Holm-adjusted threshold"
        ),
        "power_target": args.power_target,
        "n_sims": args.sims,
        "seed": args.seed,
        "discordant_pair_rates": d_rate,
        "discordant_rate_provenance": (
            "M2 frozen paired contrast, strict_paired_success, V_ALL cells: "
            "DIJA m1_only=3 + m2_only=3 = 6/54 = 11.11%; "
            "RENELLM m1_only=8 + m2_only=2 = 10/37 = 27.03%. AGGREGATE ONLY."
        ),
        "yield_scenarios": {
            "PRIMARY_blend_M2construction_M1qualification": yields_blend,
            "bound_optimistic_M1_pure_80_20": yields_m1,
            "bound_conservative_M2_pure_35_65": yields_m2,
        },
        "sweeps": [],
    }

    max_n = 5000
    ptab = build_pvalue_table(max_n)

    for effect in (5.0, 7.5, 10.0, 12.5, 15.0):
        eff = {c: effect for c in COMPARISONS}
        entry = {"minimum_meaningful_effect_pp": effect, "by_n_strict": []}
        found = None
        for n_strict_common in list(range(20, 601, 10)) + list(range(650, 3001, 50)):
            n_by_family = {f: n_strict_common for f in FAMILIES}
            pw, both, err = simulate(n_by_family, d_rate, eff, args.sims, rng, ptab)
            if err:
                entry["unreachable"] = err
                break
            rec = {
                "n_strict_eligible_per_family": n_strict_common,
                "power_DIJA": round(pw["DIJA"], 4),
                "power_RENELLM": round(pw["RENELLM"], 4),
                "power_both_families": round(both, 4),
            }
            entry["by_n_strict"].append(rec)
            if found is None and min(pw.values()) >= args.power_target:
                found = rec
                entry["n_strict_meeting_target_both_families"] = n_strict_common
                entry["required_raw_groups_PRIMARY_blend"] = required_raw_groups(
                    n_by_family, yields_blend)
                entry["required_raw_groups_optimistic_M1"] = required_raw_groups(
                    n_by_family, yields_m1)
                entry["required_raw_groups_conservative_M2"] = required_raw_groups(
                    n_by_family, yields_m2)
                break
        if found is None and "unreachable" not in entry:
            entry["n_strict_meeting_target_both_families"] = None
            entry["note"] = "power target not reached within simulated range"
        results["sweeps"].append(entry)

    # ---- structural ceiling on detectable effect ----
    # delta = r * (2*pi - 1) and pi <= 1, so the largest improvement that can exist
    # at discordant rate r is r itself (every discordant pair favouring TGAS).
    ceiling = {f: round(100 * min(d_rate[f][c] for c in COMPARISONS), 2)
               for f in FAMILIES}
    results["arithmetic_effect_ceiling_pp"] = {
        "by_family": ceiling,
        "binding_family": min(ceiling, key=ceiling.get),
        "binding_ceiling_pp": min(ceiling.values()),
        "explanation": (
            "At discordant-pair rate r, the maximum possible strict_paired_success "
            "improvement is r (100% of discordant pairs favouring TGAS). Any "
            "minimum-meaningful-effect above the binding ceiling is not merely "
            "underpowered, it is arithmetically unreachable at the assumed rate."
        ),
    }

    # ---- sensitivity: the assumed discordant rate is itself uncertain ----
    # Power scales roughly as delta*sqrt(n/r), so a LOWER assumed r is OPTIMISTIC.
    # M2-vs-M1 discordance may understate M1-vs-TGAS discordance if TGAS departs
    # from M1 more than M2 did, which would raise required N.
    sens = []
    for mult in (0.75, 1.0, 1.5, 2.0):
        dr = {f: {c: min(0.95, d_rate[f][c] * mult) for c in COMPARISONS}
              for f in FAMILIES}
        eff = {c: 10.0 for c in COMPARISONS}
        rec = {"discordant_rate_multiplier": mult,
               "effective_rates": {f: round(dr[f]["A_vs_M1_ADD"], 4) for f in FAMILIES},
               "minimum_meaningful_effect_pp": 10.0}
        hit = None
        for n in list(range(50, 1001, 25)) + list(range(1100, 5001, 100)):
            pw, both, err = simulate({f: n for f in FAMILIES}, dr, eff,
                                     max(4000, args.sims // 4), rng, ptab)
            if err:
                rec["unreachable"] = err
                break
            if min(pw.values()) >= args.power_target:
                hit = n
                break
        if "unreachable" not in rec:
            rec["n_strict_meeting_target"] = hit
            if hit:
                rec["required_raw_groups_PRIMARY_blend"] = required_raw_groups(
                    {f: hit for f in FAMILIES}, yields_blend)
                rec["required_raw_groups_conservative_M2"] = required_raw_groups(
                    {f: hit for f in FAMILIES}, yields_m2)
        sens.append(rec)
    results["discordant_rate_sensitivity"] = sens

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
