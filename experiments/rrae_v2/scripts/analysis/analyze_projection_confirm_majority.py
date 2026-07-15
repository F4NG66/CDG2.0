#!/usr/bin/env python

import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


WORK_ROOT = Path("/path/to/rrae_steering_work_v2")

JUDGE_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/judges/"
    "projection_removal_confirm_majority"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/analysis/"
    "projection_removal_confirm_majority"
)

PASSES = [1, 2, 3]
TARGETS = ["baseline", "L11", "L16"]


def read_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def canonical_key(row, target):
    """
    Unique key for the same generated output across judge passes.

    Baseline:
        one output per group/case.

    Steered:
        one output per group/case/layer/lambda.
    """
    base = (
        row["position_config"],
        row["case_id"],
        row["variant"],
        row["group"],
        float(row["alpha"]),
        row["mode"],
        int(row.get("seed", 0)),
        row["judge_template"],
    )

    return (target,) + base


def exact_binomial_two_sided(k, n):
    """
    Exact two-sided binomial p-value under p=0.5.

    Used for McNemar exact test:
        k = min(b, c)
        n = b + c
    """
    if n == 0:
        return 1.0

    def pmf(i):
        return math.comb(n, i) * (0.5 ** n)

    observed = pmf(k)

    p = sum(
        pmf(i)
        for i in range(n + 1)
        if pmf(i) <= observed + 1e-15
    )

    return min(1.0, p)


def build_majority_table(target):
    pass_maps = {}

    for judge_pass in PASSES:
        path = (
            JUDGE_ROOT
            / f"{target}_pass{judge_pass}"
            / "judged__template_injection.jsonl"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        rows = read_jsonl(path)

        current = {}

        for row in rows:
            key = canonical_key(row, target)

            if key in current:
                raise RuntimeError(
                    f"Duplicate key in {target} pass {judge_pass}: {key}"
                )

            current[key] = row

        pass_maps[judge_pass] = current

    reference_keys = set(pass_maps[1])

    for judge_pass in PASSES[1:]:
        other_keys = set(pass_maps[judge_pass])

        if other_keys != reference_keys:
            missing = reference_keys - other_keys
            extra = other_keys - reference_keys

            raise RuntimeError(
                f"Key mismatch for {target} pass {judge_pass}. "
                f"missing={len(missing)}, extra={len(extra)}"
            )

    output_rows = []

    for key in sorted(reference_keys, key=str):
        source = pass_maps[1][key]

        votes = [
            int(pass_maps[p][key]["success"])
            for p in PASSES
        ]

        labels = [
            pass_maps[p][key].get("label")
            for p in PASSES
        ]

        majority = int(sum(votes) >= 2)
        unanimous = len(set(votes)) == 1
        vote_pattern = "".join(str(v) for v in votes)

        output_rows.append({
            "target": target,
            "layer": (
                0
                if target == "baseline"
                else int(target.replace("L", ""))
            ),
            "position_config": source["position_config"],
            "case_id": source["case_id"],
            "variant": source["variant"],
            "group": source["group"],
            "alpha": float(source["alpha"]),
            "mode": source["mode"],
            "seed": int(source.get("seed", 0)),
            "judge_template": source["judge_template"],
            "success_pass1": votes[0],
            "success_pass2": votes[1],
            "success_pass3": votes[2],
            "success_majority": majority,
            "unanimous": unanimous,
            "vote_pattern": vote_pattern,
            "label_pass1": labels[0],
            "label_pass2": labels[1],
            "label_pass3": labels[2],
        })

    return pd.DataFrame(output_rows)


def agreement_summary(df):
    rows = []

    for target, part in df.groupby("target", sort=False):
        n = len(part)

        unanimous_n = int(part["unanimous"].sum())
        split_n = n - unanimous_n

        pair_12 = (
            part["success_pass1"]
            == part["success_pass2"]
        ).mean()

        pair_13 = (
            part["success_pass1"]
            == part["success_pass3"]
        ).mean()

        pair_23 = (
            part["success_pass2"]
            == part["success_pass3"]
        ).mean()

        rows.append({
            "target": target,
            "n": n,
            "unanimous_n": unanimous_n,
            "unanimous_rate": unanimous_n / n,
            "split_2_vs_1_n": split_n,
            "split_2_vs_1_rate": split_n / n,
            "agreement_pass1_pass2": pair_12,
            "agreement_pass1_pass3": pair_13,
            "agreement_pass2_pass3": pair_23,
            "mean_pairwise_agreement": (
                pair_12 + pair_13 + pair_23
            ) / 3,
        })

    return pd.DataFrame(rows)


def majority_rate_summary(df):
    grouped = (
        df.groupby(
            ["target", "layer", "group", "alpha", "mode"],
            dropna=False,
        )
        .agg(
            n=("success_majority", "size"),
            success_n=("success_majority", "sum"),
            success_rate=("success_majority", "mean"),
            unanimous_rate=("unanimous", "mean"),
        )
        .reset_index()
    )

    return grouped


def paired_transition_analysis(df):
    baseline = df[df["target"] == "baseline"].copy()
    steered = df[df["target"].isin(["L11", "L16"])].copy()

    baseline_map = {
        (row.group, row.case_id): int(row.success_majority)
        for row in baseline.itertuples()
    }

    rows = []
    paired_detail = []

    for (target, layer, group, alpha), part in steered.groupby(
        ["target", "layer", "group", "alpha"],
        sort=True,
    ):
        counts = Counter()
        paired_values = []

        for row in part.itertuples():
            key = (row.group, row.case_id)

            if key not in baseline_map:
                raise RuntimeError(
                    f"Missing baseline for {key}"
                )

            base_success = baseline_map[key]
            steered_success = int(row.success_majority)

            if base_success == 1 and steered_success == 0:
                transition = "success_to_failure"
            elif base_success == 0 and steered_success == 1:
                transition = "failure_to_success"
            elif base_success == 1 and steered_success == 1:
                transition = "success_to_success"
            else:
                transition = "failure_to_failure"

            counts[transition] += 1
            paired_values.append((base_success, steered_success))

            paired_detail.append({
                "target": target,
                "layer": layer,
                "group": group,
                "alpha": alpha,
                "case_id": row.case_id,
                "baseline_success": base_success,
                "steered_success": steered_success,
                "transition": transition,
                "steered_vote_pattern": row.vote_pattern,
                "steered_unanimous": row.unanimous,
            })

        n = len(paired_values)

        baseline_success_n = sum(x for x, _ in paired_values)
        steered_success_n = sum(y for _, y in paired_values)

        success_to_failure = counts["success_to_failure"]
        failure_to_success = counts["failure_to_success"]

        net_success_reduction = (
            success_to_failure - failure_to_success
        )

        baseline_rate = baseline_success_n / n
        steered_rate = steered_success_n / n
        delta = steered_rate - baseline_rate

        discordant = success_to_failure + failure_to_success

        p_value = exact_binomial_two_sided(
            min(success_to_failure, failure_to_success),
            discordant,
        )

        rows.append({
            "target": target,
            "layer": layer,
            "group": group,
            "alpha": alpha,
            "n": n,
            "baseline_success_n": baseline_success_n,
            "baseline_success_rate": baseline_rate,
            "steered_success_n": steered_success_n,
            "steered_success_rate": steered_rate,
            "delta_success_rate": delta,
            "delta_percentage_points": delta * 100,
            "success_to_failure": success_to_failure,
            "failure_to_success": failure_to_success,
            "success_to_success": counts["success_to_success"],
            "failure_to_failure": counts["failure_to_failure"],
            "net_success_reduction_count": net_success_reduction,
            "net_success_reduction_rate": (
                net_success_reduction / n
            ),
            "discordant_pairs": discordant,
            "mcnemar_exact_p": p_value,
        })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(paired_detail),
    )


def specificity_summary(paired_df):
    """
    B is harmful injected:
        lower success is desirable.

    C is benign injected:
        preserving success is generally desirable.

    Therefore:
        harmful_reduction = baseline_B - steered_B
        benign_retention  = steered_C / baseline_C
        specificity_gap   = reduction_B - reduction_C
    """
    rows = []

    for (target, layer, alpha), part in paired_df.groupby(
        ["target", "layer", "alpha"],
        sort=True,
    ):
        by_group = {
            row.group: row
            for row in part.itertuples()
        }

        if "B" not in by_group or "C" not in by_group:
            continue

        b = by_group["B"]
        c = by_group["C"]

        b_reduction = (
            b.baseline_success_rate
            - b.steered_success_rate
        )

        c_reduction = (
            c.baseline_success_rate
            - c.steered_success_rate
        )

        benign_retention = (
            c.steered_success_rate
            / c.baseline_success_rate
            if c.baseline_success_rate > 0
            else float("nan")
        )

        rows.append({
            "target": target,
            "layer": layer,
            "alpha": alpha,
            "harmful_B_baseline_rate": b.baseline_success_rate,
            "harmful_B_steered_rate": b.steered_success_rate,
            "harmful_B_reduction": b_reduction,
            "benign_C_baseline_rate": c.baseline_success_rate,
            "benign_C_steered_rate": c.steered_success_rate,
            "benign_C_reduction": c_reduction,
            "benign_C_retention_ratio": benign_retention,
            "specificity_gap": b_reduction - c_reduction,
            "B_mcnemar_p": b.mcnemar_exact_p,
            "C_mcnemar_p": c.mcnemar_exact_p,
        })

    return pd.DataFrame(rows)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_majority = []

    for target in TARGETS:
        df = build_majority_table(target)
        all_majority.append(df)

        print(
            f"{target}: rows={len(df)}, "
            f"unanimous={df['unanimous'].mean():.3f}"
        )

    majority_df = pd.concat(
        all_majority,
        ignore_index=True,
    )

    agreement_df = agreement_summary(majority_df)
    rate_df = majority_rate_summary(majority_df)

    paired_df, paired_detail_df = paired_transition_analysis(
        majority_df
    )

    specificity_df = specificity_summary(paired_df)

    majority_path = OUT_ROOT / "majority_votes_all_rows.csv"
    agreement_path = OUT_ROOT / "judge_agreement_summary.csv"
    rate_path = OUT_ROOT / "majority_success_rates.csv"
    paired_path = OUT_ROOT / "paired_transition_summary.csv"
    detail_path = OUT_ROOT / "paired_transition_details.csv"
    specificity_path = OUT_ROOT / "specificity_summary.csv"

    majority_df.to_csv(majority_path, index=False)
    agreement_df.to_csv(agreement_path, index=False)
    rate_df.to_csv(rate_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    paired_detail_df.to_csv(detail_path, index=False)
    specificity_df.to_csv(specificity_path, index=False)

    print("\n===== JUDGE AGREEMENT =====")
    print(
        agreement_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n===== MAJORITY SUCCESS RATES =====")
    print(
        rate_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n===== PAIRED TRANSITIONS =====")
    print(
        paired_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n===== SPECIFICITY SUMMARY =====")
    print(
        specificity_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved:")
    for path in [
        majority_path,
        agreement_path,
        rate_path,
        paired_path,
        detail_path,
        specificity_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
