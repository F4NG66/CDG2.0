#!/usr/bin/env python

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


WORK_ROOT = Path("/path/to/rrae_steering_work_v2")

JUDGE_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/judges/"
    "p11_best_L11_lambda2"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/analysis/"
    "p11_best_L11_lambda2"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)


B_FILES = [
    JUDGE_ROOT
    / f"B_p11_pass{p}/"
    "judged__p11_valence_specificity_collapse.jsonl"
    for p in (1, 2, 3)
]

C_FILES = [
    JUDGE_ROOT
    / f"C_utility_pass{p}/"
    "judged__benign_utility_collapse.jsonl"
    for p in (1, 2, 3)
]


def load_jsonl(path, pass_id, evaluation):
    rows = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)
            judge = row.get("judge") or {}

            record = {
                "evaluation": evaluation,
                "pass_id": pass_id,
                "position_config": row.get("position_config"),
                "case_id": row.get("case_id"),
                "variant": row.get("variant"),
                "group": row.get("group"),
                "mode": row.get("mode"),
                "alpha": float(row.get("alpha")),
                "seed": str(row.get("seed")),
                "behavior": row.get("behavior", ""),
                "output": row.get("output", ""),
                "response_length": len(row.get("output", "") or ""),
                "reasoning": judge.get("reasoning"),
            }

            if evaluation == "B":
                record.update({
                    "category": judge.get("valence_category"),
                    "score_1": judge.get("valence_score"),
                    "score_2": judge.get("specificity_score"),
                    "collapse_score": judge.get("collapse_score"),
                })

            else:
                record.update({
                    "category": judge.get("outcome_category"),
                    "score_1": judge.get("task_completion_score"),
                    "score_2": judge.get("relevance_score"),
                    "score_3": judge.get(
                        "unnecessary_refusal_score"
                    ),
                    "collapse_score": judge.get("collapse_score"),
                })

            rows.append(record)

    return rows


def majority_vote(values):
    values = [
        value
        for value in values
        if value is not None and str(value) != "nan"
    ]

    if not values:
        return None

    counts = Counter(values)
    highest = max(counts.values())
    winners = sorted(
        value
        for value, count in counts.items()
        if count == highest
    )

    return winners[0]


def agreement_type(values):
    values = [v for v in values if v is not None]

    if len(values) != 3:
        return "incomplete"

    unique = len(set(values))

    if unique == 1:
        return "unanimous"
    if unique == 2:
        return "majority_2_of_3"

    return "three_way_disagreement"


def safe_median(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.median())


def aggregate_outputs(df, evaluation):
    keys = [
        "evaluation",
        "position_config",
        "case_id",
        "variant",
        "group",
        "mode",
        "alpha",
        "seed",
    ]

    records = []

    for keys_value, group_df in df.groupby(
        keys,
        dropna=False,
        sort=True,
    ):
        base = dict(zip(keys, keys_value))

        categories = group_df["category"].tolist()

        record = {
            **base,
            "behavior": group_df["behavior"].iloc[0],
            "output": group_df["output"].iloc[0],
            "response_length": int(
                group_df["response_length"].iloc[0]
            ),
            "n_passes": len(group_df),
            "majority_category": majority_vote(categories),
            "category_agreement": agreement_type(categories),
            "median_collapse_score": safe_median(
                group_df["collapse_score"]
            ),
            "mean_collapse_score": pd.to_numeric(
                group_df["collapse_score"],
                errors="coerce",
            ).mean(),
            "judge_reasonings": " || ".join(
                str(x)
                for x in group_df["reasoning"].tolist()
            ),
        }

        if evaluation == "B":
            record.update({
                "median_valence_score": safe_median(
                    group_df["score_1"]
                ),
                "median_specificity_score": safe_median(
                    group_df["score_2"]
                ),
            })

        else:
            record.update({
                "median_task_completion_score": safe_median(
                    group_df["score_1"]
                ),
                "median_relevance_score": safe_median(
                    group_df["score_2"]
                ),
                "median_unnecessary_refusal_score": safe_median(
                    group_df["score_3"]
                ),
            })

        records.append(record)

    return pd.DataFrame(records)


def summarize_agreement(agg_df, evaluation):
    counts = (
        agg_df["category_agreement"]
        .value_counts(dropna=False)
        .to_dict()
    )

    n = len(agg_df)

    row = {
        "evaluation": evaluation,
        "n_outputs": n,
        "unanimous_count": counts.get("unanimous", 0),
        "majority_2_of_3_count": counts.get(
            "majority_2_of_3",
            0,
        ),
        "three_way_disagreement_count": counts.get(
            "three_way_disagreement",
            0,
        ),
        "incomplete_count": counts.get("incomplete", 0),
    }

    for key in [
        "unanimous",
        "majority_2_of_3",
        "three_way_disagreement",
        "incomplete",
    ]:
        row[f"{key}_rate"] = (
            counts.get(key, 0) / n
            if n
            else np.nan
        )

    return pd.DataFrame([row])


def summarize_groups(agg_df, evaluation):
    rows = []

    for keys, group_df in agg_df.groupby(
        ["group", "mode", "alpha"],
        sort=True,
    ):
        group, mode, alpha = keys

        row = {
            "evaluation": evaluation,
            "group": group,
            "mode": mode,
            "alpha": alpha,
            "n": len(group_df),
            "mean_response_length": group_df[
                "response_length"
            ].mean(),
            "median_collapse_score": group_df[
                "median_collapse_score"
            ].median(),
            "mean_collapse_score": group_df[
                "median_collapse_score"
            ].mean(),
        }

        category_counts = (
            group_df["majority_category"]
            .value_counts(dropna=False)
            .to_dict()
        )

        for category, count in category_counts.items():
            row[f"category__{category}"] = count
            row[f"rate__{category}"] = count / len(group_df)

        if evaluation == "B":
            row.update({
                "median_valence_score": group_df[
                    "median_valence_score"
                ].median(),
                "mean_valence_score": group_df[
                    "median_valence_score"
                ].mean(),
                "median_specificity_score": group_df[
                    "median_specificity_score"
                ].median(),
                "mean_specificity_score": group_df[
                    "median_specificity_score"
                ].mean(),
            })

        else:
            row.update({
                "median_task_completion_score": group_df[
                    "median_task_completion_score"
                ].median(),
                "mean_task_completion_score": group_df[
                    "median_task_completion_score"
                ].mean(),
                "median_relevance_score": group_df[
                    "median_relevance_score"
                ].median(),
                "mean_relevance_score": group_df[
                    "median_relevance_score"
                ].mean(),
                "median_unnecessary_refusal_score": group_df[
                    "median_unnecessary_refusal_score"
                ].median(),
                "mean_unnecessary_refusal_score": group_df[
                    "median_unnecessary_refusal_score"
                ].mean(),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def make_paired(agg_df, evaluation):
    baseline = agg_df[
        (agg_df["mode"] == "baseline")
        & (agg_df["alpha"] == 0.0)
    ].copy()

    steered = agg_df[
        (agg_df["mode"] == "steered")
        & (agg_df["alpha"] == 2.0)
    ].copy()

    pair_keys = [
        "position_config",
        "case_id",
        "variant",
        "group",
        "seed",
    ]

    base_columns = pair_keys + [
        "majority_category",
        "median_collapse_score",
        "response_length",
    ]

    steer_columns = pair_keys + [
        "majority_category",
        "median_collapse_score",
        "response_length",
    ]

    if evaluation == "B":
        base_columns += [
            "median_valence_score",
            "median_specificity_score",
        ]
        steer_columns += [
            "median_valence_score",
            "median_specificity_score",
        ]

    else:
        base_columns += [
            "median_task_completion_score",
            "median_relevance_score",
            "median_unnecessary_refusal_score",
        ]
        steer_columns += [
            "median_task_completion_score",
            "median_relevance_score",
            "median_unnecessary_refusal_score",
        ]

    paired = baseline[base_columns].merge(
        steered[steer_columns],
        on=pair_keys,
        how="inner",
        suffixes=("_baseline", "_steered"),
        validate="one_to_one",
    )

    paired["category_transition"] = (
        paired["majority_category_baseline"]
        .fillna("None")
        + " -> "
        + paired["majority_category_steered"]
        .fillna("None")
    )

    paired["collapse_delta"] = (
        paired["median_collapse_score_steered"]
        - paired["median_collapse_score_baseline"]
    )

    paired["response_length_delta"] = (
        paired["response_length_steered"]
        - paired["response_length_baseline"]
    )

    if evaluation == "B":
        paired["valence_delta"] = (
            paired["median_valence_score_steered"]
            - paired["median_valence_score_baseline"]
        )
        paired["specificity_delta"] = (
            paired["median_specificity_score_steered"]
            - paired["median_specificity_score_baseline"]
        )

    else:
        paired["task_completion_delta"] = (
            paired["median_task_completion_score_steered"]
            - paired["median_task_completion_score_baseline"]
        )
        paired["relevance_delta"] = (
            paired["median_relevance_score_steered"]
            - paired["median_relevance_score_baseline"]
        )
        paired["unnecessary_refusal_delta"] = (
            paired[
                "median_unnecessary_refusal_score_steered"
            ]
            - paired[
                "median_unnecessary_refusal_score_baseline"
            ]
        )

    return paired


def summarize_paired(paired, evaluation):
    rows = []

    for group, group_df in paired.groupby("group"):
        row = {
            "evaluation": evaluation,
            "group": group,
            "n_pairs": len(group_df),
            "mean_collapse_delta": group_df[
                "collapse_delta"
            ].mean(),
            "median_collapse_delta": group_df[
                "collapse_delta"
            ].median(),
            "mean_response_length_delta": group_df[
                "response_length_delta"
            ].mean(),
        }

        if evaluation == "B":
            row.update({
                "mean_valence_delta": group_df[
                    "valence_delta"
                ].mean(),
                "median_valence_delta": group_df[
                    "valence_delta"
                ].median(),
                "mean_specificity_delta": group_df[
                    "specificity_delta"
                ].mean(),
                "median_specificity_delta": group_df[
                    "specificity_delta"
                ].median(),
            })

        else:
            row.update({
                "mean_task_completion_delta": group_df[
                    "task_completion_delta"
                ].mean(),
                "median_task_completion_delta": group_df[
                    "task_completion_delta"
                ].median(),
                "mean_relevance_delta": group_df[
                    "relevance_delta"
                ].mean(),
                "median_relevance_delta": group_df[
                    "relevance_delta"
                ].median(),
                "mean_unnecessary_refusal_delta": group_df[
                    "unnecessary_refusal_delta"
                ].mean(),
                "median_unnecessary_refusal_delta": group_df[
                    "unnecessary_refusal_delta"
                ].median(),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def transition_summary(paired, evaluation):
    result = (
        paired.groupby(
            ["group", "category_transition"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    result["evaluation"] = evaluation
    result["rate_within_group"] = result.groupby(
        "group"
    )["count"].transform(
        lambda x: x / x.sum()
    )

    return result[
        [
            "evaluation",
            "group",
            "category_transition",
            "count",
            "rate_within_group",
        ]
    ].sort_values(
        ["group", "count"],
        ascending=[True, False],
    )


def extract_high_collapse(agg_df, evaluation):
    result = agg_df[
        (agg_df["mode"] == "steered")
        & (agg_df["alpha"] == 2.0)
    ].copy()

    result = result.sort_values(
        [
            "median_collapse_score",
            "case_id",
        ],
        ascending=[False, True],
    )

    return result.head(30)


def main():
    all_b = []
    all_c = []

    for pass_id, path in enumerate(B_FILES, 1):
        if not path.exists():
            raise FileNotFoundError(path)

        all_b.extend(
            load_jsonl(
                path,
                pass_id=pass_id,
                evaluation="B",
            )
        )

    for pass_id, path in enumerate(C_FILES, 1):
        if not path.exists():
            raise FileNotFoundError(path)

        all_c.extend(
            load_jsonl(
                path,
                pass_id=pass_id,
                evaluation="C",
            )
        )

    b_raw = pd.DataFrame(all_b)
    c_raw = pd.DataFrame(all_c)

    print("B judgments:", len(b_raw))
    print("C judgments:", len(c_raw))

    b_agg = aggregate_outputs(b_raw, "B")
    c_agg = aggregate_outputs(c_raw, "C")

    print("B unique outputs:", len(b_agg))
    print("C unique outputs:", len(c_agg))

    b_pairs = make_paired(b_agg, "B")
    c_pairs = make_paired(c_agg, "C")

    print("B paired cases:", len(b_pairs))
    print("C paired cases:", len(c_pairs))

    outputs = {
        "B_majority_per_output.csv": b_agg,
        "C_majority_per_output.csv": c_agg,
        "B_agreement_summary.csv": summarize_agreement(
            b_agg,
            "B",
        ),
        "C_agreement_summary.csv": summarize_agreement(
            c_agg,
            "C",
        ),
        "B_group_summary.csv": summarize_groups(
            b_agg,
            "B",
        ),
        "C_group_summary.csv": summarize_groups(
            c_agg,
            "C",
        ),
        "B_paired_details.csv": b_pairs,
        "C_paired_details.csv": c_pairs,
        "B_paired_summary.csv": summarize_paired(
            b_pairs,
            "B",
        ),
        "C_paired_summary.csv": summarize_paired(
            c_pairs,
            "C",
        ),
        "B_transition_summary.csv": transition_summary(
            b_pairs,
            "B",
        ),
        "C_transition_summary.csv": transition_summary(
            c_pairs,
            "C",
        ),
        "B_high_collapse_examples.csv": extract_high_collapse(
            b_agg,
            "B",
        ),
        "C_high_collapse_examples.csv": extract_high_collapse(
            c_agg,
            "C",
        ),
    }

    for filename, dataframe in outputs.items():
        path = OUT_ROOT / filename
        dataframe.to_csv(path, index=False)
        print("saved:", path)

    print("\n===== B AGREEMENT =====")
    print(
        outputs["B_agreement_summary.csv"].to_string(
            index=False
        )
    )

    print("\n===== C AGREEMENT =====")
    print(
        outputs["C_agreement_summary.csv"].to_string(
            index=False
        )
    )

    print("\n===== B GROUP SUMMARY =====")
    print(
        outputs["B_group_summary.csv"].to_string(
            index=False
        )
    )

    print("\n===== C GROUP SUMMARY =====")
    print(
        outputs["C_group_summary.csv"].to_string(
            index=False
        )
    )

    print("\n===== B PAIRED SUMMARY =====")
    print(
        outputs["B_paired_summary.csv"].to_string(
            index=False
        )
    )

    print("\n===== C PAIRED SUMMARY =====")
    print(
        outputs["C_paired_summary.csv"].to_string(
            index=False
        )
    )

    print("\n===== TOP B TRANSITIONS =====")
    print(
        outputs["B_transition_summary.csv"]
        .head(20)
        .to_string(index=False)
    )

    print("\n===== TOP C TRANSITIONS =====")
    print(
        outputs["C_transition_summary.csv"]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
