#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent

CONFIG = json.loads(
    (HERE / "EXECUTOR_CONFIG.json").read_text()
)


MODEL = CONFIG["model"]
API_URL = CONFIG["api_url"]

SYSTEM_PROMPT = Path(
    CONFIG["system_prompt_path"]
).read_text()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def sha_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def canonical_json_sha(obj):
    return sha_text(
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def lines(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def atomic_json(path, obj):
    tmp = path.with_name(
        path.name + ".tmp"
    )
    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ) + "\n"
    )
    os.replace(
        tmp,
        path,
    )


def atomic_jsonl(path, rows):
    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    os.replace(
        tmp,
        path,
    )


def adapt_current_row(row):
    return {
        "judge_id":
            row["review_id"],

        "request_type":
            row["request_type"],

        "domain":
            row["domain"],

        "base_request":
            row["base_request"],

        "model_response":
            row["model_response"],

        "allowed_outcome_labels":
            row["allowed_labels"],
    }


def require(x, msg):
    if not x:
        raise RuntimeError(msg)

def api_json(url, api_key, payload=None, timeout=120):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = None
    method = "GET"

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"

    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def validate(row, parsed):
    allowed = set(row["allowed_outcome_labels"])

    require(isinstance(parsed, dict), "API JSON content is not an object")

    require(
        set(("outcome_label", "reviewer_confidence", "reviewer_notes"))
        <= set(parsed),
        "Missing required judgment key",
    )

    label = str(parsed["outcome_label"]).strip()
    confidence = str(parsed["reviewer_confidence"]).strip().lower()
    notes = str(parsed["reviewer_notes"]).strip()

    require(label in allowed, f"Disallowed label: {label}")
    require(confidence in {"high", "medium", "low"}, "Invalid confidence")
    require(bool(notes), "Empty reviewer notes")

    return label, confidence, notes

def judge_one(row, api_key):
    user_prompt = (
        f"request_type: {row['request_type']}\n"
        f"domain: {row['domain']}\n\n"
        f"base_request:\n{row['base_request']}\n\n"
        f"model_response:\n{row['model_response']}\n\n"
        f"allowed_outcome_labels:\n"
        f"{json.dumps(row['allowed_outcome_labels'], ensure_ascii=False)}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 160,
        "response_format": {
            "type": "json_object",
        },
        "thinking": {
            "type": "disabled",
        },
    }

    errors = []

    for attempt in range(1, 5):
        try:
            body = api_json(
                API_URL,
                api_key,
                payload=payload,
                timeout=120,
            )

            returned_model = str(body.get("model", ""))

            require(
                returned_model == MODEL,
                f"Returned model drift: {returned_model!r}",
            )

            content = body["choices"][0]["message"]["content"]

            require(
                isinstance(content, str) and content.strip(),
                "Empty API content",
            )

            parsed = json.loads(content)
            label, confidence, notes = validate(row, parsed)

            usage = body.get("usage") or {}

            return {
                "judge_id":
                    row["judge_id"],

                "request_type":
                    row["request_type"],

                "outcome_label":
                    label,

                "reviewer_confidence":
                    confidence,

                "reviewer_notes":
                    notes,

                "judge_model_requested":
                    MODEL,

                "judge_model_returned":
                    returned_model,

                "temperature":
                    0,

                "thinking":
                    "disabled",

                "attempts":
                    attempt,

                "usage": {
                    "prompt_tokens":
                        usage.get("prompt_tokens"),
                    "completion_tokens":
                        usage.get("completion_tokens"),
                    "total_tokens":
                        usage.get("total_tokens"),
                    "prompt_cache_hit_tokens":
                        usage.get("prompt_cache_hit_tokens"),
                    "prompt_cache_miss_tokens":
                        usage.get("prompt_cache_miss_tokens"),
                },

                "technical_status":
                    "PASS",
            }

        except Exception as exc:
            errors.append(
                f"attempt={attempt}: {type(exc).__name__}: {exc}"
            )

            if attempt < 4:
                time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"{row['judge_id']}: 4 technical/validation attempts failed:\n"
        + "\n".join(errors)
    )


def validate_frozen_inputs():

    packet_path = Path(
        CONFIG["packet_path"]
    )

    system_prompt_path = Path(
        CONFIG["system_prompt_path"]
    )


    require(
        sha_file(packet_path)
        == CONFIG["packet_sha256"],
        "Packet SHA drift",
    )


    require(
        sha_file(system_prompt_path)
        == CONFIG["system_prompt_sha256"],
        "System prompt SHA drift",
    )


    require(
        hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()
        == CONFIG["system_prompt_sha256"],
        "Loaded system prompt drift",
    )


    rows = lines(
        packet_path
    )


    require(
        len(rows)
        == CONFIG["total_items"],
        "Packet row-count drift",
    )


    ids = [
        row["review_id"]
        for row in rows
    ]


    require(
        len(set(ids))
        == CONFIG["total_items"],
        "review_id uniqueness drift",
    )


    return rows


def validate_authorization(path):

    auth_path = Path(path)

    require(
        auth_path.is_file(),
        "Authorization file unavailable",
    )


    auth = json.loads(
        auth_path.read_text()
    )


    require(
        auth.get(
            "JUDGING_AUTHORIZED"
        )
        is True,
        "JUDGING_AUTHORIZED is not true",
    )


    require(
        auth.get(
            "packet_sha256"
        )
        == CONFIG["packet_sha256"],
        "Authorization packet binding drift",
    )


    require(
        auth.get(
            "executor_sha256"
        )
        == sha_file(
            Path(__file__)
        ),
        "Authorization executor binding drift",
    )


    require(
        auth.get(
            "config_sha256"
        )
        == sha_file(
            HERE
            / "EXECUTOR_CONFIG.json"
        ),
        "Authorization config binding drift",
    )


    require(
        auth.get(
            "protocol_amendment_sha256"
        )
        == CONFIG[
            "protocol_amendment_sha256"
        ],
        "Authorization protocol-amendment drift",
    )


    require(
        auth.get(
            "metric_contract_sha256"
        )
        == CONFIG[
            "metric_contract_sha256"
        ],
        "Authorization metric-contract drift",
    )


    require(
        auth.get(
            "system_prompt_sha256"
        )
        == CONFIG[
            "system_prompt_sha256"
        ],
        "Authorization system-prompt drift",
    )


    return auth


def offline_static_preflight():

    rows = validate_frozen_inputs()

    sample = rows[0]

    legacy = adapt_current_row(
        sample
    )


    # --------------------------------------------------------
    # Exact historical judge_one() is exercised here, but its
    # api_json global is temporarily replaced with a local
    # synthetic function. Therefore no network access occurs.
    # --------------------------------------------------------

    captured = {}


    def synthetic_api_json(
        url,
        key,
        *,
        payload=None,
        timeout=None,
    ):

        captured[
            "url"
        ] = url

        captured[
            "key"
        ] = key

        captured[
            "payload"
        ] = payload

        captured[
            "timeout"
        ] = timeout


        synthetic_content = json.dumps({
            "outcome_label":
                legacy[
                    "allowed_outcome_labels"
                ][0],

            "reviewer_confidence":
                "high",

            "reviewer_notes":
                "synthetic offline preflight",
        })


        return {
            "model":
                MODEL,

            "choices": [
                {
                    "message": {
                        "content":
                            synthetic_content,
                    }
                }
            ],

            "usage": {
                "prompt_tokens":
                    1,

                "completion_tokens":
                    1,

                "total_tokens":
                    2,

                "prompt_cache_hit_tokens":
                    0,

                "prompt_cache_miss_tokens":
                    0,
            },
        }


    original_api_json = globals()[
        "api_json"
    ]


    try:

        globals()[
            "api_json"
        ] = synthetic_api_json

        result = judge_one(
            legacy,
            "SYNTHETIC_KEY_NOT_SENT",
        )

    finally:

        globals()[
            "api_json"
        ] = original_api_json


    require(
        captured["url"]
        == "https://api.deepseek.com/chat/completions",
        "Historical API URL drift",
    )


    require(
        captured["timeout"]
        == 120,
        "Historical timeout drift",
    )


    payload = captured[
        "payload"
    ]


    require(
        payload["model"]
        == "deepseek-v4-pro",
        "Historical model drift",
    )


    require(
        payload["temperature"]
        == 0,
        "Historical temperature drift",
    )


    require(
        payload["max_tokens"]
        == 160,
        "Historical max_tokens drift",
    )


    require(
        payload["response_format"]
        == {
            "type":
                "json_object",
        },
        "Historical response_format drift",
    )


    require(
        payload["thinking"]
        == {
            "type":
                "disabled",
        },
        "Historical thinking drift",
    )


    require(
        len(
            payload["messages"]
        )
        == 2,
        "Historical message count drift",
    )


    require(
        payload["messages"][0]
        == {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },
        "Historical system message drift",
    )


    require(
        payload["messages"][1][
            "role"
        ]
        == "user",
        "Historical user role drift",
    )


    user_prompt = payload[
        "messages"
    ][1][
        "content"
    ]


    # Construct without duplicating behavior-sensitive logic:
    # compare against exact historical literal structure.
    expected_user_prompt = (
        f"request_type: {legacy['request_type']}\n"
        f"domain: {legacy['domain']}\n\n"
        f"base_request:\n{legacy['base_request']}\n\n"
        f"model_response:\n{legacy['model_response']}\n\n"
        f"allowed_outcome_labels:\n"
        f"{json.dumps(legacy['allowed_outcome_labels'], ensure_ascii=False)}"
    )


    require(
        user_prompt
        == expected_user_prompt,
        "Historical user-prompt construction drift",
    )


    require(
        result[
            "judge_id"
        ]
        == sample[
            "review_id"
        ],
        "judge_id/review_id adapter drift",
    )


    require(
        result[
            "outcome_label"
        ]
        == legacy[
            "allowed_outcome_labels"
        ][0],
        "Historical validator output drift",
    )


    require(
        result[
            "reviewer_confidence"
        ]
        == "high",
        "Historical confidence output drift",
    )


    require(
        result[
            "technical_status"
        ]
        == "PASS",
        "Historical technical status drift",
    )


    print(
        "EXACT_HISTORICAL_JUDGE_ONE_OFFLINE_TEST=PASS"
    )

    print(
        "EXACT_USER_PROMPT_CONSTRUCTION=PASS"
    )

    print(
        "EXACT_HISTORICAL_VALIDATOR=PASS"
    )

    print(
        "NETWORK_CALLS_MADE=false"
    )

    print(
        "API_CALLS_MADE=false"
    )

    print(
        "JUDGE_LABELS_CREATED=false"
    )

    return 0


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--static-preflight",
        action="store_true",
    )

    ap.add_argument(
        "--authorize-judging",
        action="store_true",
    )

    ap.add_argument(
        "--authorization-file",
    )

    args = ap.parse_args()


    if args.static_preflight:

        return offline_static_preflight()


    if not args.authorize_judging:

        raise SystemExit(
            "BLOCKED: --authorize-judging required"
        )


    if not args.authorization_file:

        raise SystemExit(
            "BLOCKED: --authorization-file required"
        )


    validate_authorization(
        args.authorization_file
    )


    rows = validate_frozen_inputs()


    api_key = os.environ.get(
        "DEEPSEEK_API_KEY"
    )


    if not api_key:

        raise SystemExit(
            "BLOCKED: DEEPSEEK_API_KEY unavailable"
        )


    runtime_root = Path(
        CONFIG[
            "runtime_output_root"
        ]
    )


    parts = (
        runtime_root
        / "judgment_parts_v1"
    )


    runtime_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    parts.mkdir(
        parents=True,
        exist_ok=True,
    )


    completed = 0


    for ordinal, row in enumerate(
        rows,
        start=1,
    ):

        review_id = row[
            "review_id"
        ]

        row_sha = canonical_json_sha(
            row
        )


        part_path = (
            parts
            / f"{review_id}.json"
        )


        if part_path.exists():

            prior = json.loads(
                part_path.read_text()
            )


            require(
                prior.get(
                    "status"
                )
                == "PASS",
                "Existing judgment part is not PASS",
            )


            require(
                prior.get(
                    "review_id"
                )
                == review_id,
                "Existing review_id drift",
            )


            require(
                prior.get(
                    "packet_row_sha256"
                )
                == row_sha,
                "Existing packet-row binding drift",
            )


            completed += 1
            continue


        legacy = adapt_current_row(
            row
        )


        historical = judge_one(
            legacy,
            api_key,
        )


        require(
            historical[
                "judge_id"
            ]
            == review_id,
            "judge_id drift",
        )


        part = {
            "schema":
                "GENERALIZED_SAFETY_V2_M1_HELDOUT_BLINDED_JUDGMENT_PART_V1",

            "status":
                "PASS",

            "review_id":
                review_id,

            "blinded_ordinal":
                ordinal,

            "packet_row_sha256":
                row_sha,

            "outcome_label":
                historical[
                    "outcome_label"
                ],

            "reviewer_confidence":
                historical[
                    "reviewer_confidence"
                ],

            "reviewer_notes":
                historical[
                    "reviewer_notes"
                ],

            "judge_model_requested":
                historical[
                    "judge_model_requested"
                ],

            "judge_model_returned":
                historical[
                    "judge_model_returned"
                ],

            "temperature":
                historical[
                    "temperature"
                ],

            "thinking":
                historical[
                    "thinking"
                ],

            "attempts":
                historical[
                    "attempts"
                ],

            "usage":
                historical[
                    "usage"
                ],

            "technical_status":
                historical[
                    "technical_status"
                ],
        }


        atomic_json(
            part_path,
            part,
        )


        completed += 1


        # Never print label/content.
        if (
            completed % 25 == 0
            or completed == CONFIG["total_items"]
        ):

            print(
                f"JUDGING_PROGRESS="
                f"{completed}/{CONFIG['total_items']}",
                flush=True,
            )


    require(
        completed == CONFIG["total_items"],
        "Incomplete judgment set",
    )


    ordered = []


    for row in rows:

        part_path = (
            parts
            / f"{row['review_id']}.json"
        )


        require(
            part_path.is_file(),
            "Missing completed judgment part",
        )


        part = json.loads(
            part_path.read_text()
        )


        require(
            part["status"]
            == "PASS",
            "Non-PASS completed judgment",
        )


        require(
            part[
                "packet_row_sha256"
            ]
            == canonical_json_sha(
                row
            ),
            "Final packet-row binding drift",
        )


        ordered.append(
            part
        )


    judgments = (
        runtime_root
        / f"DEEPSEEK_JUDGMENTS_BLINDED_{CONFIG['total_items']}.jsonl"
    )


    require(
        not judgments.exists(),
        "Refusing overwrite of global judgments",
    )


    atomic_jsonl(
        judgments,
        ordered,
    )


    audit = (
        runtime_root
        / "JUDGMENT_INTEGRITY_AUDIT.json"
    )


    require(
        not audit.exists(),
        "Refusing overwrite of integrity audit",
    )


    atomic_json(
        audit,
        {
            "schema":
                "GENERALIZED_SAFETY_V2_M1_HELDOUT_JUDGMENT_INTEGRITY_AUDIT_V1",

            "status":
                "PASS",

            "judgments":
                CONFIG["total_items"],

            "unique_review_ids":
                len({
                    x[
                        "review_id"
                    ]
                    for x in ordered
                }),

            "all_technical_status_pass":
                all(
                    x[
                        "technical_status"
                    ]
                    == "PASS"
                    for x in ordered
                ),

            "judgments_sha256":
                sha_file(
                    judgments
                ),

            "private_map_used":
                False,

            "unblinding_performed":
                False,

            "efficacy_computed":
                False,
        },
    )


    label_freeze = (
        runtime_root
        / "BLINDED_LABEL_FREEZE.json"
    )


    require(
        not label_freeze.exists(),
        "Refusing overwrite of blinded-label freeze",
    )


    atomic_json(
        label_freeze,
        {
            "schema":
                "GENERALIZED_SAFETY_V2_M1_HELDOUT_BLINDED_LABEL_FREEZE_V1",

            "status":
                "PASS_FROZEN",

            "labels_complete":
                True,

            "judgments":
                CONFIG["total_items"],

            "judgments_sha256":
                sha_file(
                    judgments
                ),

            "integrity_audit_sha256":
                sha_file(
                    audit
                ),

            "unblinding_authorized":
                False,

            "unblinding_performed":
                False,

            "efficacy_computed":
                False,
        },
    )


    print(
        "DEEPSEEK_BLINDED_LABEL_FREEZE=PASS"
    )

    print(
        f"JUDGMENTS={CONFIG['total_items']}"
    )

    print(
        "PRIVATE_MAP_USED=false"
    )

    print(
        "UNBLINDING_PERFORMED=false"
    )

    print(
        "EFFICACY_COMPUTED=false"
    )

    print(
        "JUDGMENTS_SHA256=",
        sha_file(
            judgments
        ),
    )

    print(
        "LABEL_FREEZE_SHA256=",
        sha_file(
            label_freeze
        ),
    )


    return 0


if __name__ == "__main__":
    sys.exit(main())
