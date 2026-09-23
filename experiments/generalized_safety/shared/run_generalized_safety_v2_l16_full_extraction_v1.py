#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import os

import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__import__("os").environ["CDG_WORK_ROOT"])

MODEL = Path(__import__("os").environ["CDG_LLADA_MODEL"])

HIST = ROOT / (
    "scripts/official_rrae/"
    "run_canonical_train_layer_specific_"
    "standardized_response_activation_v1.py"
)

EXTRACTOR = ROOT / (
    "scripts/official_rrae/"
    "extract_iterative_hidden_canonical_dija_v1.py"
)

INPUT_ROOT = ROOT / (
    "analysis_output/rrae_development/"
    "generalized_safety_v2_l16_extraction_inputs_v1"
)

INPUTS = INPUT_ROOT / (
    "GENERALIZED_SAFETY_V2_L16_EXTRACTION_INPUTS.jsonl"
)

INPUT_FREEZE = INPUT_ROOT / "EXTRACTION_INPUT_FREEZE.json"

SMOKE_ROOT = ROOT / (
    "analysis_output/rrae_development/"
    "generalized_safety_v2_l16_extraction_smoke_v1"
)

SMOKE_SUMMARY = SMOKE_ROOT / "SMOKE_SUMMARY.json"
SMOKE_FREEZE = SMOKE_ROOT / "SMOKE_FREEZE.json"
SMOKE_ACTIVATIONS = SMOKE_ROOT / "SMOKE_ACTIVATIONS.pt"


EXPECTED_INPUTS_SHA = (
    "84453f2e9dbf35ffbcc3134c8eaae1ff"
    "d868e961bdb86ca7f06b34715ee64ac7"
)

EXPECTED_INPUT_FREEZE_SHA = (
    "25109b625c236d9112efbf187f992b00"
    "c23d16ad196ca6f4b0421022ab1084cf"
)

EXPECTED_SMOKE_SUMMARY_SHA = (
    "ae6a7abda6a9eab68ef31a7e3e7fd249"
    "8b4f8e5ad800d6a13b11dcfd0ab80e70"
)

EXPECTED_SMOKE_ACTIVATIONS_SHA = (
    "373b3e7eed01bf2a2edc4b5c0645e232"
    "42dddf1a309f72ff5247c69f28ac6e4b"
)

EXPECTED_HIST_SHA = (
    "da611076397139b13043f610f224f3bd"
    "c56b64df3a78beeba724f4479e885c9c"
)

EXPECTED_EXTRACTOR_SHA = (
    "67832c5cb193699b2fe83b545b6013b2"
    "90dc8cddbd8d35504c983bda1cb92680"
)

LAYER = 16
HIDDEN_DIM = 4096


def require(x, msg):
    if not x:
        raise RuntimeError(msg)


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    require(
        spec is not None and spec.loader is not None,
        f"Cannot import {path}",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def atomic_json(path, obj):
    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_jsonl(path, rows):
    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                ) + "\n"
            )
    os.replace(tmp, path)


def atomic_torch(path, obj):
    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()

    require(
        not out.exists() or not any(out.iterdir()),
        f"Output directory non-empty: {out}",
    )

    # ========================================================
    # Immutable authorization gates
    # ========================================================

    require(
        sha_file(INPUTS) == EXPECTED_INPUTS_SHA,
        "Extraction inputs SHA mismatch",
    )

    require(
        sha_file(INPUT_FREEZE)
        == EXPECTED_INPUT_FREEZE_SHA,
        "Extraction input freeze SHA mismatch",
    )

    require(
        sha_file(SMOKE_SUMMARY)
        == EXPECTED_SMOKE_SUMMARY_SHA,
        "Smoke summary SHA mismatch",
    )

    require(
        sha_file(SMOKE_ACTIVATIONS)
        == EXPECTED_SMOKE_ACTIVATIONS_SHA,
        "Smoke activations SHA mismatch",
    )

    require(
        sha_file(HIST) == EXPECTED_HIST_SHA,
        "Historical response runner changed",
    )

    require(
        sha_file(EXTRACTOR) == EXPECTED_EXTRACTOR_SHA,
        "Canonical extractor changed",
    )

    smoke = json.loads(
        SMOKE_FREEZE.read_text(encoding="utf-8")
    )

    require(smoke["status"] == "PASS", "Smoke not PASS")
    require(smoke["smoke_only"] is True, "Unexpected smoke mode")
    require(
        smoke["full_extraction_run"] is False,
        "Smoke reports full extraction",
    )
    require(
        smoke["model_forward_calls"] == 4,
        "Unexpected smoke forward count",
    )
    require(
        smoke["development_used"] is False,
        "Smoke used DEVELOPMENT",
    )
    require(
        smoke["direction_constructed"] is False,
        "Smoke constructed direction",
    )
    require(
        smoke["beta_calibrated"] is False,
        "Smoke calibrated beta",
    )

    rows = read_jsonl(INPUTS)

    require(len(rows) == 436, "Expected 436 rows")

    counts = Counter(x["family"] for x in rows)

    require(counts["DIJA"] == 237, "DIJA != 237")
    require(counts["RENELLM"] == 199, "RENELLM != 199")

    keys = {
        (x["family"], int(x["pair_id"]))
        for x in rows
    }

    require(
        len(keys) == 436,
        "Duplicate family/pair identity",
    )

    require(
        all(x["source_role"] == "CONSTRUCTION_BUILD" for x in rows),
        "Non-BUILD row found",
    )

    # ========================================================
    # Load model only after all static gates PASS
    # ========================================================

    out.mkdir(parents=True, exist_ok=True)

    hist = load_module(
        HIST,
        "historical_response_full_v2",
    )

    extractor = load_module(
        EXTRACTOR,
        "canonical_extractor_full_v2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL),
        trust_remote_code=True,
        local_files_only=True,
    )

    require(
        torch.cuda.is_available(),
        "CUDA required",
    )

    device = torch.device("cuda")

    model = AutoModel.from_pretrained(
        str(MODEL),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)

    blocks, resolution = hist.resolve_transformer_blocks(
        model,
        LAYER,
    )

    require(
        resolution["selected_module_path"]
        == "model.transformer.blocks",
        "Unexpected transformer module path",
    )

    require(
        int(resolution["selected_block_count"]) == 32,
        "Expected 32 transformer blocks",
    )

    capture = hist.HiddenCapture(
        blocks[LAYER],
        extractor,
    )

    family_data = {
        "DIJA": {
            "pair_ids": [],
            "domains": [],
            "matched_prefix_k": [],
            "safe": [],
            "harm": [],
            "safe_all": [],
            "harm_all": [],
            "records": [],
        },
        "RENELLM": {
            "pair_ids": [],
            "domains": [],
            "matched_prefix_k": [],
            "safe": [],
            "harm": [],
            "safe_all": [],
            "harm_all": [],
            "records": [],
        },
    }

    completed = 0

    try:
        for row in rows:
            fam = row["family"]
            pid = int(row["pair_id"])

            require(
                fam in family_data,
                f"Unknown family {fam}",
            )

            safe_ids, safe_span, safe_meta = (
                hist.build_standardized_response_input(
                    tokenizer=tokenizer,
                    extractor=extractor,
                    base_request=row["base_request"],
                    response=row["safe_response"],
                )
            )

            harm_ids, harm_span, harm_meta = (
                hist.build_standardized_response_input(
                    tokenizer=tokenizer,
                    extractor=extractor,
                    base_request=row["base_request"],
                    response=row["harmful_response"],
                )
            )

            require(
                hist.tensor_sha256(safe_ids)
                == row["safe_input_ids_sha256"],
                f"Safe tokenization changed {fam}:{pid}",
            )

            require(
                hist.tensor_sha256(harm_ids)
                == row["harmful_input_ids_sha256"],
                f"Harm tokenization changed {fam}:{pid}",
            )

            safe_prefix = safe_ids[
                0, :safe_span[0]
            ].tolist()

            harm_prefix = harm_ids[
                0, :harm_span[0]
            ].tolist()

            require(
                safe_prefix == harm_prefix,
                f"Chat prefix mismatch {fam}:{pid}",
            )

            require(
                safe_meta["fully_visible"] is True
                and harm_meta["fully_visible"] is True,
                f"Response visibility changed {fam}:{pid}",
            )

            require(
                int(safe_meta["mask_count"]) == 0
                and int(harm_meta["mask_count"]) == 0,
                f"Mask unexpectedly present {fam}:{pid}",
            )

            require(
                safe_meta["generation_used"] is False
                and harm_meta["generation_used"] is False,
                f"Generation unexpectedly used {fam}:{pid}",
            )

            k = min(
                32,
                int(safe_meta["response_token_count"]),
                int(harm_meta["response_token_count"]),
            )

            require(
                k == int(row["matched_prefix_k"]),
                f"k changed {fam}:{pid}",
            )

            require(
                7 <= k <= 32,
                f"Invalid k={k} {fam}:{pid}",
            )

            s = hist.forward_and_pool(
                model=model,
                capture=capture,
                input_ids_cpu=safe_ids,
                response_span=safe_span,
                prefix_k=k,
                device=device,
            )

            h = hist.forward_and_pool(
                model=model,
                capture=capture,
                input_ids_cpu=harm_ids,
                response_span=harm_span,
                prefix_k=k,
                device=device,
            )

            vectors = (
                s["prefix_vector"],
                h["prefix_vector"],
                s["all_vector"],
                h["all_vector"],
            )

            for vec in vectors:
                require(
                    tuple(vec.shape) == (HIDDEN_DIM,),
                    f"Bad hidden shape {fam}:{pid}",
                )
                require(
                    torch.isfinite(vec).all(),
                    f"Non-finite hidden vector {fam}:{pid}",
                )

            d = family_data[fam]

            d["pair_ids"].append(pid)
            d["domains"].append(row["domain"])
            d["matched_prefix_k"].append(k)

            d["safe"].append(
                s["prefix_vector"].float().cpu()
            )
            d["harm"].append(
                h["prefix_vector"].float().cpu()
            )
            d["safe_all"].append(
                s["all_vector"].float().cpu()
            )
            d["harm_all"].append(
                h["all_vector"].float().cpu()
            )

            d["records"].append({
                "family":
                    fam,

                "pair_id":
                    pid,

                "domain":
                    row["domain"],

                "matched_prefix_k":
                    k,

                "safe_response_token_count":
                    int(safe_meta["response_token_count"]),

                "harmful_response_token_count":
                    int(harm_meta["response_token_count"]),

                "safe_input_ids_sha256":
                    s["input_ids_sha256"],

                "harmful_input_ids_sha256":
                    h["input_ids_sha256"],

                "safe_hidden_all_sha256":
                    s["hidden_all_sha256"],

                "harmful_hidden_all_sha256":
                    h["hidden_all_sha256"],

                "safe_prefix_vector_sha256":
                    hist.tensor_sha256(
                        s["prefix_vector"]
                    ),

                "harmful_prefix_vector_sha256":
                    hist.tensor_sha256(
                        h["prefix_vector"]
                    ),

                "safe_all_vector_sha256":
                    hist.tensor_sha256(
                        s["all_vector"]
                    ),

                "harmful_all_vector_sha256":
                    hist.tensor_sha256(
                        h["all_vector"]
                    ),

                "fully_visible_response":
                    True,

                "mask_count":
                    0,

                "generation_used":
                    False,
            })

            completed += 1

            print(
                f"completed={completed}/436 "
                f"family={fam} pair={pid} k={k}",
                flush=True,
            )

    finally:
        capture.close()

    require(completed == 436, "Did not complete all rows")

    # ========================================================
    # Family artifacts
    # ========================================================

    top_summary = {}

    for fam, expected_n in (
        ("DIJA", 237),
        ("RENELLM", 199),
    ):
        d = family_data[fam]

        require(
            len(d["pair_ids"]) == expected_n,
            f"{fam} output count mismatch",
        )

        safe = torch.stack(d["safe"]).float()
        harm = torch.stack(d["harm"]).float()
        safe_all = torch.stack(d["safe_all"]).float()
        harm_all = torch.stack(d["harm_all"]).float()

        require(
            tuple(safe.shape)
            == (expected_n, HIDDEN_DIM),
            f"{fam} safe shape mismatch",
        )

        require(
            tuple(harm.shape)
            == (expected_n, HIDDEN_DIM),
            f"{fam} harm shape mismatch",
        )

        require(torch.isfinite(safe).all(), f"{fam} safe nonfinite")
        require(torch.isfinite(harm).all(), f"{fam} harm nonfinite")
        require(torch.isfinite(safe_all).all(), f"{fam} safe_all nonfinite")
        require(torch.isfinite(harm_all).all(), f"{fam} harm_all nonfinite")

        famdir = out / fam
        famdir.mkdir()

        activation_path = (
            famdir / "STANDARDIZED_RESPONSE_ACTIVATIONS.pt"
        )

        atomic_torch(
            activation_path,
            {
                "schema":
                    "GENERALIZED_SAFETY_V2_L16_"
                    "STANDARDIZED_RESPONSE_ACTIVATIONS_V1",

                "family":
                    fam,

                "layer_zero_based":
                    LAYER,

                "hidden_dim":
                    HIDDEN_DIM,

                "pair_ids":
                    d["pair_ids"],

                "domains":
                    d["domains"],

                "matched_prefix_k":
                    d["matched_prefix_k"],

                "safe_prefix":
                    safe,

                "harmful_prefix":
                    harm,

                "safe_all":
                    safe_all,

                "harmful_all":
                    harm_all,

                "orientation":
                    "safe_minus_harmful",

                "fully_visible_response":
                    True,

                "masking_used":
                    False,

                "generation_used":
                    False,
            },
        )

        records_path = famdir / "EXTRACTION_RECORDS.jsonl"
        atomic_jsonl(records_path, d["records"])

        family_summary = {
            "protocol":
                "GENERALIZED_SAFETY_V2_L16_"
                f"{fam}_EXTRACTION_SUMMARY_V1",

            "status":
                "PASS",

            "family":
                fam,

            "pair_count":
                expected_n,

            "layer_zero_based":
                LAYER,

            "hidden_dim":
                HIDDEN_DIM,

            "tensor_shape":
                [expected_n, HIDDEN_DIM],

            "k_min":
                min(d["matched_prefix_k"]),

            "k_max":
                max(d["matched_prefix_k"]),

            "fully_visible_response":
                True,

            "masking_used":
                False,

            "generation_used":
                False,

            "development_used":
                False,

            "direction_constructed":
                False,

            "beta_calibrated":
                False,

            "activations_sha256":
                sha_file(activation_path),

            "records_sha256":
                sha_file(records_path),
        }

        family_summary_path = famdir / "EXTRACTION_SUMMARY.json"
        atomic_json(
            family_summary_path,
            family_summary,
        )

        top_summary[fam] = {
            "pair_count":
                expected_n,

            "activations_path":
                str(activation_path),

            "activations_sha256":
                sha_file(activation_path),

            "records_path":
                str(records_path),

            "records_sha256":
                sha_file(records_path),

            "summary_path":
                str(family_summary_path),

            "summary_sha256":
                sha_file(family_summary_path),
        }

    # ========================================================
    # Top-level audit/freeze
    # ========================================================

    summary = {
        "protocol":
            "GENERALIZED_SAFETY_V2_L16_"
            "FULL_EXTRACTION_V1",

        "status":
            "PASS",

        "full_extraction_run":
            True,

        "extraction_row_count":
            436,

        "model_forward_calls":
            872,

        "DIJA_pair_count":
            237,

        "RENELLM_pair_count":
            199,

        "layer_zero_based":
            LAYER,

        "hidden_dim":
            HIDDEN_DIM,

        "transformer_module_path":
            resolution["selected_module_path"],

        "transformer_block_count":
            int(resolution["selected_block_count"]),

        "fully_visible_response":
            True,

        "masking_used":
            False,

        "generation_used":
            False,

        "development_used":
            False,

        "direction_constructed":
            False,

        "beta_calibrated":
            False,

        "orientation":
            "safe_minus_harmful",

        "source_inputs_sha256":
            sha_file(INPUTS),

        "source_input_freeze_sha256":
            sha_file(INPUT_FREEZE),

        "source_smoke_summary_sha256":
            sha_file(SMOKE_SUMMARY),

        "source_smoke_activations_sha256":
            sha_file(SMOKE_ACTIVATIONS),

        "source_smoke_freeze_sha256":
            sha_file(SMOKE_FREEZE),

        "families":
            top_summary,
    }

    summary_path = out / "FULL_EXTRACTION_SUMMARY.json"
    atomic_json(summary_path, summary)

    freeze = {
        "protocol":
            "GENERALIZED_SAFETY_V2_L16_"
            "FULL_EXTRACTION_FREEZE_V1",

        "status":
            "PASS_FROZEN",

        "frozen":
            True,

        "gpu_used":
            True,

        "full_extraction_completed":
            True,

        "extraction_row_count":
            436,

        "model_forward_calls":
            872,

        "DIJA_pair_count":
            237,

        "RENELLM_pair_count":
            199,

        "layer_zero_based":
            16,

        "hidden_dim":
            4096,

        "development_used":
            False,

        "direction_constructed":
            False,

        "beta_calibrated":
            False,

        "source_input_freeze_sha256":
            sha_file(INPUT_FREEZE),

        "source_smoke_freeze_sha256":
            sha_file(SMOKE_FREEZE),

        "full_extraction_summary_sha256":
            sha_file(summary_path),

        "DIJA_activations_sha256":
            top_summary["DIJA"]["activations_sha256"],

        "RENELLM_activations_sha256":
            top_summary["RENELLM"]["activations_sha256"],
    }

    freeze_path = out / "FULL_EXTRACTION_FREEZE.json"
    atomic_json(freeze_path, freeze)

    artifacts = [
        out / "DIJA" / "STANDARDIZED_RESPONSE_ACTIVATIONS.pt",
        out / "DIJA" / "EXTRACTION_RECORDS.jsonl",
        out / "DIJA" / "EXTRACTION_SUMMARY.json",
        out / "RENELLM" / "STANDARDIZED_RESPONSE_ACTIVATIONS.pt",
        out / "RENELLM" / "EXTRACTION_RECORDS.jsonl",
        out / "RENELLM" / "EXTRACTION_SUMMARY.json",
        summary_path,
        freeze_path,
    ]

    with (out / "SHA256SUMS.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        for p in artifacts:
            f.write(
                f"{sha_file(p)}  {p.relative_to(out)}\n"
            )

    print(
        "GENERALIZED_SAFETY_V2_L16_FULL_EXTRACTION=PASS"
    )
    print("STATUS=PASS_FROZEN")
    print("EXTRACTION_ROWS=436")
    print("MODEL_FORWARD_CALLS=872")
    print("DIJA_PAIRS=237")
    print("RENELLM_PAIRS=199")
    print("LAYER_ZERO_BASED=16")
    print("HIDDEN_DIM=4096")
    print(
        "TRANSFORMER_MODULE="
        + resolution["selected_module_path"]
    )
    print("TRANSFORMER_BLOCK_COUNT=32")
    print("DEVELOPMENT_USED=false")
    print("DIRECTION_CONSTRUCTED=false")
    print("BETA_CALIBRATED=false")
    print(
        "DIJA_ACTIVATIONS_SHA256="
        + top_summary["DIJA"]["activations_sha256"]
    )
    print(
        "RENELLM_ACTIVATIONS_SHA256="
        + top_summary["RENELLM"]["activations_sha256"]
    )
    print(
        "FULL_EXTRACTION_FREEZE_SHA256="
        + sha_file(freeze_path)
    )


if __name__ == "__main__":
    main()
