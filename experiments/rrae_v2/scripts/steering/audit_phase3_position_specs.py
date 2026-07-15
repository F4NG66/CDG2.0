#!/usr/bin/env python

import sys
from collections import defaultdict

import torch


RRAE_DATA = "/path/to/rrae_data"
if RRAE_DATA not in sys.path:
    sys.path.insert(0, RRAE_DATA)

from cdg.config import get_backend_config
from cdg.backends import build_runner
from cdg.data import load_cdg_root

from run_rrae_steering_generation_matrix_v2 import (
    parse_position_spec,
    build_position_mask,
    pad_posmask_for_generation,
)


PROMPT_ROOT = (
    "/path/to/rrae_steering_work_v2/"
    "data/phase3_prompt_roots/abcd_v2_in_sample_full"
)

MODEL_PATH = "/path/to/LLaDA-8B-Instruct"


def main():
    cfg = get_backend_config("llada_attack")
    cfg.model_id = MODEL_PATH

    runner = build_runner(
        cfg,
        sae_root="/path/to/rrae_data/saes",
        device="cpu",
        dummy=True,
    )

    cases = load_cdg_root(PROMPT_ROOT)

    selected = {}
    for case in cases:
        g = case.group_letter
        if g not in selected:
            selected[g] = case

    specs = {
        "R0": "template:mask",
        "R2": "template:mask+output:mask",
    }

    for name, spec in specs.items():
        definition = parse_position_spec(spec)

        print("=" * 100)
        print(name, spec)

        for group in ["A", "B", "C", "D"]:
            case = selected[group]

            posmask = build_position_mask(
                runner,
                case,
                definition,
            )

            padded = pad_posmask_for_generation(
                posmask,
                definition,
                cfg.decode.gen_length,
            )

            prompt_selected = int(posmask.sum().item())
            total_selected = int(padded.sum().item())
            output_selected = total_selected - prompt_selected

            print(
                group,
                "prompt_selected=",
                prompt_selected,
                "output_selected=",
                output_selected,
                "total_selected=",
                total_selected,
            )


if __name__ == "__main__":
    main()
