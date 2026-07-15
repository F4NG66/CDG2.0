# Artifact Manifest

This document distinguishes source-controlled research materials from large,
generated, licensed, or sensitive artifacts that are intentionally excluded
from Git.

## Included in this repository

The repository includes:

- experiment source code
- data-construction and auditing scripts
- hidden-state extraction code
- RRAE training code
- vector-extraction code
- steering and projection-removal code
- judging and analysis code
- portable SLURM job templates
- experiment configurations
- model configuration metadata
- compact result summaries
- documentation of the experiment design and findings

## Base model

The experiments use:

    LLaDA-8B-Instruct

The local checkpoint used during the experiments contains six safetensors
shards with a combined size of approximately 15 GB.

The base-model weight shards are not included in this Git repository.

Expected files include:

    model-00001-of-00006.safetensors
    model-00002-of-00006.safetensors
    model-00003-of-00006.safetensors
    model-00004-of-00006.safetensors
    model-00005-of-00006.safetensors
    model-00006-of-00006.safetensors
    model.safetensors.index.json

See:

    model_metadata/MODEL_MANIFEST.md

## Hidden-state tensors

Generated hidden-state tensors are excluded, including artifacts corresponding
to locations such as:

    hidden_states/abcd_v2/
    hidden_states/heldout_v1/

These tensors can be regenerated using the scripts under:

    scripts/extraction/

## RRAE checkpoints

Trained RRAE checkpoints are excluded from ordinary Git.

The original experiment layout used directories such as:

    rrae_runs_v2/input_region_L11_f0p05_rank<RANK>/
    rrae_runs_v2/input_region_L16_f0p05_rank<RANK>/

Typical generated files include:

    best_model.pt

These checkpoints can be regenerated using the training scripts and SLURM
templates under:

    scripts/training/
    slurm/rrae_v2/

## Steering vectors

Generated steering vectors are excluded from ordinary Git.

The experiment used vectors including:

- raw hidden-state contrast vectors
- RRAE reconstruction vectors
- RRAE residual vectors
- layer-specific and rank-specific vectors
- projection-removal ablation vectors

The expected output location is:

    steering_vectors_v2/

Vector filenames encode properties such as:

- layer
- rank
- source representation
- contrast definition
- training dataset

The vectors can be regenerated using:

    scripts/vectors/

## Generated model outputs

Raw generation files are excluded, including outputs under paths such as:

    steering_runs/
    generated_outputs/

These files may contain large numbers of model generations and potentially
sensitive harmful-content evaluation examples.

## Judge outputs

Raw judge outputs are excluded, including:

    judge_outputs/
    steering_runs/**/judges/

Compact aggregate summaries required to interpret the experiments are included
under:

    results/summaries/

## Datasets

Raw, third-party, or restricted datasets are not redistributed in this
repository.

The dataset-construction scripts reference sources used during the project,
including, where applicable:

- HarmBench
- AdvBench
- MedSafetyBench
- SALAD-Bench
- MedQuAD
- XSTest
- CARES-derived evaluation material
- EGIDA-derived evaluation material

Users are responsible for obtaining each dataset from its official source and
complying with its license and usage terms.

The repository includes scripts documenting:

- source inspection
- sampling
- filtering
- rewriting
- DIJA injection
- auditing
- merging
- held-out construction
- training-set blocklisting

## External repositories and models

The following external dependencies are not vendored:

- LLaDA
- DIJA
- Qwen2.5-7B-Instruct
- third-party benchmark repositories
- third-party model checkpoints

Their local paths are provided through command-line arguments or environment
variables.

## Credentials

No API keys, private tokens, passwords, or local credential files are included.

Judge jobs expect credentials through environment variables such as:

    DEEPSEEK_API_KEY

Credentials must never be committed to Git.

## Recommended artifact distribution

Large experiment-specific checkpoints and vectors should be distributed using
an approved artifact-storage system rather than ordinary Git.

Before publishing binary artifacts, verify:

- repository-owner approval
- model and dataset license compatibility
- institutional data-sharing requirements
- absence of credentials and private information
- checksums and provenance metadata
