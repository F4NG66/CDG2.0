# RRAE v2: Injection-Signal Analysis and Steering for LLaDA

This directory contains the complete RRAE v2 research pipeline used to study
injection-related internal representations and steering behavior in
LLaDA-8B-Instruct.

RRAE stands for Rank Reduction Auto Encoder.

## Motivation

The original pilot dataset contained substantial label noise: approximately
78% of prompts initially treated as harmful were later found not to be
genuinely harmful. The v2 pipeline therefore rebuilds the dataset from clean,
balanced sources and reruns representation learning, vector extraction,
steering, and evaluation.

The central research question is:

> Can a low-rank representation model capture a transferable injection signal,
> and can that signal be used for selective safety steering without degrading
> benign utility?

## Dataset design

The v2 dataset contains four balanced buckets:

| Bucket | Safety label | Injection status |
|---|---|---|
| A | Harmful | Clean |
| B | Harmful | DIJA-injected |
| C | Benign | DIJA-injected |
| D | Benign | Clean |

Each bucket contains 500 examples:

- 400 health-related examples
- 100 non-health control examples

Total:

- 2,000 training examples
- 1,000 harmful examples
- 1,000 benign examples
- 1,000 injected examples
- 1,000 clean examples

This factorial design separates injection-related variation from topic and
harmfulness variation.

Held-out evaluation data is constructed from independent sources including
SALAD-BENCH, MedQuAD, XSTest, CARES, and EGIDA-style sources where applicable.

Raw datasets are not committed to this repository. The included scripts
document the construction, auditing, filtering, merging, and blocklisting
pipeline.

## Pipeline overview

The pipeline is organized into the following stages.

### 1. Data construction

Scripts under:

    scripts/data/

perform:

- harmful-clean dataset construction
- benign rewriting
- DIJA injection generation
- bucket finalization
- source auditing
- held-out dataset construction
- training-data blocklisting
- conversion into CDG-compatible prompt roots

### 2. Hidden-state extraction

The extraction stage records internal LLaDA hidden states from selected layers
and prompt regions.

Main script:

    scripts/extraction/extract_hidden_states_abcd_v2_input_region.py

The primary analyzed layers are:

- Layer 11
- Layer 16

### 3. RRAE training

Main script:

    scripts/training/train_rrae_v2.py

The training sweep includes ranks such as:

- 8
- 16
- 24
- 32
- 48
- 64

The RRAE learns a low-rank representation of selected hidden-state regions.

### 4. Steering-vector extraction

Main script:

    scripts/vectors/extract_rrae_v2_vectors.py

Compared vector sources include:

- raw hidden-state difference
- RRAE reconstruction difference
- RRAE residual difference

The principal contrast is based on injection-controlled differences such as:

    (B - A) + (C - D)

This contrast is designed to isolate injection-related variation while
controlling for harmfulness.

### 5. Steering

Steering scripts are located under:

    scripts/steering/

Two intervention regions are evaluated:

- R0: template-mask positions
- R2: template-mask plus output-mask positions

The pipeline supports:

- additive steering
- projection removal
- vector-source ablation
- layer and rank sweeps
- alpha and lambda sweeps

### 6. Judging

Judge scripts are located under:

    scripts/judging/

The evaluation framework includes:

- attack success
- refusal behavior
- harmful leakage
- collapse
- specificity
- task completion
- relevance
- benign utility

DeepSeek-based judging reads the API key from:

    DEEPSEEK_API_KEY

API keys must never be stored in source files.

### 7. Diagnostics and analysis

Diagnostic scripts are located under:

    scripts/diagnostics/

Analysis scripts are located under:

    scripts/analysis/

These scripts evaluate:

- train-to-held-out direction transfer
- signal preservation through the RRAE
- reconstruction quality
- raw-versus-reconstruction vector similarity
- residual-vector behavior
- projection-removal effects
- majority-vote judge outputs
- collapse and benign-utility degradation

## Main findings

The representation analysis found a strong transferable injection direction
between training and held-out data.

Observed train-to-held-out cosine similarity was approximately:

- 0.902 at Layer 11
- 0.902 at Layer 16

The injection signal was strongly preserved through the encoder and
reconstruction stages.

Raw and reconstruction steering vectors were nearly identical:

- Layer 11 cosine: approximately 0.999956
- Layer 16 cosine: approximately 0.999959

The residual component was substantially weaker and less reliably
reconstructed on held-out data.

Projection removal showed measurable causal leverage over model behavior.
However, stronger projection settings, especially lambda 2, caused severe
behavioral instability.

For the strongest apparent Layer-11 configuration:

- harmful-injected success decreased from approximately 84% to 55%
- benign-injected success changed from approximately 67% to 62%

More detailed P11 and benign-utility judging showed that much of the apparent
safety improvement was caused by output collapse and loss of specificity.

Observed degradation included:

- sharply increased collapse
- shorter outputs
- reduced task completion
- reduced relevance
- reduced benign helpfulness
- harmful leakage followed by late refusal in some cases

The current conclusion is:

> The injection direction is real, transferable, and behaviorally causal, but
> the tested interventions are not yet sufficiently selective or calibrated
> for reliable safety steering.

## Repository structure

    experiments/rrae_v2/
    ├── configs/
    ├── docs/
    ├── model_metadata/
    ├── results/
    │   └── summaries/
    ├── scripts/
    │   ├── analysis/
    │   ├── data/
    │   ├── diagnostics/
    │   ├── extraction/
    │   ├── job_generation/
    │   ├── judging/
    │   ├── steering/
    │   ├── training/
    │   └── vectors/
    ├── slurm/
    └── tests/

## Environment variables

The portable SLURM templates use the following variables:

    export RRAE_WORK_ROOT="/path/to/rrae_steering_work_v2"
    export RRAE_ENV_ROOT="/path/to/python-environment"
    export RRAE_DATA_ROOT="/path/to/rrae_data"
    export LLADA_MODEL_PATH="/path/to/LLaDA-8B-Instruct"
    export DIJA_ROOT="/path/to/DIJA"

For judge jobs:

    export DEEPSEEK_API_KEY="your-key"

## Model weights

The base LLaDA-8B-Instruct model is approximately 15 GB and is not stored in
this Git repository.

See:

    model_metadata/MODEL_MANIFEST.md

The repository includes only model metadata and the expected directory layout.

## Large experiment artifacts

The following artifacts are intentionally excluded from Git:

- LLaDA weight shards
- hidden-state tensors
- RRAE checkpoints
- steering vectors
- raw generations
- raw judge outputs
- scheduler logs
- restricted or externally licensed datasets

Small summary files are included under:

    results/summaries/

## SLURM usage

The original experiments were executed using SLURM on GPU compute nodes.

Portable job templates are provided under:

    slurm/

Before submission, define the required environment variables described in:

    slurm/README.md

## Reproducibility status

The repository contains:

- the full research code
- data-construction scripts
- extraction code
- training code
- steering code
- evaluation code
- analysis code
- SLURM templates
- experiment configurations
- compact result summaries

Large binary artifacts and third-party datasets must be downloaded or generated
separately.
