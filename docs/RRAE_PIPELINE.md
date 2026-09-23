# RRAE pipeline and historical frozen 35-pair replication: how to run

> **Historical path.** This page preserves the run instructions for the RRAE representation-learning pipeline and the earlier frozen 35-pair steering replication. That replication is a historical result, **not** the final project conclusion. For the final generalized safety-direction program (M1 → M3′), see [`EXPERIMENTAL_HISTORY.md`](EXPERIMENTAL_HISTORY.md). The instructions below are moved verbatim from the previous root README; only the closing sentence was relabeled as historical and its links re-pointed.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights and SAE checkpoints are downloaded or placed separately. The existing backend uses `transformers`, LLaDA remote model code, and the configured mask/unmask SAE bundles.

Expected external SAE layout:

```text
/path/to/saes/
├── llada_mask/
└── llada_unmask/
```

The model and SAE identifiers are defined in `cdg/config.py`; override local cache locations through normal Hugging Face environment variables rather than editing source paths.

## Reproduce the experiment end to end

The archive contains code, configuration, and the published aggregate results. It does **not** contain the canonical dataset, generated activations, model weights, SAE checkpoints, or private judge credentials.

### 1. Prepare the external canonical dataset

Create four directories with 500 matched cases each:

```text
/path/to/canonical_abcd/
├── A_harmful_clean/
├── B_harmful_injected/
├── C_benign_injected/
└── D_benign_clean/
```

Files may be JSON or JSONL. A minimal row is:

```json
{
  "id": "shared_pair_identifier",
  "behavior": "prompt text",
  "user_content": "{behavior}",
  "content_type": "harmful",
  "has_template": false,
  "attack_method": "none"
}
```

Injected B/C rows set `has_template` to `true`, `attack_method` to `DIJA`, and supply the externally generated DIJA scaffold in `user_content`. Preserve the same family or pair identifier across matched conditions. The finalized counts and health/non-health composition are specified in `configs/final_rrae_steering.yaml`.

### 2. Run a CPU smoke test

This checks data loading, recording, and output layout without downloading LLaDA or SAEs:

```bash
python run_record.py \
  --prompt-root /path/to/canonical_abcd \
  --dummy \
  --limit 1 \
  --out outputs/smoke
```

### 3. Extract final LLaDA hidden-state records

```bash
python run_record.py \
  --backend llada_attack \
  --prompt-root /path/to/canonical_abcd \
  --sae-root /path/to/saes \
  --out /path/to/final_records \
  --seeds 17
```

This records layers 4, 11, 16, and 26 at the configured denoising fractions and token scopes. Keep the resulting `.pt` files and `manifest.jsonl` outside Git.

### 4. Screen representations and train RRAE

Run the representation-screening and rank-sweep commands below. For the frozen paper result, the selected view is `harm / f=0.05 / L11`, and the selected checkpoint is rank 4. `selected_model.json` records metric-based selection; reconstruction loss is not the sole selection criterion.

### 5. Construct directions

Build `v_inj` from the RRAE residuals. To also build `v_safety`, provide a separate externally prepared record set containing matched harmful-compliance and safe-refusal/redirection states. By default, the safety view is layer 16, `out_mask`, fraction 0.05.

### 6. Run steering and paired evaluation

Use `configs/final_rrae_steering.yaml` as the frozen intervention specification. The existing `DLMRunner` accepts `ScheduledHookController`, so interventions are reapplied dynamically at the configured layer, token scope, dose, and denoising window. After judging generations, provide one JSONL row per family and B/C pair:

```json
{"family":"safety_direction","pair_id":"pair_001","b_safe":true,"c_helpful":true}
```

Then run the frozen-replication scoring command below. It requires exactly 35 unique pairs per evaluated family and computes strict success only when both B and C satisfy their respective criteria.

## Example commands

Representation screening:

```bash
python -m rrae.screen_representations \
  --records /path/to/records
```

RRAE rank sweep:

```bash
python -m rrae.train_rrae \
  --records /path/to/records \
  --layer 11 \
  --scope harm \
  --frac 0.05 \
  --ranks 4 8 16 24 32 48 64
```

Controlled direction construction:

```bash
python -m rrae.build_residual_directions \
  --checkpoint /path/to/rrae_rank4.pt \
  --records /path/to/records
```

To include `v_safety`, add `--safety-records /path/to/paired_behavior_records`; those records must contain both safe/refusal and harmful/compliance labels in `behavior_label` (configurable with `--safety-label-field`).

Frozen replication configuration check:

```bash
python -m steering.run_frozen_replication \
  --config configs/final_rrae_steering.yaml
```

Score externally judged paired generations:

```bash
python -m steering.run_frozen_replication \
  --config configs/final_rrae_steering.yaml \
  --predictions /path/to/frozen_35_judgments.jsonl
```

Each judgment row contains `family`, `pair_id`, `b_safe`, and `c_helpful`. The evaluator rejects duplicates and incomplete 35-pair family cohorts.

## Reproducibility and legacy code

- Seeds are deterministic and train/validation splitting occurs at the family level.
- Normalization statistics are learned from training data only.
- The final config contains no credentials or machine-specific paths.
- Legacy B-vs-C probing/steering scripts remain for historical reproduction only; they are not the final injection-mechanism path.
- Generated activations, checkpoints, datasets, credentials, and large outputs are ignored by Git.

The machine-readable condition and results of this historical frozen 35-pair replication are in [`configs/final_rrae_steering.yaml`](../configs/final_rrae_steering.yaml) and [`results/frozen_35_summary.json`](../results/frozen_35_summary.json).
