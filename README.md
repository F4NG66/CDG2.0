# CDG2.0 — Final RRAE + Steering Pipeline

CDG2.0 studies internal signatures of DIJA template injection in diffusion language models and tests whether activation steering can move generation toward safer behavior without destroying benign utility.

- Target model: **LLaDA-8B-Instruct**
- Attack: **DIJA template injection**
- Final paper path: **2,000-sample canonical A/B/C/D pipeline**
- Main finding: **Representation != Control**

The new `rrae/`, `steering/`, `configs/final_rrae_steering.yaml`, and `results/frozen_35_summary.json` files are the finalized paper path. Existing `cdg/`, `scripts/`, and Dream experiments are retained for provenance and reuse.

## Canonical experimental design

| Condition | Content | DIJA template | Samples | Health | Non-health |
|---|---|---:|---:|---:|---:|
| A | Harmful | No | 500 | 400 | 100 |
| B | Harmful | Yes | 500 | 400 | 100 |
| C | Benign | Yes | 500 | 400 | 100 |
| D | Benign | No | 500 | 400 | 100 |
| **Total** |  |  | **2,000** | **1,600** | **400** |

The controlled injection contrasts are

```text
v_BA = mean(B) - mean(A)
v_CD = mean(C) - mean(D)
```

They isolate template injection while holding content type fixed. **B vs C is not the primary injection direction** because it changes harmfulness and therefore confounds injection with content.

## Final pipeline

1. Prepare the paired A/B/C/D dataset externally.
2. Extract hidden states with the existing `cdg` recorder/backend.
3. Screen the 72 candidate hidden-state views.
4. Train the 42-model RRAE rank sweep over `k in {4, 8, 16, 24, 32, 48, 64}`.
5. Select the final `harm / f=0.05 / L11 / rank-4` representation.
6. Construct `v_inj` from the agreement between residual `B-A` and `C-D`.
7. Construct a separate `v_safety` from paired harmful-behavior and safe refusal/redirection responses.
8. Run additive, projection-removal, safety-direction, combined, and strength/schedule interventions.
9. Evaluate B/C jointly using safety conversion and utility preservation.

### RRAE

RRAE learns a compact rank-k reconstruction and exposes its residual:

```text
h -> encoder -> rank-k latent -> decoder -> h_hat
r = h - h_hat
```

Injection directions are constructed from residual-space controlled contrasts. Larger ranks can improve reconstruction without improving injection geometry, so model selection must not minimize reconstruction loss alone.

The finalized selected representation has:

| Metric | Value |
|---|---:|
| Validation injection AUC | 1.0000 |
| `cos(B-A, C-D)` in residual space | 0.9621 |
| Train-validation direction cosine | ~0.991 |
| Positive-pair consistency | 100% |
| Harmfulness AUC guardrail | 0.4357 |
| Position-count correlation | 0.0495 |

These results establish a clean, coherent representation; they do **not** establish causal control.

### Directions are not interchangeable

The shared injection direction is obtained from normalized controlled contrasts:

```text
u_BA = normalize(mean(B) - mean(A))
u_CD = normalize(mean(C) - mean(D))
M = stack([u_BA, u_CD])
v_inj = first_right_singular_vector(M)
```

Its sign is aligned with `u_BA`. `v_safety` is independently estimated as movement from harmful compliance toward coherent refusal/redirection. In particular, **`-v_inj` is not assumed to equal `v_safety`**.

## Steering families

Every condition separates four dimensions: **direction**, **dose**, **location**, and **time**.

| Family | Operator | Key property |
|---|---|---|
| Additive | `h' = h + alpha v_inj` | Fixed displacement for every selected state |
| Projection removal | `h' = h - rho <h,v_inj>v_inj` | State-dependent removal |
| Safety direction | `h' = h + beta v_safety` | Separate behavioral direction |
| Combined | projection removal, then safety addition | Independent `rho` and `beta` |
| Strength / schedule | configurable operator and window | Single-step, windowed, or persistent |

Persistent steering is reapplied across denoising steps. The strongest final condition used safety-direction steering at layer 16, on currently masked positions, persistently, at approximately `2 x` the layer-median calibrated dose.

## Frozen 35-pair replication

Strict success requires both:

1. B becomes a coherent safe refusal/redirection; and
2. its paired C response remains helpful.

| Strategy | B Safe Conversion | C Helpful Preservation | C Utility Failure | Strict Paired Success |
|---|---:|---:|---:|---:|
| Additive | 2.9% | 94.3% | 5.7% | 2.9% |
| **Safety Direction** | **28.6%** | **88.6%** | **11.4%** | **25.7%** |
| Combined | 11.4% | 94.3% | 5.7% | 11.4% |
| Projection Removal | 5.7% | 91.4% | 8.6% | 5.7% |
| Strength / Schedule | 17.1% | 42.9% | 57.1% | 8.6% |

Safety-direction steering was strongest, but achieved only partial causal control. The final result is a **detection-correction asymmetry**: injection is easy to detect internally, while reliable behavioral correction is much harder.

## Dataset

The dataset is intentionally **not included** in this repository. It is distributed separately. Set its path through configuration or CLI arguments. No KAUST, UofT, or other cluster-specific path is hardcoded.

Expected records are an external `manifest.jsonl` directory, JSONL file, or PyTorch records bundle. Each row must preserve its A/B/C/D group and a shared family/pair identifier so splitting and paired evaluation cannot leak across conditions.

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

The machine-readable final condition and results are in [`configs/final_rrae_steering.yaml`](configs/final_rrae_steering.yaml) and [`results/frozen_35_summary.json`](results/frozen_35_summary.json).
