# Reproducibility

## What this repository contains

- **Frozen aggregate results.** [`results/generalized_safety/`](../results/generalized_safety/) holds one JSON per stage plus `lineage_status.json` and `INDEX.json`. Each file carries source-artifact SHA-256s, exact numerators and denominators, and status flags.
- **Frozen settings index.** [`configs/generalized_safety/final_program.yaml`](../configs/generalized_safety/final_program.yaml).
- **Pipeline source.** [`experiments/generalized_safety/`](../experiments/generalized_safety/) holds 44 scripts, copies of the frozen originals with paths sanitized. Every file maps to its original SHA in [`SCRIPT_PROVENANCE.md`](SCRIPT_PROVENANCE.md).
- **Dataset handoff.** [`data/canonical_abcd_v2_2/`](../data/canonical_abcd_v2_2/) holds the README, provenance and a read-only verifier for the private canonical file.

## What it does not contain

The following are deliberately **not published**:
- datasets and population manifests;
- the 576-group final pool;
- private blinding maps;
- raw or blinded judgments, and unblinded joins;
- raw generations;
- API call journals;
- hidden states and activations, and the directions tensor;
- model weights;
- cluster logs;
- credentials.

As a result, no stage can be re-executed from this repository alone. The published results can be checked against the SHA-256s recorded in each summary by anyone who holds the private frozen artifacts.

## Private dataset policy

- The canonical datasets are **not redistributed**. Authorized collaborators receive them separately.
- Place the v2.2 file at `data/canonical_abcd_v2_2/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl`. The path is git-ignored.
- Verify it:

  ```bash
  python data/canonical_abcd_v2_2/verify_dataset.py            # default path
  python data/canonical_abcd_v2_2/verify_dataset.py /path/to/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl
  ```

  Expected output: `DATASET_VERIFICATION=PASS` (SHA-256 `4392afca…8eaaf`, A/B/C/D = 500 each).
- The older Qwen-lineage file used for V2 BUILD/DEV has no public verifier yet. Its hash and role are in [`DATA_LINEAGE.md`](DATA_LINEAGE.md).

## Environment

- Python dependencies: `requirements.txt`. The generalized-safety scripts need `torch`, `transformers`, `numpy`, `scipy` and `scikit-learn`.
- The scripts read these environment variables:

| Variable | Meaning |
|---|---|
| `CDG_WORK_ROOT` | root of the private frozen artifact tree (`analysis_output/rrae_development/...`) |
| `CDG_LLADA_MODEL` | local LLaDA-8B-Instruct directory |
| `CDG_DIJA_ROOT` | local DIJA checkout (DIJA attack construction) |
| `DEEPSEEK_API_KEY` | judge credential (judging scripts); never stored in the repository |
| `OPENAI_API_KEY` | credential read by the ReNeLLM construction executor; never stored in the repository |

## Pipeline map

| Stage | Scripts (`experiments/generalized_safety/…`) |
|---|---|
| Shared | `shared/run_generalized_safety_v2_l16_full_extraction_v1.py` (L16 fully-visible extraction); `shared/run_generalized_safety_m1_eval_v1.py` (steered-generation runner) and its imported helpers; DIJA and ReNeLLM construction executors |
| M1 generation | `m1/run_m1_heldout_shard_v1.py` |
| M1 judging / metrics | `m1/run_heldout_m1_deepseek_judging_final.py`, `m1/run_heldout_m1_unblinding_metrics_v1.py` |
| M2 calibration | `m2/preflight_m2_measurement_v1.py`, `m2/run_m2_measurement_shard_v1.py`, `m2/solve_m2_betas_v1.py` |
| M2 generation | `m2/m2_norm_preserving_hook_v1.py`, `m2/m2_llada_runtime_v1.py`, `m2/generate_one_m2dmnp_source.py`, `m2/build_manifests_v1.py`, `m2/run_m2_llada_confirmatory_shard_v1.py` |
| M2 judging / metrics | `m2/02_build_packet_and_private_map.py`, `m2/run_m2_fresh_deepseek_judging.py`, `m2/07_post_judging_audit_and_freeze.py`, `m2/run_m2_unblinding_metrics_v1.py` |
| M3 feasibility / power / support | `m3/tgas_feasibility_spike_v1.py`, `m3/m3_power_simulation_v1.py`, `m3/01…04_*_v2.py` (judging v2 → DEV support check) |
| Scenario-match audit | `m3/scenario_match_audit.py`, `m3/validate_against_known_cases.py` |
| M3′ observer / baseline | `m3prime/m3prime_llada_generate.py`, `m3prime/05_freeze_generation.py` |
| M3′ judging / support | `m3prime/01_build_packet.py` … `m3prime/04_support_check.py` |
| M3′ BUILD capture / gate / τ | `m3prime/06_build_capture.py`, `m3prime/07_byte_identity_and_freeze.py`, `m3prime/08_gate_fit.py`, `m3prime/09_gate_qualification.py`, `m3prime/build_final_result.py` |

## Provenance gaps

These are recorded honestly. **No reconstruction is published.** Any future reconstruction must be labeled `RECONSTRUCTED_FROM_FROZEN_SPEC`, never `ORIGINAL_EXECUTOR`.

| Gap | What exists |
|---|---|
| **V2 direction averaging / normalization.** No standalone original script on disk. | The frozen spec (handoff §18–19: first-k=min(32,…) response-token mean at L16, per-pair safe−harmful, family means, `V_ALL = normalize((m_DIJA+m_RENELLM)/2)`), `DIRECTION_FREEZE.json` (`0584fed6…`) and the directions container SHA (`e43916ff…`). |
| **V2 β calibration builder.** No standalone original script on disk. | The frozen rule (β = 2 × median own-direction BUILD gap; `β_ALL` = equal-family weighted median) and `BETA_FREEZE.json` (`3e41744b…`). |
| **M3′ NEW_DEV sizing.** The script is not on disk. | Its frozen output `NEW_DEV_SIZING_SIMULATION.json` (adopted N = 165). |
| **M3′ ReNeLLM construction executor.** No executor SHA is recorded in the M3′ construction freeze. | Frozen outputs and the scenario-match PASS. The M2 ReNeLLM executor (`shared/run_m2_renellm_construction_v1.py`) is the likely executor, but that is **unverified**. |

The handoff's "direction construction script" label on `fa5ad065…` is a mislabel; see [`PROVENANCE_ERRATA.md`](PROVENANCE_ERRATA.md).

## Historical RRAE pipeline

The RRAE representation-learning pipeline and the earlier frozen 35-pair steering replication (`rrae/`, `steering/`, `configs/final_rrae_steering.yaml`, `results/frozen_35_summary.json`) remain runnable as documented in the root README. They are historical results, not the final program conclusion.
