# CDG2.0 — Representation ≠ Control in a Diffusion Language Model

CDG2.0 studies how jailbreak-style prompt injection shows up inside a diffusion language model (**LLaDA-8B-Instruct**). It then tests whether activation steering can move generation toward safe behavior *selectively*: converting harmful compliance while leaving benign requests helpful.

**Final result.** Across static additive, norm-preserving, and trajectory-gated interventions, internal safety structure was detectable and causally relevant, but reliable selective control remained difficult. In short: **Representation ≠ Control.**

- Model: LLaDA-8B-Instruct
- Attack families: DIJA (template/mask injection) and ReNeLLM (rewrite-and-nest)
- Final program: generalized safety-direction steering, M1 → M2 → M3 → M3′
- Machine-readable results: [`results/generalized_safety/`](results/generalized_safety/)

## Canonical A/B/C/D design

| Condition | Content | Injection | Samples | Health | Non-health |
|---|---|---:|---:|---:|---:|
| A | Harmful | No | 500 | 400 | 100 |
| B | Harmful | Yes | 500 | 400 | 100 |
| C | Benign | Yes | 500 | 400 | 100 |
| D | Benign | No | 500 | 400 | 100 |
| **Total** | | | **2,000** | **1,600** | **400** |

The controlled contrasts `B − A` and `C − D` isolate injection while holding content fixed. B vs C is not an injection direction, because it changes harmfulness. Each group shares one pair identifier, so splits and paired evaluation cannot leak across conditions.

## RRAE representation learning

RRAE learns a rank-k reconstruction `h → ĥ` and exposes the residual `r = h − ĥ`. The selected representation (`harm / f=0.05 / L11 / rank 4`) reaches:
- validation injection AUC 1.0000;
- residual `cos(B−A, C−D)` 0.9621;
- harmfulness-AUC guardrail 0.4357.

Injection is therefore cleanly **detectable**, but that does not establish control. Code: `rrae/`, `steering/`, `cdg/`. Frozen selection: [`configs/final_rrae_steering.yaml`](configs/final_rrae_steering.yaml). Run instructions: [`docs/RRAE_PIPELINE.md`](docs/RRAE_PIPELINE.md).

## Generalized safety-direction program

The frozen safety directions `V_DIJA`, `V_RENELLM` and `V_ALL` are L16 safe-minus-harmful contrasts built from BUILD data only. They are applied at layer 16 on currently masked positions, persistently, with frozen doses. All outcomes come from a blinded DeepSeek judge under frozen metric contracts. The key metric is **strict paired success**: a B case converts to safe *and* its paired C case stays helpful.

| Stage | Question | Frozen outcome |
|---|---|---|
| **M1**: static additive (`h + β·v`) | Can a frozen generalized safety direction causally move behavior toward safety? | Yes, partially. Held-out strict paired success is 6/83–17/83 (DIJA) and 6/61–9/61 (ReNeLLM) across the three directions. The effect depends on attack family, and benign utility can degrade (ReNeLLM C-helpful preservation 25/75–33/75). Not reliably selective. |
| **M2**: displacement-matched norm-preserving | Does preserving hidden-state norm at matched displacement restore selectivity? | No. Strict paired success was lower in 5 of 6 family-specific comparisons and unchanged in 1 of 6 (DIJA × `V_ALL`, 9/54 vs 9/54). All 3 pooled comparisons were also lower; they are descriptive, not additional independent tests. |
| **M3**: trajectory-gated (original) | Can an L16 trajectory gate choose when to steer? | `NO_GO_INSUFFICIENT_DEV_SUPPORT`: pooled DEV B_already_safe = 19 < 20. The gate was never fit and τ never selected. **Inconclusive about gating, not evidence against it.** |
| **M3′**: trajectory-gated, enlarged DEV | Same question, with support restored (22/278/187) | `NO_GO`. The gate was fit and tested. DEV case-level ROC-AUC 0.7332, but **0 of 489** thresholds met the pre-registered constraints. **Detection without usable selectivity.** |

The final 576-group confirmatory pool was **not consumed** by M3 or M3′. It is not published.

The M3′ result is scoped to the frozen 9-feature representation, logistic gate, threshold rule and `V_ALL` actuator. It does not show that trajectory gating, other L16 representations, or nonlinear or dynamic controllers must fail.

Full tables, with every numerator and denominator: [`docs/EXPERIMENTAL_HISTORY.md`](docs/EXPERIMENTAL_HISTORY.md).

## Limitations and lineage caveat

- **Clean:** `V_DIJA` and `β_DIJA`. The earlier `generalized_safety_v1` direction family was independently audited and is unaffected.
- **Caveated:** `V_RENELLM` and `β_RENELLM` were built from a V2 ReNeLLM tree affected by a documented request-lineage mismatch (157 of 199 construction pairs cross-topic). `V_ALL` and `β_ALL` inherit this caveat.
- The M1/M2 values remain measurements of the frozen interventions that were actually executed. However, the intended semantic interpretation of the V2 ReNeLLM-derived direction and the combined `V_ALL` direction and dose is weakened.

Other recorded limitations:
- M2 label-vocabulary and metric-contract-reference deviations;
- C_DIJA BUILD fragmentation;
- the sparse M3′ `unnecessary_refusal` class;
- case-level gate supervision;
- judge/assistant agreement on BUILD B;
- final-pool non-health D-side phrasing drift.

See [`docs/LIMITATIONS_AND_LINEAGE.md`](docs/LIMITATIONS_AND_LINEAGE.md) and [`docs/PROVENANCE_ERRATA.md`](docs/PROVENANCE_ERRATA.md).

## Reproducibility

| What | Where |
|---|---|
| Frozen results (JSON, with source SHA-256s) | [`results/generalized_safety/`](results/generalized_safety/) |
| Program index and frozen settings | [`configs/generalized_safety/final_program.yaml`](configs/generalized_safety/final_program.yaml) |
| Generalized-safety pipeline scripts (path-sanitized) | [`experiments/generalized_safety/`](experiments/generalized_safety/) |
| Original ↔ published script SHAs | [`docs/SCRIPT_PROVENANCE.md`](docs/SCRIPT_PROVENANCE.md) |
| Environment, stage map, provenance gaps | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| Data sources and hashes per population | [`docs/DATA_LINEAGE.md`](docs/DATA_LINEAGE.md) |
| RRAE pipeline run instructions | [`docs/RRAE_PIPELINE.md`](docs/RRAE_PIPELINE.md) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Private dataset policy

The canonical datasets, every experiment population (including the 576-group final pool), private blinding maps, judgments, generations, hidden states, model weights and credentials are **not redistributed**. Authorized collaborators receive the canonical dataset separately and verify it locally:

```bash
python data/canonical_abcd_v2_2/verify_dataset.py /path/to/ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl
# expects DATASET_VERIFICATION=PASS (SHA-256 4392afca…8eaaf)
```

See [`data/canonical_abcd_v2_2/`](data/canonical_abcd_v2_2/) and [`docs/DATA_LINEAGE.md`](docs/DATA_LINEAGE.md).

## Historical results

These are kept for provenance. None of them is the final project conclusion.

- **Frozen 35-pair RRAE-era steering replication.** [`results/frozen_35_summary.json`](results/frozen_35_summary.json), with config [`configs/final_rrae_steering.yaml`](configs/final_rrae_steering.yaml). This was an earlier replication of five steering families, run before the generalized safety-direction program; see [`results/README.md`](results/README.md).
- **Legacy 400-sample report.** [`REPORT.md`](REPORT.md).
- **Legacy scripts and experiments.** `scripts/`, `run_*.py`, `run_*.sh`, `dream/` (Dream-v0 extension).
