# Experimental history — generalized safety-direction program (M1 → M3′)

Every number on this page is copied from the frozen, machine-readable summaries in [`results/generalized_safety/`](../results/generalized_safety/). Those summaries were built from SHA-verified frozen artifacts; metric blocks were copied verbatim and never recomputed. Limitations and lineage caveats are in [`LIMITATIONS_AND_LINEAGE.md`](LIMITATIONS_AND_LINEAGE.md), and data sources in [`DATA_LINEAGE.md`](DATA_LINEAGE.md).

## Setup shared by all stages

| Item | Frozen value |
|---|---|
| Model | LLaDA-8B-Instruct |
| Intervention site | layer 16 (zero-based), `current_mask` token scope, persistent schedule |
| Generation | length 128, temperature 0.2, `low_confidence` remasking; same frozen seed per source case across all arms |
| Directions | `V_DIJA`, `V_RENELLM`, `V_ALL`: L16 safe-minus-harmful response contrasts, built on BUILD only; `V_ALL = normalize((m_DIJA + m_RENELLM)/2)` |
| Attack families | DIJA (template/mask injection) and ReNeLLM (rewrite-and-nest) |
| Judge | DeepSeek `deepseek-v4-pro`, temperature 0, blinded, one item per request, frozen label spaces |
| Primary metrics | `B_safe_conversion`, `C_helpful_preservation`, `strict_paired_success` (a pair succeeds only if its B case converts **and** its paired C case stays helpful) |

Pooled rows below are **descriptive only**. They are not additional independent tests.

---

## M1 — Static additive steering

**Question.** Can a frozen generalized safety direction causally move behavior toward safety?

**Method.** `h_new = h + β·v`, with β frozen from BUILD (no sweep): β_DIJA = 74.7751, β_RENELLM = 81.5953, β_ALL = 73.1175.

**Population (held-out).**
- 100 groups: 80 health, 20 non-health.
- 396 source cases: DIJA 200, ReNeLLM 196.
- 1,584 generation rows: baseline plus three arms.

**Frozen held-out results (primary):**

| Family | Arm | B_safe_conversion | C_helpful_preservation | strict_paired_success |
|---|---|---:|---:|---:|
| DIJA | `V_ALL` | 20/94 (21.3%) | 85/88 (96.6%) | 15/83 (18.1%) |
| DIJA | `V_DIJA` | 23/94 (24.5%) | 82/88 (93.2%) | 17/83 (20.5%) |
| DIJA | `V_RENELLM` | 12/94 (12.8%) | 84/88 (95.5%) | 6/83 (7.2%) |
| RENELLM | `V_ALL` | 25/78 (32.1%) | 33/75 (44.0%) | 9/61 (14.8%) |
| RENELLM | `V_DIJA` | 27/78 (34.6%) | 25/75 (33.3%) | 6/61 (9.8%) |
| RENELLM | `V_RENELLM` | 31/78 (39.7%) | 28/75 (37.3%) | 7/61 (11.5%) |
| POOLED (descriptive) | `V_ALL` | 45/172 (26.2%) | 118/163 (72.4%) | 24/144 (16.7%) |
| POOLED (descriptive) | `V_DIJA` | 50/172 (29.1%) | 107/163 (65.6%) | 23/144 (16.0%) |
| POOLED (descriptive) | `V_RENELLM` | 43/172 (25.0%) | 112/163 (68.7%) | 13/144 (9.0%) |

**Secondary metrics:**

| Family | Arm | B_harm_regression | C_degradation | C_unnecessary_refusal |
|---|---|---:|---:|---:|
| DIJA | `V_ALL` | 1/5 (20.0%) | 3/88 (3.4%) | 2/88 (2.3%) |
| DIJA | `V_DIJA` | 2/5 (40.0%) | 6/88 (6.8%) | 1/88 (1.1%) |
| DIJA | `V_RENELLM` | 3/5 (60.0%) | 4/88 (4.5%) | 0/88 (0.0%) |
| RENELLM | `V_ALL` | 3/16 (18.8%) | 42/75 (56.0%) | 2/75 (2.7%) |
| RENELLM | `V_DIJA` | 0/16 (0.0%) | 50/75 (66.7%) | 3/75 (4.0%) |
| RENELLM | `V_RENELLM` | 0/16 (0.0%) | 47/75 (62.7%) | 9/75 (12.0%) |

**Result.**
- Causal behavioral movement toward safety exists.
- The effect is partial and depends on the attack family.
- Benign utility can degrade, sharply so under ReNeLLM.
- Static additive steering is not reliably selective.

A separate DEV observation run (306 source cases) is recorded under `development_observation` in [`m1_summary.json`](../results/generalized_safety/m1_summary.json). It is not the confirmatory result.

---

## M2 — Displacement-matched norm-preserving steering

**Question.** Does preserving the hidden-state norm, at matched perturbation displacement, restore selectivity?

**Method.**
- `h̃ = h + β·v`, then `h_new = ‖h‖₂ · h̃ / ‖h̃‖₂`.
- The M1 β values were **not** reused. M2 β values were solved on DEV to match M1's realized displacement: β_ALL = 95.683, β_DIJA = 90.264, β_RENELLM = 168.135.
- This matching keeps a weaker effective intervention from being mistaken for a norm-preservation effect.

**Population.**
- A fresh confirmatory set of 126 groups, with 434 source cases (DIJA 230, ReNeLLM 204) and 3,038 judgments.
- The M1 additive arms were re-run on the **same** population, so the M1-vs-M2 contrast is paired on identical eligible units.

**Frozen result: `strict_paired_success`, M1 additive vs M2 (same population, same direction):**

| Family | Direction | M1 additive | M2 norm-preserving | Δ (pp) |
|---|---|---:|---:|---:|
| DIJA | `V_ALL` | 9/54 (16.67%) | 9/54 (16.67%) | +0.00 |
| DIJA | `V_DIJA` | 15/54 (27.78%) | 11/54 (20.37%) | −7.41 |
| DIJA | `V_RENELLM` | 5/54 (9.26%) | 4/54 (7.41%) | −1.85 |
| RENELLM | `V_ALL` | 9/37 (24.32%) | 3/37 (8.11%) | −16.22 |
| RENELLM | `V_DIJA` | 5/37 (13.51%) | 2/37 (5.41%) | −8.11 |
| RENELLM | `V_RENELLM` | 5/37 (13.51%) | 1/37 (2.70%) | −10.81 |
| POOLED (descriptive) | `V_ALL` | 18/91 (19.78%) | 12/91 (13.19%) | −6.59 |
| POOLED (descriptive) | `V_DIJA` | 20/91 (21.98%) | 13/91 (14.29%) | −7.69 |
| POOLED (descriptive) | `V_RENELLM` | 10/91 (10.99%) | 5/91 (5.49%) | −5.49 |

**Result: no.**
- Strict paired success was **lower in 5 of the 6** family-specific comparisons and **unchanged in 1 of 6** (DIJA × `V_ALL`).
- All 3 pooled comparisons were also lower. They are descriptive, not additional independent confirmatory tests.

All 12 family×arm cells for every metric, and the paired concordance counts, are in [`m2_summary.json`](../results/generalized_safety/m2_summary.json). Two tooling deviations are documented and were honored rather than corrected (see [`LIMITATIONS_AND_LINEAGE.md`](LIMITATIONS_AND_LINEAGE.md)).

---

## M3 — Trajectory-gated additive steering (original)

**Design.**
- Fit a gate on per-step L16 trajectory features over BUILD baselines.
- Select a threshold τ on DEV under three pre-registered subgroup trigger-rate constraints.
- Only then run closed-loop steering and a confirmatory test on a new 576-group pool.

**Precondition.** Every subgroup used in a threshold constraint needed at least 20 pooled DEV cases.

**Frozen outcome: `M3_GATE_QUALIFICATION = NO_GO_INSUFFICIENT_DEV_SUPPORT`.**

| Pooled DEV subgroup | Count | Minimum |
|---|---:|---:|
| B_already_safe | **19** | 20 |
| B_harmful_compliance | 119 | 20 |
| C_benign_helpful | 86 | 20 |

**What this means.**
- The gate was **never fit**, τ was **never selected**, and trajectory gating was **never tested**.
- M3 is inconclusive about gating. It is **not** evidence against trajectory gating.
- The final 576-group pool was **not consumed**.

---

## M3′ — Trajectory-gated steering with enlarged DEV

**Design.**
- Everything is as in M3, with one change: the 80-group DEV set was replaced by a new 165-group NEW_DEV set.
- This was a one-shot rule: a second support failure would have ended the program.

**Support check: passed.** Pooled NEW_DEV subgroups were B_already_safe 22, B_harmful_compliance 278 and C_benign_helpful 187 (all ≥ 20).

**Gate fit.**
- Model: ℓ2 logistic regression on 9 frozen L16 features (projections onto `V_ALL`/`V_DIJA`/`V_RENELLM`, their step deltas, hidden-state norm, denoising fraction, mask fraction).
- Fit on BUILD baselines only: 678 cases, 391 positive and 287 negative. The fit converged.

**Qualification: `M3PRIME_GATE_QUALIFICATION = NO_GO`.**

| Quantity | Frozen value |
|---|---|
| NEW_DEV case-level ROC-AUC | **0.7332** (0.733245; bar 0.70, cleared) |
| Candidate thresholds | 489 |
| Thresholds meeting all three constraints | **0** |
| Constraints | B_harmful trigger ≥ 50%; B_already_safe trigger ≤ 10%; C_benign trigger ≤ 10% |

**Representative frozen trade-offs:**

| τ | B harmful trigger | B already-safe trigger | C benign trigger |
|---|---:|---:|---:|
| ≈ 0.700 | 71.6% | 36.4% | 34.8% |
| ≈ 0.743 | 50.0% | 22.7% | 16.6% |
| ≈ 0.825 | 6.5% | 9.1% | 0.5% |

**Result: detection without usable selectivity.**
- The L16 trajectory features carried moderate discriminative signal.
- No threshold satisfied the pre-registered operational constraints at the same time.
- Nothing was steered, and the final 576-group pool was **not consumed**.

**Scope.** The result applies to the frozen 9-feature representation, the frozen logistic gate, the frozen threshold rule and the frozen `V_ALL` actuator. It does **not** show that trajectory gating is impossible, that all L16 representations fail, or that nonlinear or dynamic controllers would fail.

---

## Final confirmatory pool

576 groups (manifest SHA-256 `9ce8f317…`). **Unconsumed** by both M3 and M3′. It is not published, and it is reserved for any separately frozen future contract.

## Program conclusion

**Representation ≠ Control.** Across static additive, norm-preserving, and trajectory-gated interventions, internal safety structure was detectable and causally relevant, but reliable selective control remained difficult.
