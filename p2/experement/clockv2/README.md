# clockv2 — Plan B: defensive, READ-ONLY interaction study

Clean vs DIJA-injected on the same 100 seeds (`A000..A099`), comparing what three
read-only probes see inside LLaDA-8B-Instruct. **No steering. No activation is ever
modified. Nothing outside `clockv2/` is edited.**

## Absolute constraints (enforced)

| constraint | how it is enforced |
|---|---|
| probes READ hidden states only | every hook returns `None`; `common.assert_readonly_hooks()` drives a probe tensor through each registered hook and raises if any returns a replacement output. Verified to catch a mutating hook of exactly the shape `StepSteerer` uses. |
| ZERO steering | `clock_attack.steerer.StepSteerer` is **never imported**. Its hook does `h[0, mrow, :] += vec`. Out of scope by construction. |
| no safety-breaking intervention | `v_refusal` is built as a measurement axis and only ever dotted with hidden states. Never added, never used to generate, never used to reduce refusal. |
| no harmful text authored | all prompts come from files already on disk. Held-out DIJA scaffolds use the existing verified `dija_refiner.stub_scaffold` + `ablate_scaffold("minimal")` generic skeleton. |
| new folder only | `dija_attack/`, `clock_attack/`, `ladaAndH.py` are import-only. |

Interpreter: `/home/ore99/env_llada/bin/python` (py3.10, torch 2.6, numpy 1.26).
Not `.venv` — it has no numpy and `cdg_denoise` imports it.
Outputs live on `/scratch` (`probes/`, `results/`, `heldout/` are symlinks); `/home` is 99% full.

## Phase 0 findings — recapture is mandatory

**clock_attack never saved hidden states.** There is no `refs_hook`/`turn_XX.pt` under
`clock_attack/`; `refs_hook` lives in root `hidden_states.py` and served older runs.
`run_clock_dija.py` only *reads* the μ-banks. The `.pt` files that do exist elsewhere fail
on three independent grounds:

| saved set | layers | steps × positions | seeds |
|---|---|---|---|
| `hidden_states_test/test_traj` | 0,4,8,12,16,20,24,28,31 | 64 × 8 | 2 turns, not A-ids |
| `seed_states/seed_run` | same | 64 × 8 | 10 turns, not A-ids |
| `shield_dataset/run_with_heuristic` | **12, 24 only** | 128 × 128 | 10 medical turns |

τ needs L25/L29 — **no saved set contains either**. None are the A-set. None has a
clean-vs-DIJA pair. Nothing to reuse.

## Two corrected premises

**There is no τ probe in clock_attack — only a τ steerer.** The μ-bank is a read-only data
artifact (reusable in principle), but the shipped bank's meta says `block_length=32`
(a 4-block restricted schedule). Phase 2 runs both conditions on the **unified
`fill_all_masks` schedule** — the only way DIJA's prompt-embedded blanks are fillable
(`cdg_denoise.py:57-83`; `block_length` is ignored entirely on that path). Different
schedules ⇒ different hidden trajectories ⇒ the step→hidden mapping would not transfer.
So `build_tau.py` **rebuilds** the bank under the exact Phase-2 schedule and extends it
from {25,29} to L16–L31.

For a *clean* prompt (no prompt blanks), `fill_all_masks=True` is exactly equivalent to
`ladaAndH.generate_with_capture` at `block_length=gen_length=128`. Both conditions
therefore share one schedule and the rebuilt bank is valid for both.

**`v_shared` was never an SVD injection signature.** `clock/execute_latent_clock_attack.py:66-68`
is `manual_seed(1337); randn(4096)` normalized — a random direction, merely *named*
`v_shared`. There is no SVD anywhere outside `external/`. It is kept, honestly relabelled
`v_random_null`, as the null floor; a real `v_injection_svd` is built alongside it.

## Leakage control

The eval set is `cases.json` (`A000..A099`, 100 medical-harmful). Probes must not see it.

| pool | n | max Jaccard vs A-set | verdict |
|---|---|---|---|
| `curated/seeds_curated.jsonl` | 100 | **1.00 (99/100 exact)** | **IS the eval set — excluded** |
| `neutral/case_*.json` | 100 | same structure | **eval-set counterparts — excluded** |
| `clock_attack/dija_scaffolds.jsonl` | 100 | 100/100 id match | **IS the eval set — excluded** |
| `unsafe.jsonl` (MedSafetyBench) | 50 → **43** | 0.80 → 0.40 after filter | harmful half |
| `D_neutral_clean` (`D000..`) | 100 → **99** | 0.41 → 0.375 after filter | harmless half / τ seeds |

Exact-match testing was **not** sufficient: `unsafe.jsonl` showed 0 exact overlap yet
contained near-duplicates of eval cases at J=0.80 and J=0.70 (the A-set is a first-person
paraphrase of MedSafetyBench). Everything within J>0.4 of any eval behavior is dropped.

Both contrastive halves are **medical**, so `mean_harmful − mean_harmless` isolates harm
rather than domain. τ's fit seeds and convention-verification seeds are disjoint.

## The probes (all read-only)

- `build_tau.py` → `probes/tau_bank.pt` — denoising-progress bank + decoder. τ convention
  is **verified empirically** by correlating decoded τ against the physical canvas clock
  `1 − mask_ratio` on held-out neutral runs; the sign is reported, never assumed.
- `build_vshared.py` → `probes/v_random_null.pt` (seed-1337 floor) and
  `probes/v_injection_svd.pt` (top right singular vector of centred clean→DIJA prompt-only
  differences; sign fixed empirically so +v = "more injected").
- `build_vrefusal.py` → `probes/v_refusal.pt` — `normalize(mean_harmful − mean_harmless)`,
  Arditi Eq.1–2, prompt-only.

Each probe stores **both** extraction points (`last` prompt token and `mean` over prompt
tokens) so Phase 2 can read on the axis that matches its read point.

## Phase 1 results (built, verified)

| probe | artifact | headline sanity |
|---|---|---|
| τ | `tau_bank.pt` | `r(tau_soft, 1−mask_ratio)` = **+0.956 … +0.978** across L16–L31 → convention **empirically confirmed positive**. L25 +0.967, L29 +0.971. MAE ≈ 0.12. |
| injection | `v_injection_svd.pt` | PC1 var-explained only **0.10–0.13** (mean) → signature is **diffuse**. Usable at mean-pool L16–L27; collapses deeper (L30: proj 2.8 ± 40). |
| null | `v_random_null.pt` | `cos(v_injection, v_random_null) = −0.005` → random axis carries **zero** injection signal, as predicted. norm 0.999959 (bf16 rounding in the original, reproduced faithfully). |
| refusal | `v_refusal.pt` | `‖mean_h − mean_l‖` 16.5 (L16) → 196 (L31); 0.20 → 0.65 relative to `‖mean_l‖`. Real separation, **not** degenerate. |

## Locked decisions

- **temp = 0 for both conditions.** Deterministic; matches the τ bank. Deviates from the
  verified DIJA runner's 0.2 — this study reads representations, it does not measure ASR.
- **mean-pooling is the primary read point**, last-token secondary. Forced by the geometry
  below.

## Cross-probe geometry (measured at L29, drives Phase 3 framing)

| pair | cos | consequence |
|---|---|---|
| `v_refusal_last` vs `v_injection_last` | **+0.654** | at last-token the two axes are **65% aligned** (both fit on the same 43 held-out prompts) — "moves along refusal" and "moves along injection" would be near-restatements |
| `v_refusal_mean` vs `v_injection_mean` | −0.052 | at mean-pool the axes are **independent** → separable claims |
| `v_refusal_last` vs `v_refusal_mean` | +0.502 | pooling conventions are **not** interchangeable; every claim must name its read point |
| any probe vs `v_random_null` | ≈ ±0.005 | null floor behaves as a null floor |

## THE confound to beat in Phase 3

Clean generation fills **exactly one token per step** (verified: `t0=1.000, t64=0.500,
t127=0.008`), so `tau_from_mask = t/128` deterministically and identically for all 100 clean
ids. Under `fill_all_masks` the 128-step budget spreads over `gen_length + n_inject` masks,
so DIJA's **response** canvas drains at a different rate **by arithmetic, before any
representational effect exists**. Demonstrated on a stub model:

    clean (n_inject=0):  resp [1.00, 0.88, 0.75, 0.62, 0.50, 0.38, 0.25, 0.12]
    dija  (n_inject=4):  resp [1.00, 0.88, 0.62, 0.50, 0.38, 0.25, 0.25, 0.12]

A raw clean-vs-DIJA comparison of `disc_t = tau_read − (1 − mask_ratio)` therefore partly
measures **bookkeeping, not the model**. `capture_denoise` logs `mask_ratio_global` and
`n_inject` alongside; **Phase 3 must regress `disc_t` on `n_inject`** before claiming the
discrepancy is representational. This is the single biggest threat to the headline claim.

## Known limitations (carry into the report)

- **Scaffold style gap.** Eval scaffolds are Qwen-refined with trailing context
  (`"Procedure: <mask:20>. Proceed with caution…"`); the held-out fit uses the bare
  `stub_scaffold` skeleton (`"Step 1: <mask:14>"`). `v_injection_svd` is therefore a
  **structural** injection signature, not a refined-DIJA signature. Closing the gap would
  require authoring scaffold text or re-running the Qwen refiner; we do neither.
- **temp=0 deviation.** The verified DIJA runner used `temperature=0.2`; we run both
  conditions at `temperature=0.0` for determinism (this study reads representations, it
  does not measure ASR). Single-seed either way, but at temp=0 the trajectory is exact.
- **τ is not localized, and its verification is partly circular.** τ decodes nearly
  equally well at *every* layer (r = 0.956–0.978, essentially flat), and because the clean
  canvas fills deterministically at 1 token/step, `mu_step[t]` simply *is* the mean hidden
  state at step t — so nearest-μ decoding approximates decoding the step index. The
  verification establishes the decoder works and its sign is positive; it does **not**
  establish τ as a nontrivial internal clock. It may largely reflect the visible fill
  level. Do not claim otherwise.
- **The injection axis is weak.** PC1 explains 10–13% of the clean→DIJA difference, so the
  signature is diffuse rather than a single direction; treat `vshared_proj` as a coarse
  reference, not a precise detector.
- **Cross-position application.** `v_refusal` is extracted at the last prompt token but
  Phase 2 also projects the response-canvas pooled state onto it. In a diffusion LM the
  prompt's hidden state evolves per step (bidirectional attention), so the last-prompt-token
  read point is tracked per step as the matched comparison.
- `cos(v_refusal_last, v_refusal_mean)` ≈ 0.26–0.54 — the two pooling conventions are **not**
  interchangeable; conclusions must name their read point.
