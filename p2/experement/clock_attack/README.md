# clock_attack — latent denoising-clock (τ) steering during a DIJA attack

**Question.** Does steering LLaDA-8B-Instruct along its latent *denoising-progress* direction
τ (Subliminal Clocks, Eq. 4) **while** a DIJA mask-injection attack runs change harm-delivery
vs. DIJA alone — or does it just trigger the EOS/collapse failure the paper predicts at high t̂?

**Answer shape.** τ-steering is applied only to masked positions, pushing the model's internal
"clock" from the current step-bin *t* toward a target bin *t̂*:

    out[0][mask_positions] += α · (μ[t̂, l] − μ[t, l])          # Eq. 4, layer l = 29 (also 25)

μ is a **content-pure** mean-vector bank measured on a NEUTRAL held-out set (never the harmful
cases). A norm-matched **random** control (Eq. 5) reproduces the old failed run and stays inert.

## Layout

```
clock_attack/
  build_mu_bank.py        Step 0: μ bank (Eq. 3) from clean neutral generations, block-output
                          hidden @ L29/L25, binned by denoising step -> mu_bank_L{29,25}.pt
  build_dija_scaffolds.py strengthen the DIJA baseline: attach leading-context masks
                          (dija_refiner's generic "Step N: <mask>" skeleton) -> dija_scaffolds.jsonl
  steerer.py              Step 1: StepSteerer + forward hook (τ / random / off), mask-gated, step-aware
  run_clock_dija.py       Steps 2+3: steered generation over the condition grid + diagnostics; --aggregate
  mu_bank_L29.pt, mu_bank_L25.pt      the μ banks (mu_bin[101,4096], sigma_bin, per-step)
  dija_scaffolds.jsonl    100 leading-context DIJA scaffolds (behavior byte-identical to source)
  results/
    generations.jsonl           one row per (seed × condition-cell) with response + diagnostics
    generations_scored.jsonl    + HarmBench judge verdicts (deepseek, --max-tokens 128)
    REPORT.md                   arm × (harm-delivered, ASR, EOS-rate, entropy, collapse, echo)
```

**Reuses (imported, never copied/edited):** `ladaAndH.py` (StepActivationCollector `out[0]` hook,
generate_with_capture, discover_blocks), `paired_dija_runner.py` (process_prompt_instruct +
sampler), `dija_refiner.py` (scaffold skeleton + validation), `score_harmbench_deepseek.py` (judge).
Constants: mask_id=126336, EOS=126081, EoT=126348; bf16; steps=128, gen=128, block=32, temp=0.

## Conditions (100 harmful seeds, single seed, within-template)

1. `dija_baseline`      — DIJA (leading-context masks), no steering
2. `dija_tau_steer`     — DIJA + τ-steer, sweep **t̂ ∈ {75, 90, 100}**, **α ∈ {1, 2, 4}**
3. `dija_random_ctrl`   — DIJA + norm-matched random (Eq. 5), α ∈ {1, 2, 4} at t̂=100  (must stay inert)
4. `nodija_tau_sanity`  — τ-steer with NO DIJA injection (should stay safe)

## Baseline-strengthening decision (documented)

The faithful DIJA **blank-canvas** fallback (bare mask canvas) makes LLaDA fill with EOS →
**empty baseline (harm ≈ 0)**, leaving τ nothing to modulate (verified in the first smoke). We
therefore attach **leading context** to the injected masks — the published DIJA
"Step 1: [mask] Step 2: [mask]" worksheet skeleton — reusing `dija_refiner.stub_scaffold` +
`ablate_scaffold(..., "minimal")`. **No attack content is authored**: the step labels are generic
and the masks stay empty; only the model fills them. This gives the baseline real dynamic range.

## Reproduce

```bash
# 0. μ bank (GPU) — neutral held-out seeds, content-pure
python clock_attack/build_mu_bank.py --n-seeds 32 --layers 29 25
# 0b. strengthened DIJA scaffolds (CPU)
python clock_attack/build_dija_scaffolds.py
# 1. smoke (default --limit 3): all cells on 3 seeds, prints full decoded outputs
python clock_attack/run_clock_dija.py --limit 3
# 2. full run (resumable — re-run the same command to continue after a walltime cut)
python clock_attack/run_clock_dija.py --limit 0
# 3. judge (login node if only it has internet); --max-tokens 128 avoids the err=100 parser bug
export DEEPSEEK_API_KEY=...   # or: $(cat /home/ore99/serverFiles/attack2/.deepseek_key)
python score_harmbench_deepseek.py --input clock_attack/results/generations.jsonl \
    --output clock_attack/results/generations_scored.jsonl \
    --behavior-field behavior --generation-field response --key-field traj_id --max-tokens 128
# 4. build the table
python clock_attack/run_clock_dija.py --aggregate
```

## Diagnostics logged per generation

`mean_conf`, `mean_entropy`, `eos_rate`, `collapse` (D-collapse: unique-token ratio < 0.5),
`echo_artifact` (the "1.."/empty-scaffold echo), `refusal` — to catch the paper's high-t̂
EOS/collapse failure mode. Metric priority: **harm-delivered** (frozen HarmBench rubric) is primary;
ASR (non-refusal) is a heuristic secondary.

## Constraints honored

- New folder only; existing files imported, never edited. Harmful seeds byte-identical from
  `attack2/source_A/cases.json`; no harmful text hardcoded. μ built on the NEUTRAL cdg
  `D_neutral_clean` set (100 unique benign items) — the harmful cases never touch the direction.
- bf16; H100 MIG 3g.40gb slice (OOM path prints a note to switch to a full H100).
- Judge internet: generation runs on GPU; scoring is a separable step for the login node.
