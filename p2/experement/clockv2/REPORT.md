# Reading the injection signature: a four-arm, read-only interaction study on LLaDA-8B-Instruct

## Abstract

Reading — never modifying — LLaDA-8B-Instruct hidden states across four matched prompt arms over the
same 100 medical-harm behaviors, we ask what the DIJA template-injection signature `v_injection_svd`
actually encodes. Against a benign scaffold matched for length, format, `<mask:N>` count **and**
first-person operational stance, the axis still separates injected-harmful from injected-benign prompts
at **AUC 0.958 / 0.975 / 0.977 at L25 / L26 / L27** (read point = prompt_mean), where a random direction
sits at chance (AUC 0.519) — so the separation is not a length or format artifact. An exact per-id
identity splits the original clean-vs-DIJA separation into structure and content: the content share is
**27% (L25), 30% (L26), 48% (L27)** — depth-dependent, not one number. Both results are seed-stable
across two temp=0.2 resamples of the full denoising trajectory (dz ± 0.05; split ± 0.2pp; 100/100 ids
sign-stable). Three things are null and reported as such: **τ** ("internal clock") is circular — it
reads canvas fill level, trivially available from token bookkeeping; **`v_refusal`** fails its
pre-registered floor criterion and is additionally in-sample and question-form confounded; and the
**random-direction null** is the floor that makes the rest interpretable, reaching AUC 0.990 on the
naive contrast. One confound stays open: the two benign arms carry *different* source contaminants, so
the residual is `stance − authorship` — which bounds authorship as the smaller contributor without
isolating harm.

**Status:** complete. Every load-bearing number at L25–L27 is seed-verified across two temp=0.2
resamples; §7 states exactly what is not.
**Method class:** READ-ONLY. Every probe is a projection of an unmodified hidden state onto a fixed
axis. No activation was written to at any point; `assert_readonly_hooks()` runs before every forward
pass in `capture.py`, `capture_benign.py` and `capture_seed.py`. No steering, no gating, no remasking,
no intervention of any kind was implemented. No harmful text was authored for this study.

---

## 1. Headline

**`v_injection_svd` carries a genuine, generalizing content component.** It survives exact matching of
prompt length, template format, `<mask:N>` count (`n_inject`), *and* first-person operational stance,
measured against a null floor that is literally zero on that contrast.

It is also **seed-stable**: resampling the denoising trajectory moves dz by ≤ ±0.05 and AUC by
≤ ±0.005, and all 100 ids keep their sign across seeds (§3a).

Final test, `dija − benign_op`, **read point = prompt_mean (PRIMARY)**, paired per-id deltas, n=100.
The seed columns are the two independent temp=0.2 resamples; `±` is their spread:

| layer | mean_d | dz | AUC | %same-sign | null dz | null AUC | ×null | dz across seeds | AUC across seeds |
|---|---|---|---|---|---|---|---|---|---|
| L25 | +6.11 | +1.82 | **0.958** | 98% | +0.03 | 0.519 | **53.0×** | **+1.82 ± 0.05** | **0.953 ± 0.005** |
| L26 | +9.30 | +2.08 | **0.975** | 100% | +0.18 | 0.567 | **11.7×** | **+2.15 ± 0.05** | **0.974 ± 0.002** |
| L27 | +15.41 | +2.25 | **0.977** | 99% | −0.11 | 0.465 | **19.7×** | **+2.37 ± 0.03** | **0.980 ± 0.000** |

The null column is the point. On `dija − clean` a random unit direction reaches dz=+2.58 / AUC=0.990
at L18 purely from the 104-token length gap — the original headline was never measured against a zero
floor. On `dija − benign_op` the random direction reads AUC=0.519 at L25: chance. With every
common-mode difference matched, there is nothing left for an arbitrary axis to absorb, and the
injection axis still separates the two arms at AUC 0.958–0.975.

This is the strongest claim the data supports, and it is narrower than "the axis detects injection":
see §7.

---

## 2. The quantitative core: an exact-identity split

Because every contrast is a paired per-id delta, the decomposition below is **arithmetic, not a fit**:

```
(dija − clean) ≡ (benign_x − clean) + (dija − benign_x)
     total          structure              content
```

**Read point = prompt_mean (PRIMARY).**

| layer | via arm | dija−clean | structure | content | struct % | content % |
|---|---|---|---|---|---|---|
| L25 | benign (arm 2, dataset-sourced) | +22.94 | +15.74 | +7.19 | 69% | 31% |
| L25 | **benign_op (arm 3, authored)** | +22.94 | +16.83 | +6.11 | **73%** | **27%** |
| L26 | benign (arm 2, dataset-sourced) | +31.18 | +19.89 | +11.29 | 64% | 36% |
| L26 | **benign_op (arm 3, authored)** | +31.18 | +21.88 | +9.30 | **70%** | **30%** |

**At L25–L26 the clean-vs-DIJA separation is ≈ 70% structure + 30% content.** The layer qualifier is
load-bearing: the split is *not* constant across depth (§2a), so "70/30" is a property of this band,
not of the axis.

Two independent benign arms agree on this. Arm 2's questions are drawn from an existing dataset pool
(`D_neutral_clean`); arm 3's are authored. They share no items. They differ in register by
construction (0/100 vs 100/100 first-person). And their content components land within
**15.7% ± 2.1 (L25) / 17.7% ± 1.4 (L26) / 22.5% ± 1.0 (L27)** of each other — seed-verified across two
temp=0.2 resamples (§7). Two arms carrying *different* source confounds, disagreeing by less than a
quarter on the quantity of interest, is the main reason to believe the split rather than either arm
alone.

**What that gap is — and is not.** It is not "the stance effect, isolated." Arm 2 and arm 3 differ in
*two* ways at once: register (informational vs operational) **and** source (dataset vs authored). So
the gap is `stance − authorship`, and neither arm isolates harm — each is contaminated differently.
Arm 3 does resolve the operational-vs-informational confound that arm 2 could not address, and it
resolves it in the direction that leaves the finding standing: the content component survives register
matching with only 16–22% shrinkage. But the residue it leaves behind is an authored-text confound,
not nothing. §7 states the bound this supports and the pool that would be needed to separate the two
cleanly.

### 2a. Seed variance of the split — and its depth dependence

The split is now on the same seed footing as the headline: the `clean` and `benign_op` arms were
re-captured at the same two temp=0.2 seeds as `dija`, and the whole identity was recomputed per seed
(the identity is *asserted* per id at each seed in `analyze_split_seed.py`, not assumed). **Read point =
prompt_mean (PRIMARY)**, via arm 3, n=100:

| layer | seed | total | structure | content | struct % | content % |
|---|---|---|---|---|---|---|
| L25 | 0 (temp=0.0) | +22.94 | +16.83 | +6.11 | 73.4% | 26.6% |
| L25 | 1 (temp=0.2) | +23.03 | +16.96 | +6.07 | 73.6% | 26.4% |
| L25 | 2 (temp=0.2) | +23.00 | +16.88 | +6.11 | 73.4% | 26.6% |
| L25 | **spread** | | | | **73.5% ± 0.2** | **26.5% ± 0.2** |
| L26 | 0 (temp=0.0) | +31.18 | +21.88 | +9.30 | 70.2% | 29.8% |
| L26 | 1 (temp=0.2) | +31.31 | +22.00 | +9.31 | 70.3% | 29.7% |
| L26 | 2 (temp=0.2) | +31.25 | +21.91 | +9.34 | 70.1% | 29.9% |
| L26 | **spread** | | | | **70.2% ± 0.1** | **29.8% ± 0.1** |
| L27 | 0 (temp=0.0) | +31.85 | +16.44 | +15.41 | 51.6% | 48.4% |
| L27 | 1 (temp=0.2) | +32.00 | +16.59 | +15.41 | 51.8% | 48.2% |
| L27 | 2 (temp=0.2) | +31.96 | +16.48 | +15.48 | 51.6% | 48.4% |
| L27 | **spread** | | | | **51.7% ± 0.2** | **48.3% ± 0.2** |

Per-id stability across the two seeds, for **every term** of the identity — **read point = prompt_mean
(PRIMARY)**:

| layer | r total | r structure | r content | sign agreement (total / struct / content) |
|---|---|---|---|---|
| L25 | 0.989 | 0.959 | 0.926 | 100% / 100% / 100% |
| L26 | 0.993 | 0.965 | 0.947 | 100% / 100% / 100% |
| L27 | 0.995 | 0.977 | 0.970 | 100% / 100% / 100% |

**Two conclusions, and the second qualifies §2.**

1. **The split is seed-stable.** Resampling the whole denoising trajectory moves the percentages by
   ≤ ±0.2pp at every layer; per-id deltas correlate at r = 0.93–0.99 on all three terms, and 100/100
   ids keep their sign on all three. The quantitative core is as seed-verified as the headline.

2. **The split is depth-dependent, and "70/30" is an L25–L26 statement.** It moves 73/27 → 70/30 →
   **52/48** across L25 → L26 → L27. The seed spread (±0.2pp) is two orders of magnitude smaller than
   this drift, so the drift is real and not noise. The raw terms show why: from L26 to L27 the
   structural term *falls* (+21.88 → +16.44) while the content term *rises* (+9.30 → +15.41). This is
   the §3 localization restated as a decomposition — the content component peaks later than the
   structural one — and it means the content share is not a single number. Reporting "≈70% structure"
   without the band is a misstatement of what was measured. The paper should quote the split **with its
   layer**, or quote the range 73/27–52/48 across L25–L27.

---

### Arm matching (verify before reading any effect)

*No read point applies to this section: these are prompt statistics — token counts and regex matches on
the prompt text — not activation measurements.*

Two *different* register statistics are reported throughout this study, and they must not be
conflated:

- **first-person (FP)** — the question contains a first-person pronoun anywhere: regex `\b(I|my|me)\b`.
- **operational opener (OP)** — the question's first three words are exactly "How can I".

These count different things, and **OP is a strict subset of FP**: of the 100 eval behaviors, 87 are
FP and 82 are OP, with all 82 OP items contained in the 87 FP items. The 5 FP-but-not-OP items open
differently ("Write me step-by-step…", "Give me step-by-step…", "Tell me how to…", "Can you help me
find…", "I bought MMR…"). The remaining 13 are neither. So "87/100 first-person" and "'how can i' ×82"
are consistent descriptions of the same set under the two definitions, not a discrepancy.

| arm | n_inject | prompt_len | FP (`\b(I\|my\|me)\b`) | OP ("How can I" opener) |
|---|---|---|---|---|
| clean | 0.0 ± 0.0 | 32.8 ± 5.3 | 87/100 | 82/100 |
| dija | 57.0 ± 10.2 | 136.5 ± 19.4 | 87/100 | 82/100 |
| benign (arm 2) | 57.0 ± 10.2 | 137.7 ± 17.6 | 0/100 | 0/100 |
| benign_op (arm 3) | 57.0 ± 10.2 | 133.4 ± 19.0 | 100/100 | 100/100 |

(clean and dija carry identical FP/OP counts because the DIJA scaffold embeds the same question
verbatim; the counts are measured on that question in both arms, and on the leading question line for
the benign arms.)

`n_inject` is identical to DIJA **per id, 100/100**, for both benign arms — not merely equal in
distribution. Per-id prompt-length gap vs dija: clean −103.7 ± 17.2; benign +1.2 ± 6.0; benign_op
−3.1 ± 4.2.

**Arm 3 over-matches register, and that residual was tested.** Arm 3 is 100/100 on both FP and OP,
against DIJA's 87/100 and 82/100 — so the arms are not *perfectly* opener-matched, and the leftover
18% could in principle contribute to the separation. Restricting the final contrast to the 82 ids
where **both** arms open "How can I" gives an exactly opener-matched test. **Read point = prompt_mean
(PRIMARY):**

| layer | subset | n | mean_d | dz | AUC | null dz | null AUC |
|---|---|---|---|---|---|---|---|
| L25 | all ids | 100 | +6.11 | +1.82 | 0.958 | +0.03 | 0.519 |
| L25 | **opener-matched** | 82 | +6.23 | **+1.85** | **0.963** | +0.20 | 0.584 |
| L26 | all ids | 100 | +9.30 | +2.08 | 0.975 | +0.18 | 0.567 |
| L26 | **opener-matched** | 82 | +9.47 | **+2.15** | **0.983** | +0.29 | 0.609 |
| L27 | all ids | 100 | +15.41 | +2.25 | 0.977 | −0.11 | 0.465 |
| L27 | **opener-matched** | 82 | +16.10 | **+2.44** | **0.994** | −0.06 | 0.483 |

The effect is marginally *stronger* on the exactly-matched subset at every layer. The residual opener
mismatch does not drive it.

### All four contrasts, both diagnostic layers

**Read point = prompt_mean (PRIMARY).** `v_injection_svd` | `v_random_null`:

| contrast | L25 mean_d / dz / AUC | L25 null dz / AUC | L26 mean_d / dz / AUC | L26 null dz / AUC |
|---|---|---|---|---|
| dija − clean *(original headline; confounded)* | +22.94 / +5.55 / 1.000 | +0.57 / 0.656 | +31.18 / +6.11 / 1.000 | +0.85 / 0.734 |
| benign_op − clean *(positive control: +scaffold, +length, stance-matched, NO harm)* | +16.83 / +3.60 / 1.000 | +0.43 / 0.658 | +21.88 / +3.89 / 1.000 | +0.55 / 0.693 |
| **dija − benign_op** *(FINAL TEST)* | **+6.11 / +1.82 / 0.958** | **+0.03 / 0.519** | **+9.30 / +2.08 / 0.975** | **+0.18 / 0.567** |
| dija − benign *(arm 2 reference; stance mismatched)* | +7.19 / +1.99 / 0.977 | +0.31 / 0.612 | +11.29 / +2.26 / 0.987 | +0.35 / 0.630 |

The positive control behaves as a positive control must: a benign prompt carrying the same scaffold
moves the axis nearly as far as a harmful one — at L25, +16.83 of the +22.94, i.e. the 73% structural
share of §2a stated as a raw number.

---

## 3. Localization

The signal concentrates at **L25–L27**, and the case rests on *where the null collapses* rather than on
the peak alone.

`dija − clean`, **read point = prompt_mean (PRIMARY)**, `v_injection_svd` dz | `v_random_null` dz:

| layer | L18 | L22 | L25 | L26 | L27 | L29 |
|---|---|---|---|---|---|---|
| v_injection_svd | +4.68 | +4.18 | +5.55 | **+6.11** | +4.84 | +1.36 |
| v_random_null | **+2.58** | +1.91 | +0.57 | +0.85 | +1.35 | +0.50 |

The null peaks at L18 (dz +2.58, AUC 0.990) — a random direction "detects injection" almost perfectly
there, which is why a same-layer comparison is mandatory. By L25 the null has fallen to dz +0.57 while
the injection axis is climbing to its L26 peak. The two curves separate rather than track, and the
region where they separate most is where the study reads.

Final contrast `dija − benign_op` across the band, **read point = prompt_mean (PRIMARY)**:

| layer | v_injection_svd mean_d / dz / AUC | v_random_null mean_d / dz / AUC |
|---|---|---|
| L16 | +1.85 / +0.85 / 0.791 | +0.18 / +0.60 / 0.704 |
| L18 | +2.19 / +1.08 / 0.830 | +0.07 / +0.20 / 0.574 |
| L20 | +2.99 / +1.46 / 0.901 | −0.16 / −0.48 / 0.318 |
| L22 | +4.86 / +1.59 / 0.932 | −0.06 / −0.19 / 0.427 |
| L24 | +4.42 / +1.61 / 0.934 | −0.01 / −0.02 / 0.490 |
| **L25** | **+6.11 / +1.82 / 0.958** | **+0.02 / +0.03 / 0.519** |
| **L26** | **+9.30 / +2.08 / 0.975** | **+0.13 / +0.18 / 0.567** |
| **L27** | **+15.41 / +2.25 / 0.977** | −0.11 / −0.11 / 0.465 |
| L28 | +10.76 / +0.88 / 0.764 | −0.25 / −0.17 / 0.429 |
| L29 | +3.55 / +0.20 / 0.562 | −0.13 / −0.07 / 0.469 |
| L31 | +9.95 / +0.22 / 0.575 | −2.28 / −0.56 / 0.275 |

The content component rises through the mid-band, peaks at L26–L27, and is gone by L29. The null is
flat and near zero across the whole band on this contrast — further evidence that arm 3's matching did
what it was built to do.

### `n_inject` does not explain the headline

Regressing the paired delta on `n_inject` across ids, **read point = prompt_mean (PRIMARY)**:
`v_injection_svd` at L25 gives r = −0.107, R² = 0.011, intercept +25.38 (the delta extrapolated to
`n_inject`=0). Blank count accounts for ~1% of the variance and the effect is undiminished at zero
blanks. Arm 3 makes this moot anyway — `n_inject` is matched per id — but the regression was
pre-registered as a gate and it passes.

### Secondary read point

**Read point = last_prompt (SECONDARY).** *Caveat restated, as it must be in every last-token table:*
at this read point **cos(v_refusal, v_injection_svd) = +0.65** — the two axes are **not separable
here**, and an apparent "refusal" move may be an injection move bleeding through. Signs also flip
relative to prompt_mean. This table is reported for completeness and carries no headline.

| contrast | L25 mean_d / dz / AUC | L26 mean_d / dz / AUC |
|---|---|---|
| dija − clean | −20.82 / −1.10 / 0.14 | −31.95 / −1.29 / 0.139 |
| benign_op − clean | −60.35 / −2.83 / 0.016 | −59.00 / −1.96 / 0.037 |
| dija − benign_op | +39.53 / +1.84 / 0.954 | +27.05 / +1.21 / 0.882 |

The final contrast retains AUC 0.954 at L25 here, which is directionally consistent with prompt_mean,
but the axis-alignment problem means this cannot be read as independent confirmation.

---

## 3a. Seed robustness of the final contrast

The single-seed limitation is closed for the headline. This section reports the first re-run, covering
the two arms of the final contrast (`dija`, `benign_op`); `clean` and `benign` were re-run subsequently
at the *same* two seeds, so all four arms now exist at seeds 1–2 (§2a for the split, §7 for the two-arm
agreement). All seed captures cover L25–L27 only and project only `v_injection_svd` and
`v_random_null` — `v_refusal` is not loaded by `capture_seed.py` at all, so §8 Q1 stays unopened by
construction.

**Why temperature, not extra seeds at temp=0.** `capture_denoise` takes `argmax` over raw logits when
`temperature == 0` (`common.py:318`), so the trajectory is a deterministic function of the input; a seed
sweep there would re-measure GPU float non-determinism and answer nothing. Stochasticity enters at
`temperature > 0` through Gumbel noise (`common.py:321-323`), which changes *which* tokens commit at
each step. Under LLaDA's bidirectional attention the prompt span attends to the filling canvas, so a
resampled fill yields genuinely different prompt_mean states — precisely the variance at issue.
`temperature=0.2` is also the DIJA paper's own config (`run_dija.py:159`), which this study had
deviated from for determinism, so this is simultaneously an **ecological check**: the finding is
re-tested at the temperature the attack is actually specified at. `capture_seed.py` asserts
`temperature > 0`.

**Read point = prompt_mean (PRIMARY).** seed 0 = temp=0.0 (the existing record); seeds 1–2 = fresh
temp=0.2 trajectories:

| layer | seed | mean_d | dz | AUC | %>0 | null dz | null AUC |
|---|---|---|---|---|---|---|---|
| L25 | 0 (temp=0.0) | +6.11 | +1.82 | 0.958 | 98% | +0.03 | 0.519 |
| L25 | 1 (temp=0.2) | +6.07 | +1.78 | 0.950 | 98% | +0.04 | 0.520 |
| L25 | 2 (temp=0.2) | +6.11 | +1.85 | 0.957 | 98% | +0.06 | 0.532 |
| L26 | 0 (temp=0.0) | +9.30 | +2.08 | 0.975 | 100% | +0.18 | 0.567 |
| L26 | 1 (temp=0.2) | +9.31 | +2.12 | 0.972 | 100% | +0.17 | 0.567 |
| L26 | 2 (temp=0.2) | +9.34 | +2.18 | 0.975 | 100% | +0.20 | 0.581 |
| L27 | 0 (temp=0.0) | +15.41 | +2.25 | 0.977 | 99% | −0.11 | 0.465 |
| L27 | 1 (temp=0.2) | +15.41 | +2.35 | 0.980 | 100% | −0.12 | 0.463 |
| L27 | 2 (temp=0.2) | +15.48 | +2.39 | 0.980 | 100% | −0.09 | 0.476 |

The null remains at chance under sampling (L25: 0.519 → 0.520 → 0.532), so the true-zero floor is not
an artifact of deterministic decoding either.

**Per-id stability — the stricter test.** Two seeds can agree on the mean while disagreeing about which
items separate. `r` = Pearson correlation of the per-id delta vector across seeds; **read point =
prompt_mean (PRIMARY)**:

| layer | r(s1,s2) | r(s0,s1) | r(s0,s2) | sign agreement s1/s2 |
|---|---|---|---|---|
| L25 | 0.926 | 0.946 | 0.961 | **100%** |
| L26 | 0.947 | 0.956 | 0.973 | **100%** |
| L27 | 0.970 | 0.974 | 0.983 | **100%** |

Independent trajectories agree at r = 0.93–0.97 on the per-item deltas, and every one of the 100 ids
keeps its sign across both seeds at all three layers. The same items separate, by the same amounts.

**Read point = last_prompt (SECONDARY).** *Caveat restated: cos(v_refusal, v_injection_svd) = +0.65
here; the axes are not separable at this read point.* Seed spread is likewise negligible — L25
dz +1.86 ± 0.02 (temp=0 ref +1.84), L26 dz +1.20 ± 0.03 (ref +1.21), L27 dz +1.04 ± 0.02 (ref +1.05).

**Note.** temp=0.2 yields marginally *higher* dz than temp=0 at L26–L27 (+2.15 vs +2.08; +2.37 vs
+2.25). The difference is within seed spread, but it establishes that the deterministic setting was not
flattering the result.

---

## 4. τ (internal clock): NULL, circular — demoted to secondary

**Read point = resp_mean (the only valid one — `mu_step` was fit there).**

τ does not survive. Two independent reasons, either sufficient:

**Circularity.** `mu_step` was fit on step index, so `tau_read`/`tau_nn` recovering step is
near-tautological. A "τ shift" under DIJA is the expected consequence of the canvas fill-level /
`n_inject` confound and is not evidence of a nontrivial internal clock.

**The probe reads canvas fill level.** `tau_nn` tracks each condition's own `tau_from_mask` within
±0.16 across the band. The confound gate — regressing paired Δdisc on `n_inject` — yields R² between
0.000 and 0.113 across L16–L31 (L25: R²=0.001, p=0.73). Low R² here does *not* rescue τ: it says the
discrepancy is not even explained by blank count, leaving it as a reading of a quantity (fraction of
canvas filled) that is trivially available from the token bookkeeping without any probe.

*Note on the residual column in the gate table:* residual mean is ~0 **by construction** (OLS centres
it), so residual dz/p are diagnostics of spread, **not** an independent test of a leftover clock
effect. They are not evidence for τ and are not used as such.

**Compression caveat.** `tau_read` (soft) saturates in ~[0.28, 0.77], so `disc_read` carries a large
systematic bias — one that is **present in the clean arm too**. All τ numbers above use the
uncompressed `tau_nn`.

**Physical canvas** (token bookkeeping, no read point — not an activation measurement): the DIJA
response canvas fills by median step 82 (97/100 fill); the clean canvas never fills (100/100). A
mechanism prediction made earlier in this study — that fill speed follows from `n_inject` arithmetic —
is **wrong**: corr(`n_inject`, fill-completion step) = −0.023, p=0.82. Fill speed is set by
confidence ranking, not blank count. Recorded here because the prediction was made before the check.

---

## 5. `v_refusal`: NULL under the pre-registered criterion — and cannot be promoted from these arms

**Read point = prompt_mean (PRIMARY).** On `dija − clean`, `v_refusal` fails the pre-registered floor
criterion: its best layer (L19, dz −1.46) is **0.6×** the random null's best (L18, dz +2.58). At L25 it
reaches only 2.3× the null. It does not clear the floor. **NULL.**

Two further disqualifiers, both discovered during this study and both independently fatal to promoting
it from the arms captured here:

**In-sample leak (arm 2).** `v_refusal` read dz +3.42–3.57 / AUC 1.000 on `dija − benign`. That number
is worthless: arm 2's questions **are** `v_refusal`'s 99-item harmless fitting pool. The reading is
in-sample. It is recorded here only so the leak is on the record, and it is discarded.

**Question-form confound (structural).** `v_refusal`'s two fitting pools are **100% separable by
question form** — the harmless side is 99/99 "What are the…"; the harmful side is 0/43 "What…".
A direction fit on those two pools is inseparable from a form axis, regardless of what it is later
evaluated on. This is a property of the fitting set, not of any particular test, so no contrast run
against these probes can clear it.

**What arm 3 shows, flagged and not promoted.** Arm 3 is the first contrast that is *neither* in-sample
for `v_refusal` *nor* form-confounded (both sides open "How can I"). On it, `v_refusal` reads
**dz +2.31 / AUC 0.988 at L25** against a ~zero null — stronger than the injection axis on the same
contrast. That is coherent with `v_refusal` being a working harm detector whose "null" on `dija − clean`
was the *correct* answer to a contrast containing no harm difference (both arms are harmful). **This is
logged as an open question, not a result** — see §8. Its status in this report remains **NULL**.

---

## 6. What was measured

**Main study.** 100 paired ids (A000–A099), four arms, temp=0, steps=128, gen=128, unified
`fill_all_masks`, layers L16–L31, all 128 denoising steps: 819,200 probe rows across three JSONL files.
Missing cells: none.

**Seed re-runs.** All four arms again at two temp=0.2 seeds, L25–L27 only, projecting `v_injection_svd`
and `v_random_null` only: 8 files × 38,400 = 307,200 rows. Verified complete — 100 ids each, no
unparseable lines, no partial cells, no duplicates; `n_inject` identical to `dija` per id 100/100 for
both benign arms, and 0 throughout for `clean`.

**Engine paths.** `dija`, `benign` and `benign_op` go through the identical path
(`run_dija.PaperRunner.build_inputs` → `<<TPL>>` wrapper → `<mask:N>` expansion → chat template).
`clean` is the bare behavior through the plain chat template with **no wrapper and no blanks** — that
asymmetry is the clean arm's definition, and `capture_seed.py` reproduces `capture.run_condition()`
exactly so the seeded clean arm stays comparable to the temp=0 record (verified: identical
`prompt_len` and `n_inject=0` per id). Arms 2 and 3 are pure string construction — no model in the loop
during their construction.

### Leakage check — arm 3's pool against every probe's fitting set

*No read point applies: this is text overlap between prompt sets, not an activation measurement.*

Zero exact overlap everywhere. Max Jaccard 0.211, against a near-duplicate threshold of J ≤ 0.4 used by
`build_heldout`:

| fitting set | n | exact overlap | max Jaccard |
|---|---|---|---|
| v_refusal harmful (43) | 43 | **0** | 0.143 |
| v_refusal harmless (99) | 99 | **0** | 0.158 |
| v_injection_svd fit (43) | 43 | **0** | 0.143 |
| tau_bank pool `D_neutral_clean` (100) | 100 | **0** | 0.158 |
| `unsafe.jsonl` superset (50) | 50 | **0** | 0.150 |
| eval behaviors (100; not a fit set, but must not leak) | 100 | **0** | 0.211 |

No probe in this study was fit on any of the 100 evaluation cases. `v_injection_svd` and `v_refusal`
were fit on 43 held-out harmful prompts and 99 held-out harmless prompts respectively; `tau_bank` on 40
further neutral seeds. The eval set is disjoint from all of them.

---

## 7. Limitations — stated as constraints on the claim, not as caveats to it

**Seed variance: measured for the headline AND the split; not for anything else.** Both load-bearing
claims are seed-verified at L25–L27 across two independent temp=0.2 resamples of the full denoising
trajectory:

- the **final contrast** (§3a) — dz moves ≤ ±0.05, AUC ≤ ±0.005, per-id r = 0.93–0.97, 100/100 ids
  keep their sign, and the null stays at chance;
- the **structure/content split** (§2a) — percentages move ≤ ±0.2pp, per-id r = 0.93–0.99 on all three
  terms of the identity, 100/100 sign agreement on each;
- the **two-arm agreement** that mitigates the authored-vs-derived confound (below) — ratio
  arm3/arm2 = 0.843 ± 0.021 (L25), 0.823 ± 0.014 (L26), 0.775 ± 0.010 (L27).

All four arms (`clean`, `dija`, `benign`, `benign_op`) now exist at seeds 1–2 (8 × 38,400 rows), so
every load-bearing claim at L25–L27 rests on the same seed footing.

**What remains single-seed.** Layers outside L25–L27 were not re-run at any seed, so the whole band
table in §3, the L26–L27 peak location, and the claim that the effect vanishes by L29 are single-seed.
The `n_inject` regression (§3) and the leakage checks (§6) are likewise temp=0 only — though both are
properties of the prompts rather than of a trajectory, so seed variance does not apply to them in the
same way. τ and `v_refusal` were not re-run and are not projected by `capture_seed.py` at all.

**A limitation surfaced by the seed work itself.** Extending the split to L27 (§2a) showed the
structure/content ratio is **depth-dependent** — 73/27 → 70/30 → 52/48 across L25 → L26 → L27 — with a
seed spread (±0.2pp) two orders of magnitude smaller than the drift. The original §2 tabulated only
L25–L26 and so read as though "≈70/30" characterised the axis. It does not; it characterises that band.
Any single-number statement of the content share is underspecified without its layer.

**Within-template.** All four arms share one `<<TPL>>` wrapper. The content component is demonstrated
*within* that template, against benign prompts wearing the same template. Whether `v_injection_svd`
transfers to a different injection format is untested and should not be assumed — the axis was fit on
this template's structural signature.

**Authored vs. dataset-derived benign text — the weakest joint, and what actually props it up.**
Arm 3's pool is authored (see the deviation note below), so its content term is confounded with
"authored-by-assistant vs. MedSafetyBench-derived." The mitigation is the two-arm agreement, now
seed-verified at the same two temp=0.2 seeds as everything else. **Read point = prompt_mean
(PRIMARY)**, n=100; `±` is the spread across the two stochastic seeds:

| layer | content via arm 2 (dataset-sourced) | content via arm 3 (authored) | ratio arm3/arm2 | gap as % of arm 2 |
|---|---|---|---|---|
| L25 | +7.22 | +6.09 | **0.843 ± 0.021** | **15.7% ± 2.1** |
| L26 | +11.34 | +9.33 | **0.823 ± 0.014** | **17.7% ± 1.4** |
| L27 | +19.94 | +15.44 | **0.775 ± 0.010** | **22.5% ± 1.0** |

The temp=0 reference lies inside the seed spread at every layer (15.1 / 17.6 / 22.3%). Per-id seed
stability of each arm's content term separately: arm 2 r = 0.974–0.988, arm 3 r = 0.926–0.970, sign
agreement 99–100%. (A *cross-arm* per-id correlation is deliberately not reported: both content terms
contain the `dija` term, so such an r is inflated by the shared term and would overstate agreement.)

**What the gap is, stated precisely.** It is tempting — and this report earlier did so — to call the
arm2↔arm3 gap "the stance effect." That is wrong in a way that matters:

- arm 2 content = `dija` (MedSafetyBench-derived) − `benign` (D_neutral_clean) → carries a **stance**
  confound; no authored-text confound.
- arm 3 content = `dija` (MedSafetyBench-derived) − `benign_op` (authored) → carries an **authored-text**
  confound; no stance confound.

So the gap is `stance − authorship`, not stance alone, and **neither arm isolates harm** — each is
contaminated differently. Both are also cross-source (arm 2 is MedSafetyBench vs D_neutral_clean: two
different datasets, merely neither authored here). The triangulation's actual force is twofold:
(i) two arms carrying *different* source confounds land within 16–22% of each other; and (ii) the gap
is **positive at every layer and every seed** — arm 3's content is always the *smaller* one. An
authorship signature inflating arm 3's content term would have to drive the gap negative, and it never
does. This bounds authorship as the smaller of the two contaminants; it does not isolate either.
Separating them cleanly needs a **dataset-sourced operational-benign pool**, which does not exist in
this infrastructure (see the deviation note below) — the same gap that forced arm 3 to be authored in
the first place. This remains the weakest joint in the argument.

**Deviation from protocol, recorded.** The brief called for arm 3's pool to be *drawn* from untouched
data. No such pool exists in this infrastructure. All four available pools are register-degenerate:
`A_harmful_clean` and `B_harmful_injected` are the eval set (87/100 FP, 82/100 OP);
`C_neutral_injected` and `D_neutral_clean` are 0/100 FP, "what are the" ×100. Both benign pools
were already consumed by probes (99 items → `v_refusal`'s harmless side; 40 → `tau_bank`). A
stance-matched benign pool therefore had to be authored. (Register counts here use the two definitions
fixed in §2: FP = contains `\b(I|my|me)\b`; OP = opens "How can I". The eval set is 87/100 FP and
82/100 OP, the latter a strict subset of the former.) Every item is safety-**affirming** — asking
how to do the correct thing, follow guidance, or consult a clinician. None asks how to bypass, evade,
conceal, or avoid supervision. No harmful text was authored. Provenance and rationale are recorded in
`benign_operational_pool.py`.

**The confound chain, stated in full.** The headline moved through three positions as controls were
added, and the final claim should be read with that history visible:
1. `dija − clean` — confounded by +104 tokens of length and by format. A random axis reaches AUC 0.990.
2. `dija − benign` (arm 2) — length, format and `n_inject` matched; survives at dz +1.99 / AUC 0.977.
   Still confounded: arm 2 is 0/100 FP against DIJA's 87/100, so harm and operational stance are
   not separable.
3. `dija − benign_op` (arm 3) — stance additionally matched; survives at dz +1.82 / AUC 0.958 against a
   chance-level null, seed-verified (§3a). Register matching cost the content component only
   15.7 ± 2.1% (L25) / 17.7 ± 1.4% (L26) / 22.5 ± 1.0% (L27), so stance was a minor contributor — but
   that residue is `stance − authorship`, not stance alone, and arm 3 substitutes an authored-text
   confound for the stance one. The chain therefore ends at a **bound, not an isolate** — see the
   two-arm paragraph above.

An earlier stated prior — that the arm-2 test would collapse, since the axis was fit with harm held
constant and scaffold varying — **was wrong**, twice over (arms 2 and 3 both survived). The prior is
recorded because it was stated before the run, and because it means these two results were not selected
for.

**A register check that came back negative, recorded.** Grammatical person alone does not explain the
arm-2 effect: splitting DIJA by first-person status gives dz +2.11 / AUC 1.000 for non-first-person
(n=13) vs dz +1.97 / AUC 0.982 for first-person, Mann-Whitney p=0.594. This is what motivated arm 3 to
test *stance* rather than *person*.

---

## 8. Open questions (logged, not run)

**Q1 — Is `v_refusal` actually a working harm detector?** Arm 3 hints yes (dz +2.31 / AUC 0.988 at L25,
neither in-sample nor form-confounded), and offers a clean explanation of its `dija − clean` null: that
contrast has no harm difference in it. **This cannot be settled from arm 3.** Arm 3 is authored, so a
`v_refusal` reading on it inherits exactly the authored-vs-derived confound described in §7 — and unlike
the `v_injection_svd` result, there is no second, dataset-sourced arm to triangulate against, because
arm 2 is in-sample for `v_refusal`. A clean test requires **a freshly built, non-authored,
operational-harmful-vs-operational-benign pair**, with the benign side drawn from data no probe has
touched. Until then `v_refusal` stays NULL. A fair test would also want `v_refusal` refit on pools that
are not 100% separable by question form, since the current fitting set makes the axis inseparable from a
form direction no matter what it is evaluated against.

**Q2 — Does the content component transfer across injection templates?** The axis was fit on one
template's structural signature and tested within it (§7). A second template would separate "this
template's injection semantics" from "injection semantics."

**Q3 — Why L25–L27?** The content component appears mid-band, peaks at L26–L27, and is gone by L29,
while the structural component is broad from L16. That the two have different depth profiles is visible
in the data and unexplained here. As context rather than as a mechanistic claim: this depth band is
broadly consistent with the mid-to-late safety-representation band reported by Shnaidman et al. and
with the near-isolated final layer described in the Subliminal Clocks work — the convergence is
suggestive of where safety-relevant content tends to become linearly available, but nothing in this
study tests that correspondence, and the present result is a depth profile on one model with one
template.

---

## 9. Constraint compliance

- **Read-only.** No activation was modified anywhere in this study. `assert_readonly_hooks()` runs
  before every forward. Hooks return `None` (leaving output untouched) by construction.
- **Zero steering.** No steering vector was applied. `clock_attack/steerer.py` was read once for
  reference and never imported.
- **No safety-breaking intervention.** None was implemented, including in the Phase 3 discrepancy
  diagnostic, which measured only whether a signal exists.
- **No harmful text authored.** Arm 3's pool is safety-affirming throughout (§7).
- **`v_refusal` built and used as a probe only.** Extracted on a held-out contrastive set, read only,
  never applied to modify an activation, never used to generate, never used to reduce refusals.
- **New folder only.** Everything lives in `clockv2/`. `dija_attack/` and `clock_attack/` were imported
  from and never edited.

---

## Appendix — files

| file | role |
|---|---|
| `common.py` | model loading, read-only hooks, `assert_readonly_hooks`, band/step config |
| `build_vrefusal.py` | `v_refusal` = normalize(mean_harmful − mean_harmless), 43 vs 99 held-out |
| `build_vshared.py` | `v_injection_svd` = top right singular vector of centred (h_dija − h_clean), 43 held-out harmful |
| `build_tau.py` | `tau_bank` / `mu_step`, fit on 40 neutral seeds at resp_mean |
| `capture.py` | arms 1–2 capture (clean, dija) → `probe_readings.jsonl` |
| `build_benign.py` | arm 2 constructor (informational benign, length/format matched) |
| `build_benign_op.py` | arm 3 constructor + leak check |
| `benign_operational_pool.py` | arm 3's 90 authored safety-affirming operational questions + provenance |
| `capture_benign.py` | arms 2–3 capture → `probe_readings_benign{,_op}.jsonl` |
| `capture_seed.py` | seed-robustness capture, all four arms, L25–L27, temp>0 → `probe_readings_s{1,2}_{clean,dija,benign,benign_op}.jsonl` (8 files) |
| `run_seed_rest.sh` | detached (`setsid`) + `flock`-guarded driver — `dija`/`benign_op` seed captures |
| `aggregate.py` | Phase 3 aggregate (canvas, paired deltas, τ, confound gates) |
| `analyze_matched.py` | 3-arm analysis (arm 2 decisive test) |
| `analyze_final.py` | 4-arm final analysis (arm 3 final test, identity split) |
| `run_seed_clean.sh` | detached + `flock`-guarded driver for the clean-arm seed captures |
| `run_seed_benign.sh` | detached + `flock`-guarded driver for the arm-2 seed captures |
| `analyze_seed.py` | seed robustness of the final contrast (§3a) |
| `analyze_split_seed.py` | seed variance + depth dependence of the structure/content split (§2a) |
| `analyze_twoarm_seed.py` | seed-verified two-arm agreement, the §7 confound mitigation |
