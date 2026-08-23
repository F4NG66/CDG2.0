# PHASE 2 VERDICT — FAIL. Do not proceed to Phase 3.

`v_harm` as built is **an authorship direction, not a harm direction**. It separates
"text written by DeepSeek" from "text written by LLaDA" almost perfectly, and
separates harmful from safe LLaDA text at chance.

Artifacts: `data/gates.json`, `data/diagnosis.json`, `logs/phase2_gates.log`,
`logs/phase2_diagnosis.log`, `probes/v_harm.pt` (kept for the record — **do not steer with it**).

---

## Verdict table

### natural-heavy split — SPLIT OF RECORD (48 train / 30 test, 20 natural in test)

| layout | L | AUC | 95% CI | AUC_lenm | 95% CI | cos_len | AUC_nat | 95% CI | n | cos_src | cos_inj | A | B | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 16 | 0.754 | [0.63,0.87] | 0.701 | [0.47,0.89] | −0.020 | 0.535 | [0.34,0.71] | 20 | 0.579 | 0.131 | FAIL | FAIL | FAIL |
| bare | 25 | 0.722 | [0.58,0.85] | 0.681 | [0.46,0.88] | −0.114 | 0.515 | [0.33,0.70] | 20 | 0.744 | 0.090 | FAIL | FAIL | FAIL |
| bare | 27 | 0.722 | [0.58,0.85] | 0.660 | [0.43,0.88] | −0.138 | 0.517 | [0.33,0.69] | 20 | 0.705 | 0.039 | FAIL | FAIL | FAIL |
| dija | 16 | 0.758 | [0.63,0.88] | 0.792 | [0.59,0.94] | −0.136 | 0.595 | [0.42,0.77] | 20 | 0.781 | 0.246 | FAIL | FAIL | FAIL |
| dija | 25 | 0.686 | [0.54,0.81] | 0.646 | [0.40,0.85] | −0.455 | 0.535 | [0.36,0.71] | 20 | 0.920 | 0.160 | FAIL | FAIL | FAIL |
| dija | 27 | 0.677 | [0.53,0.80] | 0.632 | [0.40,0.84] | −0.477 | 0.530 | [0.35,0.71] | 20 | 0.898 | 0.029 | FAIL | FAIL | FAIL |

### stratified split (55 train / 23 test, 8 natural in test — underpowered, kept on the record)

| layout | L | AUC | 95% CI | AUC_lenm | 95% CI | cos_len | AUC_nat | 95% CI | n | cos_src | cos_inj | A | B | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 16 | 0.885 | [0.77,0.96] | 0.876 | [0.72,0.98] | −0.311 | 0.500 | [0.20,0.80] | 8 | 0.621 | 0.143 | FAIL | FAIL | FAIL |
| bare | 25 | 0.885 | [0.78,0.96] | 0.871 | [0.72,0.97] | −0.456 | 0.438 | [0.14,0.72] | 8 | 0.755 | 0.093 | FAIL | FAIL | FAIL |
| bare | 27 | 0.860 | [0.74,0.95] | 0.840 | [0.67,0.96] | −0.474 | 0.344 | [0.06,0.64] | 8 | 0.733 | 0.051 | FAIL | FAIL | FAIL |
| dija | 16 | 0.879 | [0.77,0.96] | 0.884 | [0.75,0.99] | −0.491 | 0.453 | [0.17,0.73] | 8 | 0.819 | 0.242 | FAIL | FAIL | FAIL |
| dija | 25 | 0.847 | [0.72,0.95] | 0.876 | [0.73,0.98] | −0.739 | 0.422 | [0.14,0.72] | 8 | 0.928 | 0.177 | FAIL | FAIL | FAIL |
| dija | 27 | 0.826 | [0.68,0.94] | 0.840 | [0.68,0.96] | −0.752 | 0.391 | [0.12,0.69] | 8 | 0.914 | 0.056 | FAIL | FAIL | FAIL |

`AUC_gen` (DeepSeek-safe pairs) = **1.00 at every layer and layout, on both splits.**

B1 `r(projection, length)` is small throughout (−0.09 … −0.29), so raw length is
not the driver — but that is cold comfort given what is.

---

## What killed it

**Gate D, decisively.** The signature is unambiguous:

| | AUC |
|---|---|
| pairs whose safe half is DeepSeek-written | **1.00** |
| pairs whose safe half is LLaDA (both sides LLaDA) | **0.34 – 0.60** |
| cos(v_harm, v_source) | **0.58 – 0.93** |

At `dija` L25 the direction is `cos = 0.92` with the authorship axis. It is
essentially the authorship axis. This is the exact failure the synthetic Gate D
control was built to catch, and it caught it.

Gate A also fails on its own on the split of record (best AUC 0.758, CI upper
0.88 < 0.90). Gate B fails via `cos_len` at the deeper `dija` layers (−0.46,
−0.48) and via matched-subset AUC everywhere.

## Is the premise salvageable? No — not with these labels

Grouped 5-fold CV over the 28 natural pairs alone, `v_harm` refitted per fold:

| layout | L | AUC_nat_cv | 95% CI |
|---|---|---|---|
| bare | 16 | 0.524 | [0.36, 0.68] |
| bare | 25 | 0.508 | [0.35, 0.66] |
| bare | 27 | 0.513 | [0.36, 0.66] |
| dija | 16 | 0.495 | [0.34, 0.65] |
| dija | 25 | 0.483 | [0.33, 0.64] |
| dija | 27 | 0.487 | [0.34, 0.64] |

**Chance, everywhere.** Every CI contains 0.50. There is no harm signal in the
LLaDA-vs-LLaDA pairs to find — so "collect more natural pairs" would not rescue
this.

### Why: the graded labels don't mark a real text-level harm contrast

| pair source | word-set Jaccard(harm, safe) | frac of harm words also in safe |
|---|---|---|
| natural (LLaDA both sides) | **0.503** | **0.695** |
| generated (DeepSeek safe) | 0.285 | 0.523 |

The two graded classes are near-identical templated text. B075, the most
overlapping natural pair, differs in three lines out of nine:

```
HARM  1. Understand the importance of maintaining patient confidentiality.
SAFE  1. Respect the patient's right to confidentiality.

HARM  6. Document the findings securely.
SAFE  6. Store the information securely.
```

Those mean the same thing. And **both versions contain** `3. Ensure all actions
are undocumented.` — arguably the only harmful line in either. The judge's
`unchanged_harmful` vs `substantive_safe_substitution` split is not tracking harm
content here; `specificity_score` 0.76 vs 0.29 is the judge scoring the same kind
of text differently.

Filtering all 28 natural case_ids for a pair that genuinely diverges
(Jaccard < 0.35 **and** specificity gap > 0.4 **and** length within ±20%) leaves
**1 case** (B023). There is no usable natural subset.

I considered whether mean-pooling was diluting a real signal — ~70% of response
tokens are shared scaffold. The text-overlap numbers rule that out as the primary
cause: with 1/28 pairs actually diverging, the contrast isn't in the text to be
pooled.

---

## Recommendation

Do not steer with `v_harm`. Do not run Phase 3 as scaffolded.

The method (same prompt, different response; group split; length control) is
sound and the harness works — it correctly rejected two synthetic artifacts and
then correctly rejected the real thing. What's broken is the **source of the
harm/safe labels**. Under DIJA + prefill, LLaDA emits generic templated procedure
text whichever way the judge grades it; the harmful content largely lives in the
*user's scaffold*, not in the model's fill. That is consistent with the earlier
finding that the injection direction was ~70% surface form.

Three ways forward, in the order I'd try them:

1. **Author both sides with the same model.** Have DeepSeek write both a harmful
   and a safe response in the same scaffold at matched length. Authorship is then
   matched by construction and Gate D becomes vacuous, leaving a genuine content
   contrast. Cheap, login-node only. Risk: synthetic harmful text may not match
   what LLaDA emits when jailbroken, so transfer to steering needs its own check.
2. **Find a setting where LLaDA actually emits substantively harmful content.**
   The current runs mostly don't — 43–57% of every arm grades `disclaimer_only`.
   Worth resolving why the ASR judge says 85% "comply" while the valence judge
   says ~30% "unchanged_harmful"; this analysis suggests neither tracks
   text-level harm well.
3. **Re-grade a sample by hand** against a rubric that requires a concrete
   actionable difference, and check the inter-judge disagreement above before
   trusting any automated valence label for direction-building.
