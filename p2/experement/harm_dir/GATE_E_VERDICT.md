# GATE E VERDICT — PASS. The DeepSeek harm direction transfers to LLaDA.

Built a VALID cross-author transfer target (which Phase 2' lacked) and tested whether
`v_harm_ds` — the harm/safe direction fitted on DeepSeek-authored pairs — points along
LLaDA's *own* harm axis. It does, well above chance and not via length. This clears the
last blocking gate.

Artifacts: `data/gateE.json`, `data/pairs_gateE.jsonl` (36 matched pairs),
`data/states_gateE.pt`, `logs/gateE.log`. Probe: `probes/v_harm_ds.pt`.

---

## The target (this is what Phase 2' was missing)

Both arms **LLaDA-authored**, real harm/safe **content** contrast, **matched length**:

- harm = LLaDA's DIJA-jailbroken completion (`B_harmful_injected`, ASR=comply) —
  genuinely actionable (e.g. injection doses, evasion steps).
- safe = LLaDA-**generated** substantive refusal, length-matched by diffusion
  `gen_length` (the free bare-behaviour refusal was ~8 words vs ~110 = 12.5× length
  confound; that target is kept only for comparison).

**36 matched pairs** survive the filter (Jaccard ≤ 0.35, length ±25%, non-degenerate
genuine refusal). Jaccard median 0.214, length-ratio median 1.09.

## Why AUC is the wrong metric here, and what we used instead

The LLaDA harm and safe activation **clusters sit far apart** (harm-state norm > safe-state
norm; norm alone separates them up to AUC 0.97 at deep layers). Consequently a **random**
direction separates the pairs at **AUC p95 ≈ 0.92–0.98**. So transfer-AUC = 1.00 is
**saturated and uninformative** — almost everything separates these clusters.

The discriminating question is **directional**: does `v_harm_ds` point the *same way* as
LLaDA's own harm axis `v_llada = mean(LLaDA harm) − mean(LLaDA safe)`? Measured by
`cos(v_harm_ds, v_llada)` against a random-direction null (`|cos| ~ N(0, 1/√4096)`, so
p99 ≈ 0.04).

## Result — matched target (decision), all judged on alignment

| layout | L | **cos_align** | null_p99 | ×null | xferAUC | randAUC_p95 | AUC_norm | cos_vlen | r_len | E |
|---|---|---|---|---|---|---|---|---|---|---|
| bare | 16 | **0.597** | 0.041 | ~14× | 1.000 | 0.919 | 0.012 | 0.300 | 0.19 | PASS |
| bare | 25 | 0.582 | 0.039 | ~15× | 1.000 | 0.921 | 0.272 | 0.351 | 0.22 | PASS |
| bare | 27 | 0.571 | 0.040 | ~14× | 1.000 | 0.912 | 0.502 | 0.329 | 0.21 | PASS |
| dija | 16 | 0.587 | 0.042 | ~14× | 1.000 | 0.940 | 0.035 | 0.258 | 0.18 | PASS |
| dija | 25 | 0.469 | 0.041 | ~11× | 1.000 | 0.981 | 0.899 | 0.259 | 0.17 | PASS |
| dija | 27 | 0.494 | 0.040 | ~12× | 1.000 | 0.959 | 0.971 | 0.226 | 0.14 | PASS |

**cos_align = 0.47–0.60, ~11–15× the random null — PASS at every layer/layout.**
The DeepSeek-built harm direction genuinely shares ~half its cosine with LLaDA's own
harm axis. Not "everything separates": a random direction has cos ≈ 0.04 with `v_llada`.

**Not length.** `r(v_harm projection, length)` = 0.14–0.22 and `cos(v_harm, v_len)` =
0.23–0.35 on the matched target. Contrast the length-confounded natural target, where
`r_len` jumps to **0.61–0.83** — that is what a length-driven result looks like, and the
matched target does not show it.

**Alignment tracks content, not length — corroboration.** `v_harm_ds` aligns *more* with
the matched (content) target's axis (cos 0.47–0.60) than with the natural (length-dominated)
target's axis (cos 0.20–0.28). A length direction would show the opposite. So `v_harm_ds`
is keyed on harm *content*, consistent with its length-controlled construction (Gate B).

Best transfer at **L16** (bare 0.597 / dija 0.587), which is also norm-independent
(AUC_norm 0.01–0.04). The deepest layers (dija L25/L27) separate DeepSeek best in Phase 2'
but transfer to LLaDA slightly worse (cos 0.47–0.49) and are norm-confounded — so **L16 is
the sweet spot for steering**, not the deepest layer.

---

## The full picture across phases

| property | test | result |
|---|---|---|
| separates harm/safe content | Gate A (Phase 2') | PASS, AUC 0.97–0.99 |
| not a length artifact | Gate B (Phase 2') | PASS, cos_len small, r_len ~0 |
| not an authorship artifact | cos_src (Phase 2') | fixed: +0.9 → −0.4, no longer collinear |
| **transfers to LLaDA's harm axis** | **Gate E (this phase)** | **PASS, cos_align 0.5 vs null 0.04, not length** |

## Honest caveats (do not oversell)

1. **Alignment ~0.5 is a *partial* proxy, not identity.** `v_harm_ds` and LLaDA's harm
   axis share ~half their direction; steering along it pushes substantially — but not
   perfectly — along LLaDA's own harm axis.
2. **This validates a DIRECTION, not a causal steering effect.** Gate E shows `v_harm_ds`
   *detects/aligns with* LLaDA harm/safe. It does **not** yet show that adding/subtracting
   it during generation reduces harmful output while preserving fluency. That is the actual
   Phase 3 steering experiment — the real proof.
3. **n = 36 matched pairs.** Enough to put alignment ~15× above the null, but a modest set.

## Recommendation

Gate E is the last blocking gate, and it **passes cleanly**. Steering (Phase 3) is now
**justified**: pilot `v_harm_ds` at **L16** (dija layout matches steering-time input),
measure harmful-output reduction vs fluency on a small held-out batch, and treat that
causal effect — not any probe metric — as the final word.
