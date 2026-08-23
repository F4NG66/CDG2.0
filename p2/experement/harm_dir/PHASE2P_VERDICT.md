# PHASE 2' VERDICT — INCONCLUSIVE. Do not steer yet; the blocking gate has no valid target.

Option 1 (DeepSeek writes BOTH sides) did what it was designed to do: it **killed the
authorship confound that destroyed Phase 2** and produced a clean, length-controlled
harm/safe separation. But the **blocking cross-author gate (Gate E) cannot be decided**,
because the only LLaDA-authored transfer target we have — the same 28 near-identical
"natural" pairs from Phase 2 — carries no detectable harm contrast for *any* direction.
So we still cannot green-light steering, but for a different and more precise reason than
before.

Artifacts: `data/gates_ds.json`, `data/pairs_ds.jsonl`, `data/states_ds.pt`,
`probes/v_harm_ds.pt` (kept for the record — **do not steer with it yet**),
`logs/phase2p_gates.log`, `logs/build_pairs_ds.log`.

---

## Verdict table — v_harm built on DeepSeek TRAIN (54 train / 23 test), Gate E leads

| layout | L | AUC | 95% CI | AUC_lenm | 95% CI | cos_len | cos_src | AUC_xfer→LLaDA | 95% CI | A | B | **E** | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 16 | 0.968 | [0.91,1.00] | 0.953 | [0.88,1.00] | 0.262 | −0.380 | 0.596 | [0.44,0.74] | PASS | INCONC | **INCONC** | INCONC |
| bare | 25 | 0.981 | [0.95,1.00] | 0.972 | [0.92,1.00] | 0.166 | −0.417 | 0.557 | [0.40,0.70] | PASS | PASS | **INCONC** | INCONC |
| bare | 27 | 0.974 | [0.93,1.00] | 0.961 | [0.89,1.00] | 0.152 | −0.422 | 0.552 | [0.40,0.70] | PASS | PASS | **INCONC** | INCONC |
| dija | 16 | 0.974 | [0.92,1.00] | 0.961 | [0.89,1.00] | 0.246 | −0.387 | 0.603 | [0.45,0.75] | PASS | PASS | **INCONC** | INCONC |
| dija | 25 | 0.985 | [0.95,1.00] | 0.978 | [0.92,1.00] | 0.121 | −0.330 | 0.491 | [0.33,0.64] | PASS | PASS | **FAIL** | FAIL |
| dija | 27 | 0.987 | [0.95,1.00] | 0.981 | [0.93,1.00] | 0.067 | −0.357 | 0.487 | [0.33,0.64] | PASS | PASS | **FAIL** | FAIL |

Thresholds — A: AUC lo ≥ 0.90 · B: AUC_lenm lo ≥ 0.85 **and** |cos_len| < 0.30 ·
E: transfer-AUC lo > 0.50. All judged on the CI, not the point.

### Gate E diagnostics — transfer vs. oracle

| layout | L | xfer full (n=28) | xfer group-clean (n=8) | **oracle** (best LLaDA-only, grouped CV) |
|---|---|---|---|---|
| bare | 16 | 0.596 [0.44,0.74] | 0.562 [0.25,0.84] | 0.524 [0.36,0.68] |
| bare | 25 | 0.557 [0.40,0.70] | 0.547 [0.22,0.83] | 0.508 [0.35,0.66] |
| bare | 27 | 0.552 [0.40,0.70] | 0.562 [0.23,0.86] | 0.513 [0.36,0.66] |
| dija | 16 | 0.603 [0.45,0.75] | 0.578 [0.27,0.84] | 0.495 [0.34,0.65] |
| dija | 25 | 0.491 [0.33,0.64] | 0.391 [0.12,0.69] | 0.483 [0.33,0.64] |
| dija | 27 | 0.487 [0.33,0.64] | 0.438 [0.14,0.73] | 0.487 [0.34,0.64] |

**The oracle is the whole story.** The oracle is the *best possible* direction fitted on
the LLaDA pairs themselves (grouped 5-fold CV, refit per fold). It tops out at
**0.524, CI [0.36,0.68] — chance.** If the best LLaDA-only direction can't separate LLaDA
harm from LLaDA safe, then **no direction can pass Gate E on these pairs**, ours included.
Gate E is therefore *uninformative*: it neither confirms nor refutes `v_harm_ds`. This is
the exact degeneracy flagged before the run — the 28 "natural" pairs had word-set Jaccard
0.503 and were already chance-separable in Phase 2.

---

## What Option 1 actually fixed

| | Phase 2 (mixed authors) | Phase 2' (DeepSeek both sides) |
|---|---|---|
| harm/safe separation (held-out AUC) | 0.68–0.76, CI upper < 0.90 → **Gate A FAIL** | **0.97–0.99, Gate A PASS everywhere** |
| length-matched AUC | fails everywhere | 0.95–0.98, **Gate B PASS at 5/6** |
| length driver? r(proj,len) | small | small (+0.01…+0.10) — **not length** |
| authorship alignment cos(v_harm, v_source) | **+0.58 … +0.92** (v_harm *was* the author axis) | **−0.33 … −0.42** (no longer collinear with harm) |

The authorship confound is genuinely broken: in Phase 2 the harm axis and the author axis
were the *same* axis (cos up to 0.92, and DeepSeek-safe pairs separated at AUC 1.00 purely
on who wrote them). Now `v_harm` separates real harm/safe **content** at ~0.98 with length
controlled. That was the point of Option 1, and it worked.

## The remaining, honest caveat — v_harm is not author-invariant

`v_harm` is built entirely from DeepSeek text, and LLaDA text projects systematically
toward its harm pole: the **residual authorship AUC** (does `v_harm` sort DeepSeek-safe
from LLaDA-safe?) is **0.09–0.12** — far from the 0.5 we'd want. In plain terms, *all*
LLaDA text looks somewhat "harm-like" to this direction regardless of content. On its own
that isn't disqualifying (some cross-author distribution shift is unavoidable when you fit
on one author and apply to another), but it means the transfer question is genuinely open —
and Gate E, the test that would have settled it, has no valid target to settle it on.

---

## Verdict

**INCONCLUSIVE.** Do not steer with `v_harm_ds` yet, and do not run Phase 3 as scaffolded.

- The direction is real, clean, length-controlled, and no longer an authorship artifact —
  a genuine step forward from Phase 2.
- But its **transfer to LLaDA is unestablished**, because the only LLaDA harm/safe pairs we
  own have no separable harm contrast (oracle = chance). A blocking gate you can't evaluate
  is not a pass.

The bottleneck is now unambiguous and singular: **we have no LLaDA-authored pair set with a
real harm-vs-safe content gap.** Everything else is in hand.

## Ways forward (the decision is yours)

1. **Build a valid Gate E target.** Get LLaDA to emit two genuinely contrasting responses
   to the same request — a jailbroken *harmful* completion vs. a real *refusal* to the bare
   behaviour — teacher-force both, and re-run Gate E against those. This is the correct fix
   for the broken yardstick and is the only thing that can actually decide transfer. Needs
   new generation + one GPU capture; a few hours.
2. **Direct steering pilot as the transfer test.** Skip the proxy gate: steer a small batch
   of LLaDA generations with `v_harm_ds` at L25/L27 and measure whether harmful output drops
   without wrecking fluency. "The proof is in the steering." Faster to signal, but it *is*
   Phase 3, and it means acting on an inconclusive gate — against the gate-first rule.
3. **Stop here.** Bank the finding: same-author construction removes the authorship
   confound and yields a clean harm/safe content direction; validating transfer requires a
   LLaDA pair source we don't yet have.
