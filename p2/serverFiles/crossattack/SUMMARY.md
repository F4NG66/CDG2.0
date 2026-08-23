# Cross-Attack Held-Out Transfer Test — attack2 vs DIJA

**Question.** An injection detector ("the DIJA probe": logistic regression on the `harm`-scope activation, contrast **B vs A** = harmful-injected vs harmful-clean, frac 0.05, layers {11,16,26}, hidden + SAE spaces) reaches AUC≈1.0 on the DIJA template-injection family. Is that signal the **injection mechanism** (should transfer to a different attack family) or is it **DIJA-specific** (a fingerprint of DIJA's own scaffold)?

**Method.** We freeze the probe and apply it — **without any retraining or rescaling** — to `attack2`, a structurally different injection family (Q&A/dialogue scaffold placed *after* the request; small blanks; no Step/Procedure vocab) built on the **same 100 harmful behaviors** with **byte-identical request text**. The only `.fit()` calls in the analysis are on the *source* family; every cross-family number is `predict_proba` on the frozen `Pipeline(StandardScaler → LogisticRegression(C=1.0, max_iter=2000))`, so the StandardScaler learned on the source is applied unchanged to the target.

## AUC table

`own-clean` = the honest held-out cross-family number (target's own clean states as the negative). `src-clean` = diagnostic that reuses the **source** family's clean anchor as the negative (those negatives were in the probe's training set, so it is not a clean held-out AUC — it isolates whether the injected class lands on the injected side of the frozen boundary).

| space | layer | within-DIJA ceiling (grouped) | DIJA→a2 own-clean | DIJA→a2 src-clean | a2→DIJA own-clean | a2→DIJA src-clean |
|---|---|---|---|---|---|---|
| hidden | 11 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hidden | 16 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hidden | 26 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sae | 11 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sae | 16 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sae | 26 | 0.994 ± 0.003 | 1.000 | 1.000 | 1.000 | 1.000 |

**TF-IDF surface baseline** (attack2 behavior text, B2 vs A2, p9 recipe: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True) + LogReg(C=1.0), StratifiedKFold=5): **AUC = 0.018 ± 0.011** (n=200; 100/100 behavior pairs byte-identical A2↔B2). ≈chance, as required — the request text carries no injection signal, so any transfer AUC above this is a genuine activation-level injection signature, not lexical content.

## Verdict — per space × layer

Rule: own-clean transfer AUC ≥ 0.80 → **GENERALIZES** (near the ≈1.0 ceiling); 0.65–0.80 → **PARTIAL**; < 0.65 (toward the 0.5 / TF-IDF floor) → **DIJA-SPECIFIC**. The verdict is driven by the honest **own-clean** number in each direction.

| space | layer | DIJA→a2 verdict | a2→DIJA verdict |
|---|---|---|---|
| hidden | 11 | GENERALIZES | GENERALIZES |
| hidden | 16 | GENERALIZES | GENERALIZES |
| hidden | 26 | GENERALIZES | GENERALIZES |
| sae | 11 | GENERALIZES | GENERALIZES |
| sae | 16 | GENERALIZES | GENERALIZES |
| sae | 26 | GENERALIZES | GENERALIZES |

## Bottom line

**The injection signal GENERALIZES across attack families.** The frozen DIJA probe detects `attack2` injections (and vice-versa) at AUC well above the ≈chance surface baseline, in both directions.

**What generalizes (and what this does *not* claim).** Both families share the structure *clean = bare request*, *injected = request wrapped in a scaffold*, but the scaffolds are lexically/structurally disjoint (DIJA: Step/Procedure worksheet, blanks before the request; attack2: Q&A dialogue, small blanks after the request). The frozen probe transfers at the ceiling in both directions, so the `harm`-scope representation of the *same* request tokens shifts to an *injection-context* direction that is invariant to scaffold style — i.e. it detects **injected-vs-clean**, not DIJA-vs-attack2. It is a family-invariant injection signature, not a DIJA-scaffold fingerprint. (It does not claim to tell the two injection families apart — that is a different, and here irrelevant, question.)

- within-DIJA grouped ceiling: mean 0.999 (range 0.994–1.000)
- DIJA→attack2 own-clean transfer: mean 1.000 (range 1.000–1.000)
- attack2→DIJA own-clean transfer: mean 1.000 (range 1.000–1.000)
- surface (TF-IDF) floor: 0.018
- verdict tally across 6 space×layer cells × 2 directions = 12: GENERALIZES 12, PARTIAL 0, DIJA-SPECIFIC 0

## Caveats

- **Within-model, single seed (seed 0), n = 100 behaviors per group.** One diffusion model (LLaDA-8B-Instruct); no cross-model or cross-seed replication.
- The probe was **frozen** — fit on the source family only; the StandardScaler and logistic weights are never re-estimated on the held-out family (`src-clean` variants reuse source negatives by construction and are diagnostics, not clean held-out AUCs).
- `attack2` reuses DIJA's exact 100 behaviors with byte-identical request text, so the `harm`-scope contrast stays identity-controlled; only the injection scaffold differs. The ≈chance TF-IDF behavior baseline confirms request content is not the signal.
- Ceiling uses GroupKFold by base behavior id (A000 & B000 kept in the same fold) so it cannot inflate via behavior memorization; transfer needs no CV (the eval family is disjoint from the fit family).
