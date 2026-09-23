# Data lineage

**No dataset rows are published in this repository.** This page records the role, source pools, counts and hashes of each population only. Counts come from the frozen population freezes, which were read for source-pool fields only.

## Canonical files (private; shared separately with authorized collaborators)

| File | SHA-256 | Rows | Role |
|---|---|---:|---|
| `ABCD_clean_v2_DIJA_Qwen_A_B_C_D.jsonl` | `4ab0b3b5eae974575e882284f86affceca73b152b904d7110a3fab66b4ae8256` | 2,000 | **Older canonical lineage.** Declared canonical for V2 BUILD/DEV and for generalized-safety V2 direction construction (400 groups bound by `DIJA_LINEAGE_BINDING`). No public verifier yet. |
| `ABCD_clean_v2_2_OFFICIAL_DIJA_FINAL.jsonl` | `4392afca6d8f6b788a848d0417c92807ce01aeb077a67d6424bf12d69138eaaf` | 2,000 | **v2.2 OFFICIAL canonical.** RRAE pipeline dataset and source of the M1 held-out population. Verifier and provenance: [`data/canonical_abcd_v2_2/`](../data/canonical_abcd_v2_2/). |

Both files contain 500 A/B/C/D groups: A = harmful clean, B = harmful + DIJA, C = benign + DIJA, D = benign clean.

The V2 ReNeLLM request-lineage mismatch (see [`LIMITATIONS_AND_LINEAGE.md`](LIMITATIONS_AND_LINEAGE.md)) arose because BUILD/DEV ReNeLLM construction read per-pair text from the v2.2 OFFICIAL file instead of the declared Qwen-lineage file.

## Populations by stage

| Population | Groups | Harmful (A-side) source | Benign (D-side) source | Published? |
|---|---:|---|---|---|
| V2 BUILD / DEV (directions, β, M3/M3′ gate BUILD, M3 DEV) | 320 / 80 | Qwen-lineage canonical (`4ab0b3b5…`) | same | No |
| M1 held-out | 100 (80 health / 20 non-health) | v2.2 OFFICIAL canonical (`4392afca…`): its 500 groups minus BUILD 320 minus DEV 80 | same | No |
| M2 fresh confirmatory | 126 | SALAD-Bench | MedQuAD (44), xstest (82) | No |
| M3′ NEW_DEV | 165 | CARES-18K test (132), SALAD-Bench train (33) | MedQuAD (132), OR-Bench hard-1k (33) | No |
| Final confirmatory pool | 576 | CARES-18K test (461), SALAD-Bench train (115) | MedQuAD (461), OR-Bench hard-1k (115) | **No. UNCONSUMED and reserved.** |

Attack construction:
- M1 held-out ReNeLLM: 98 of 100 pairs accepted.
- M3′ NEW_DEV: DIJA 159 of 165 groups constructed; ReNeLLM 152 of 165 accepted.
- Failed or exhausted groups were kept in the record and never replaced.

Zero-overlap audits between NEW_DEV, the final pool, M2 and the canonical files are recorded in the frozen freezes.

## Upstream terms

The canonical A-side combines MedSafetyBench, BeaverTails (CC BY-NC 4.0), HarmBench, AdvBench and Do-Not-Answer (CC BY-NC-SA 4.0); see [`PROVENANCE.md`](../data/canonical_abcd_v2_2/PROVENANCE.md). Redistribution terms for CARES-18K, SALAD-Bench, OR-Bench, MedQuAD and xstest were not verified for this release. That is one more reason every population stays private. Users remain responsible for upstream terms.
