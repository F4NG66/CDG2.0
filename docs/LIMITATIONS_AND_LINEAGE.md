# Limitations and lineage

The machine-readable version of this page is [`results/generalized_safety/lineage_status.json`](../results/generalized_safety/lineage_status.json). No direction or β was rebuilt or recalibrated, and no frozen artifact was edited. Documentation-level SHA corrections are in [`PROVENANCE_ERRATA.md`](PROVENANCE_ERRATA.md).

## 1. V2 direction-construction lineage caveat

| Object | Status |
|---|---|
| `V_DIJA`, `β_DIJA` = 74.7751 | **Clean** under the audited V2 lineage. |
| `V_RENELLM` | **Caveated.** Built from the V2 ReNeLLM tree, which is affected by a documented request-lineage mismatch: the harmful-side ReNeLLM attacks were constructed against a different canonical file than the safe side. **157 of 199** construction pairs are therefore cross-topic. |
| `β_RENELLM` = 81.5953 | **Caveated.** Derived from the same affected 199-pair population. Calibrated on only the 42 unaffected pairs, it would have been 77.76 (−4.7%). |
| `V_ALL` | **Caveated (inherited).** Inherits the ReNeLLM-side direction caveat through its equal-family average. |
| `β_ALL` = 73.1175 | **Caveated (inherited).** Inherits the ReNeLLM-side calibration caveat through the weighted-median ranking. |
| Earlier `generalized_safety_v1` direction family | **Independently audited and unaffected**: separate dataset, script and era. |

The M1/M2 values remain measurements of the frozen interventions that were actually executed. However, the intended semantic interpretation of the V2 ReNeLLM-derived direction and of the combined `V_ALL` direction and dose is weakened by the documented construction-lineage mismatch.

The root-cause classification is `TOOLING_FAULT_WRONG_FILE_SOURCING`. A fail-closed scenario-match audit was added for all later construction (`experiments/generalized_safety/m3/scenario_match_audit.py`), and it ran for the M3′ NEW_DEV construction.

## 2. M2 documented tooling deviations

Both deviations were honored, not corrected, and neither changes a rubric or a metric definition.

- **D1 — label vocabulary.** The M2 contract listed a different label vocabulary. Judging and metrics used the M1 held-out label spaces, as the contract item's own "inherited, unchanged" text specifies.
- **D2 — metric-contract reference.** The M2 contract cites the DEV metric-contract SHA (`ee4446d6…`). The M1 held-out metric contract (`8fbaeec2…`) was the actual reference. The definitions are the same.

## 3. Population and judging limitations

- **C_DIJA BUILD fragmentation.** The `ambiguous_or_malformed` rate for BUILD/DEV C_DIJA baselines was 47.5% (BUILD) and 58.8% (DEV), well above M1 held-out, under byte-identical generation settings. Spot checks found on-topic but fragmented output. This thins the benign negative class available to the gate. Recorded, not corrected.
- **Judge-defined outcomes.** All outcomes are DeepSeek labels. On BUILD B, DeepSeek agreed with an independent blinded assistant review on **538/611 (88.05%)** items in the authoritative v2 judging. The **73.49% (449/611)** figure quoted in the M3′ contract comes from the superseded v1 judging, which was invalidated for a ReNeLLM `base_request` mismatch. The assistant labels were never used for fitting.
- **Final-pool non-health D-side phrasing drift.** The 115 non-health groups of the final pool draw benign prompts from OR-Bench rather than the exhausted xstest source. Compared with xstest, they are −25.7 pp in first-person framing, −7.8 pp in question form and +3 words in median length. The M3′ contract notes that the same drift applies to NEW_DEV. It matters only for direct non-health comparisons of C metrics against M1/M2.
- **Self-harm screening policy.** The frozen flagged-risk rule screened three keywords. A later read-only screen flagged 7 of 165 NEW_DEV groups and 26 of 576 final-pool groups, of which 4 and 12 were adjudicated as clearly self-harm. No rows were removed.

## 4. M3′ gate limitations

- **Sparse negative class.** BUILD `C_unnecessary_refusal` = 10 cases (and `B_safe_refusal_or_redirection` = 110). With a class this thin, the gate is closer to a B-vs-C classifier than to a true intervention-need classifier.
- **Case-level supervision.** Every trajectory step carries its case's endpoint label, and there is no step-level ground truth for when an intervention was needed.
- **Scope.** The frozen 9-feature L16 representation, the frozen ℓ2 logistic gate, the frozen threshold rule and the frozen `V_ALL` actuator.

## 5. Tooling defects recorded, not repaired

- The M3 trajectory extractor (`extract_trajectory_features_v1.py`) cannot reproduce BUILD inputs: it has no DIJA infill structure and no chat template. It never ran on BUILD, M3′ did not use it, and it is **not published**.
- **Hashing span.** DIJA outputs must be hashed over the masked-position tokens and ReNeLLM outputs over the standard output region. Comparing full decoded output falsely flags every DIJA row.
