# Provenance errata

These are documentation-level corrections found during the release audit. **The frozen artifacts were not modified.** Each erratum records what a frozen record says and what the audit verified on disk.

| # | Frozen record says | Verified | Effect |
|---|---|---|---|
| 1 | Project handoff §18.1/§57: `fa5ad065c0c1e8b69228bc12356e8ee1cdb10c38ea864d2e21f3d8bdabbfae62` is the "direction construction script". | That SHA belongs to `run_generalized_safety_direction_construction_baselines_v2.py`, a baseline-generation **helper imported by the steered-generation runner** (`BASELINE_PATH` in `run_generalized_safety_m1_eval_v1.py`). It does not build the V2 directions. No standalone script for V2 direction averaging and normalization was found (see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)). | Label only. The directions container SHA (`e43916ff…`) and per-direction SHAs are unaffected. |
| 2 | `M3_FINAL_RESULT_v2.json` `addendum_v2.finding.derivation_source.beta_freeze_sha256` = `0584fed6…`. | `0584fed689caf66e2df5c3e8c660aed3f05253e88d77d18275459a562cbe2bc9` is **`DIRECTION_FREEZE.json`**. The β freeze (`BETA_FREEZE.json`) is **`3e41744b9614e504393e41643e664dcc41cebe17ae64b7f518891879f9314408`**. | Label only. The β values and their lineage finding are unaffected. |
| 3 | `BUILD_GATE_FIT.json` (M3′) cites `build_feature_freeze_sha256` = `b148756d…`. | That is the **superseded** first BUILD feature freeze. The current freeze is `0d1257a3…`, and it records the supersession: only the `by_route` reporting field changed, and every check, count and shard-result SHA is identical. | None. The fit consumed the unchanged shard results. |
| 4 | M3′ contract §5 quotes DeepSeek/assistant agreement on BUILD B as 73.49%. | 73.49% (449/611) is from the **superseded v1** M3 BUILD/DEV judging. The authoritative v2 judging gives 88.05% (538/611). | Reporting only. The assistant labels were never used for fitting. |
