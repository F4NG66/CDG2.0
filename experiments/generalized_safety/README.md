# Generalized safety program scripts (M1 → M3′)

These are path-sanitized copies of the frozen scripts that produced the M1, M2, M3 and M3′ results. The only change from the originals is that hard-coded cluster paths now read `CDG_WORK_ROOT`, `CDG_LLADA_MODEL` and `CDG_DIJA_ROOT`. Scientific logic is unchanged.

- Original ↔ published SHA-256 mapping: [`docs/SCRIPT_PROVENANCE.md`](../../docs/SCRIPT_PROVENANCE.md)
- Stage map and environment: [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md)
- Frozen results: [`results/generalized_safety/`](../../results/generalized_safety/)

| Directory | Contents |
|---|---|
| `shared/` | L16 extraction, the steered-generation runner and its helpers, DIJA and ReNeLLM construction |
| `m1/` | M1 held-out generation, judging, unblinding and metrics |
| `m2/` | M2 contract, norm-preserving hook, displacement-matched β calibration, generation, judging, metrics |
| `m3/` | M3 feasibility spike, power simulation, scenario-match audit, BUILD/DEV judging v2 and support check |
| `m3prime/` | M3′ observer and baseline generation, judging, support check, BUILD capture, gate fit, threshold qualification |

These scripts depend on private frozen inputs, which are not published. See [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md).
