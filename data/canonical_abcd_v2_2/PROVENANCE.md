# Provenance of canonical ABCD v2.2

## Lineage
- **A** — harmful clean prompt: a verbatim upstream source prompt (500, human-curated and adjudicated).
- **B** — DIJA transformation of A (official DIJA refinement).
- **D** — benign rewrite paired with A (LLM-generated rewrite, LLM-judged, with human adjudication where flagged).
- **C** — DIJA transformation of D.

B and C therefore derive from A and D respectively; D derives from A. Each group `NNNN` holds `A_NNNN`, `B_NNNN`,
`C_NNNN`, `D_NNNN`.

## Upstream sources of the 500 A prompts (audited)

| Source | Groups |
|---|---|
| MedSafetyBench | 281 |
| BeaverTails (PKU-Alignment) | 79 |
| HarmBench | 54 |
| AdvBench | 44 |
| Do-Not-Answer | 42 |
| **Total** | **500** |

These counts come from the per-row records of the final adjudicated A set (source dataset, source row id, source
file hash); 421 rows were re-read from the upstream files by row id, and the 79 BeaverTails rows are supported by
recorded row id and file hash.

## License facts and why the file is private
- BeaverTails: CC BY-NC 4.0 (dataset card).
- Do-Not-Answer: data CC BY-NC-SA 4.0; code Apache-2.0 (upstream README).
- HarmBench, AdvBench, MedSafetyBench: the repositories are MIT-licensed; no separate data license was found, and
  MedSafetyBench asks that its data be used for research only.

Because the upstream terms are mixed, and B, C and D are derived from A, the dataset is shared privately with
authorized collaborators rather than published. Users of the file remain responsible for the upstream terms.
