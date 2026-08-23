# Study 1/2/3 union recapture — FROZEN schema (v2)

`SCHEMA_VERSION = "study1.union.v2"` (dense) / `"study1.union.v2.ragged"` (out_mask-only).

Engine: `study1/capture_union.py` — clockv2 `fill_all_masks` unified schedule
(temp=0, low_confidence remask, 128 steps) + Shield `StepActivationCollector(pool="none")`,
extended with the two fields the scalar A-set capture discarded. **No activation is
modified.** Verified on the 10-case pilot (dija + clean, on-node H100 MIG 3g.40gb).

## What changed from v1 (region coverage)

v1 captured from `gen_start = P` (the continuation start). For the **injected arms**
(`dija`, `benign_op`) the harm is not written in the continuation — it is written into
the `<mask:N>` worksheet blanks that live **inside the prompt**, at token span `[t0,t1)`
with `t1 ≤ P`. The Shield collector slices a contiguous tail (`h[0, gen_start:]`), so
`gen_start = P` would have missed every position where harm actually forms.

v2 therefore sets **`gen_start = t0` for injected arms** (still `P` for `clean`) and
labels each captured position by region:

| `region_id` | name | meaning |
|---|---|---|
| 0 | `R_SCAFFOLD` | `[t0, t1)` — the injected worksheet, where DIJA harm forms |
| 1 | `R_PROMPT_TAIL` | `[t1, P)` — template text between scaffold and continuation |
| 2 | `R_CONTINUATION` | `[P, total)` — the free reply |

`fillable_reply_mask [region_len] bool` = position was masked at step 0 (i.e. it is
actually generated). Never-masked prompt text is captured but excluded from every
analysis; it carries `commit_step = -1`.

## Per-case `.pt` payload — dense (`{id}__{arm}.pt`)

| field | shape | dtype | notes |
|---|---|---|---|
| `hidden` | `[n_steps, n_layers, region_len, d_model]` | fp16 | **never mean-pooled** |
| `reply_role_mask` | `[n_steps, region_len]` | bool | **True = out_mask, False = out_unmask**, read BEFORE forward |
| `region_id` | `[region_len]` | int8 | 0/1/2 per the table above *(v2)* |
| `fillable_reply_mask` | `[region_len]` | bool | ever-masked ⇒ generated *(v2)* |
| `commit_step` | `[region_len]` | int16 | `k_i = min{k : i ∉ M(x_tk)}`; `-1` if never masked |
| `entropy` | `[n_steps, region_len]` | fp16 | H(p) |
| `confidence` | `[n_steps, region_len]` | fp16 | argmax prob |
| `topk_probs` / `topk_ids` | `[n_steps, region_len, 5]` | fp16 / int32 | compact logits |
| `final_ids` | `[region_len]` | int64 | decoded tokens |
| `mask_ratio_reply` | `[n_steps]` | float | physical canvas clock |
| `meta` | — | — | id, arm, behavior, scaffold, response, prompt_len, `gen_start`, `region_len`, `t0`, `t1`, `total`, n_inject, schedule, gen_params, layers, vocab, d_model |

For `clean`, `t0 = t1 = P`, `region_id ≡ 2`, and every position is fillable.

**Invariants enforced by `verify_payload()` (hold on all pilot cases):**
`commit_step ∈ [0,steps)` on fillable positions (`-1` elsewhere); masked count
monotonically non-increasing; `reply_role_mask[i] == (commit_step ≥ i)` on fillable
positions (out_mask/out_unmask exactly separable); shapes/dtypes as above.
Round-trip (save→load) verified per case.

## Ragged out_mask-only variant (`--storage ragged`)

`out_unmask` is the **forbidden read** for the k_detect probe, so it is dropped at
write time. The cut is applied **after** `commit_step` is logged for every position —
no position is discarded before its commit is recorded.

CSR-style layout, cells = `reply_role_mask & fillable_reply_mask`:

| field | shape | notes |
|---|---|---|
| `hidden_flat` | `[T, n_layers, d_model]` | fp16; `T = Σ_i n_i` out_mask cells |
| `step_ptr` | `[n_steps+1]` | cells of step `i` = `hidden_flat[step_ptr[i]:step_ptr[i+1]]` |
| `cell_pos` | `[T]` | local position index of each cell |
| `cell_step` | `[T]` | step index of each cell |

All aux fields (`commit_step`, `region_id`, `fillable_reply_mask`, `entropy`,
`confidence`, `topk_*`, `final_ids`, `mask_ratio_reply`, `meta`) are kept verbatim.
`ragged_out_mask_at_step(payload, i)` reconstructs one step; `verify_ragged(dense, ragged)`
checks it is lossless over the retained cells (returned `[]` for every measured case).

## Arms
`clean` (chat template) · `dija` (harmful, `run_dija.PaperRunner.build_inputs`) ·
`benign_op` (injected-but-benign = **Group C**, `clockv2/data/benign_op_matched.json`).
The benign_op scaffold carries **byte-identical `<mask:N>` markers** to the dija scaffold
for the same A-id (asserted in `clockv2/build_benign_op.py` and re-asserted at capture),
so the two arms have identical blank geometry and differ only in outcome. Same A-id =
one base request → **grouped CV by A-id**, and k_detect must separate outcome
*within* matched behavior.

## Storage (MEASURED on real pilot cases, 5 layers)

| arm | dense MiB/case | ragged MiB/case | ratio |
|---|---|---|---|
| dija (A000/A003/A008) | 1396 / 1461 / 1411 | 410 / 444 / 426 | ~30% |
| clean (A000) | 641 | 323 | 50% |

dija compresses harder than clean because its region includes the never-masked prompt
tail, which is out_unmask at every step and so drops out entirely.

| run | ragged total | budget |
|---|---|---|
| expansion: 25 × {dija, benign_op} | **20.8 GiB** | OK |
| full: 65 × 3 arms | **74.7 GiB** | under 100 GiB |
