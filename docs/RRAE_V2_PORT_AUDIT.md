# RRAE-v2 (July) port audit

The local working tree held 12 untracked files under `experiments/rrae_v2/`, left over from a checkout of the July `research/rrae-v2-release` branch (`d8dcb34`). Each file was compared with current `main`, with the release staging, and with the full tree of `research/rrae-v2-release`. **None of the 12 was ever tracked on that branch.** They were untracked because that branch's `.gitignore` excluded `*.json` and `hidden_states/`. None exists on `main`.

| File | Class | Reason |
|---|---|---|
| `configs/phase3_candidates_v2.json` | LEGACY_NOT_NEEDED | Pilot config for the July input-region RRAE-v2 injection-direction phase 3. It refers to scripts and vector files that exist only on the unmerged July branch, so on its own it is orphaned. Not part of the final program. |
| `model_metadata/LLaDA-8B-Instruct/config.json` | DUPLICATE | Upstream Hugging Face model metadata; available from the model repository. |
| `model_metadata/LLaDA-8B-Instruct/generation_config.json` | DUPLICATE | Same |
| `model_metadata/LLaDA-8B-Instruct/model.safetensors.index.json` | DUPLICATE | Same (weight index only) |
| `model_metadata/LLaDA-8B-Instruct/special_tokens_map.json` | DUPLICATE | Same |
| `model_metadata/LLaDA-8B-Instruct/tokenizer_config.json` | DUPLICATE | Same |
| `results/summaries/directions_L11.json` | LEGACY_NOT_NEEDED | Aggregate train/held-out cosine diagnostics for the July input-region injection directions. Public-safe, but produced by scripts on the unmerged branch and not cited by the final program. |
| `results/summaries/directions_L16.json` | LEGACY_NOT_NEEDED | Same |
| `results/summaries/summary__L11__r24.json` | LEGACY_NOT_NEEDED | July RRAE r24 reconstruction summary. Refers to private checkpoint and hidden-state paths, and is superseded by the final RRAE selection on `main`. |
| `results/summaries/summary__L16__r24.json` | LEGACY_NOT_NEEDED | Same |
| `slurm/hidden_states/extract_abcd_v2_input_region.slurm` | PRIVATE/UNSAFE | Cluster launcher containing an allocation account name. It drives a script that exists only on the unmerged branch. |
| `slurm/hidden_states/extract_heldout_v1_input_region.slurm` | PRIVATE/UNSAFE | Same |

**PORT: 0 files.** The July branch's tracked `experiments/rrae_v2/` release (167 files) is also **not** merged by this release. Whether to bring it in is a separate decision.
