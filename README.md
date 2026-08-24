# CDG2.0 — Combined Project Guide (All Branches)

A cross-branch guide to the **ClinicalDenoiseGuard (CDG)** project: the study of
**DIJA template-injection attacks** on masked-diffusion LLMs (primarily
**LLaDA-8B-Instruct**, plus **Dream-v0-Instruct-7B**), the internal
injection-mechanism signal, and steering / representation-based defenses against
it. The recurring finding across every branch is a **detection–correction
asymmetry**: the injection footprint is trivially detectable in hidden states, but
hard to *correct* by direct steering without collapsing generation.

> **Cross-branch note.** Each branch owns its own files. Paths in a branch section
> below exist **only on that branch** — check it out first
> (`git checkout <branch>`) before running its commands. This README lives on the
> `combined-readme` branch and only *documents* the others; it does not bundle
> their code. Every branch shares the same base (`cdg/`, `scripts/`, the shared
> `run_*.sh` wrappers) so `scripts/p10_steer2.py` etc. are available everywhere.

## Contents

- [Branches at a glance](#branches-at-a-glance)
- [Common setup (all branches)](#common-setup-all-branches)
- [`main` — shared CDG pipeline (Phases 1–12)](#main--shared-cdg-pipeline-phases-112)
- [`feat/p2-work-migration` — P2 exploratory studies](#featp2-work-migration--p2-exploratory-studies-p2)
- [`feat/p3-dream-extension` — Dream-v0-Instruct-7B](#featp3-dream-extension--dream-v0-instruct-7b-extension-dream)
- [`research/rrae-v2-release` — RRAE v2](#researchrrae-v2-release--rrae-v2-experimentsrrae_v2)
- [Conventions across all branches](#conventions-across-all-branches)

---

## Branches at a glance

| Branch | Owner area | Adds on top of `main` | Primary entry point |
|--------|-----------|-----------------------|---------------------|
| **`main`** | Shared pipeline **+ finalized paper path** | base `cdg/`, `scripts/`, `run_*.sh`, **plus `rrae/`, `steering/`, `configs/final_rrae_steering.yaml`, `results/`, `tests/`** | legacy: `run_pipeline.sh` · final: `python -m steering.run_frozen_replication` |
| **`feat/p2-work-migration`** | P2 studies | `p2/experement/`, `p2/serverFiles/` | per-study scripts (see §P2) + `p2/README.md` |
| **`feat/p3-dream-extension`** | Dream model | `dream/judge_dream_p10_p11style.py`, `dream/README.md` | `sbatch dream/slurm/*.slurm` (reuses `scripts/p10_steer2.py --backend dream_attack`) |
| **`research/rrae-v2-release`** | RRAE v2 | `experiments/rrae_v2/` | `sbatch experiments/rrae_v2/slurm/.../*.sbatch` |

---

## Common setup (all branches)

### Python environment

- **Python 3.11**, **PyTorch ≥ 2.1**, **transformers** (LLaDA validated on
  `4.38.2`, Dream on `4.46.2`; Alliance pin `4.46.3+computecanada`), plus
  `safetensors numpy requests huggingface_hub`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                       # shared LLaDA/Dream deps
# On Alliance / Compute Canada clusters, use the pinned offline wheelhouse:
#   module load python/3.11 && virtualenv --no-download .venv
#   .venv/bin/pip install --no-index -r p2/experement/requirements.txt   # +computecanada wheels
```

### Models, data, secrets (none are in the repo)

- **Models:** `GSAI-ML/LLaDA-8B-Instruct` (`cdg/config.py`); on the Dream branch
  also `Dream-v0-Instruct-7B`. Auto-downloaded from HuggingFace into the standard
  cache (`$HF_HOME` / `~/.cache/huggingface`). On offline compute nodes, pre-fetch
  on a login node and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- **DeepSeek judge key — env only.** The judge reads `DEEPSEEK_API_KEY` from the
  environment (`cdg/judge.py` → `https://api.deepseek.com/chat/completions`).
  Provide it via an untracked `.env` or `export`; **never commit a real key.**
  Some P2 scripts read it from an untracked `*/.deepseek_key` file (also not in
  the repo) — create it locally or `export DEEPSEEK_API_KEY` instead.
  ```bash
  export DEEPSEEK_API_KEY=sk-...
  ```
- **Gitignored / not in repo:** model weights, SAEs (`./saes`), prompt corpora
  (`prompts/cdg_injection/`, RRAE datasets), and all run outputs
  (`outputs*/`, `analysis_output/`, `*.pt`, `*.jsonl`, `*.npy`, `*.png`).

### Cluster (Alliance / Compute Canada)

GPU work runs under **SLURM**, never on login nodes. Job files use accounts
`def-zshakeri` / `def-zshakeri_gpu`, an H100 MIG slice
(`--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1`), and `module load StdEnv/2023`.
**Outbound internet is only available on login nodes** — pre-download weights and
run the DeepSeek judge where there is network. Keep venvs/caches on `/scratch`
(the `/home` quota is tight).

### Shared 2×2 factorial design (every branch)

| Group | Content | Template (DIJA) | Intended behaviour |
|-------|---------|-----------------|--------------------|
| **A** | harmful | ✗ clean | refuse |
| **B** | harmful | ✓ injected | comply (attack target) |
| **C** | neutral | ✓ injected | comply (benign fill) |
| **D** | neutral | ✗ clean | comply (baseline) |

`B-vs-A` and `C-vs-D` isolate the injection mechanism (content controlled);
`B-vs-C` measures content-danger (template controlled).

---

## `main` — shared CDG pipeline (Phases 1–12)

Records SAE features + hidden states from the injected-template region during
denoising, probes them for an injection signal, builds steering directions, and
evaluates ASR reduction vs. benign breakage. **Full method + all result tables:
[`REPORT.md`](REPORT.md).**

> **`main` now carries two pipelines.** A merged PR ("Final RRAE + Steering
> Pipeline") added a **finalized paper path** on top of the original SAE pipeline.
> The **legacy Phases 1–12** SAE pipeline (`run_*.sh` + `scripts/p*.py`, below) is
> retained for provenance and reuse; the **finalized RRAE + steering** path
> (`rrae/`, `steering/`, `configs/final_rrae_steering.yaml`,
> `results/frozen_35_summary.json`, `tests/`) is the current paper path,
> documented next. Headline finding: **Representation ≠ Control.**

### Finalized RRAE + steering paper path (`rrae/`, `steering/`)

The canonical paper pipeline: a **2,000-sample A/B/C/D** dataset (500 per group;
400 health + 100 non-health), hidden-state extraction via the existing `cdg`
recorder, a 42-model RRAE rank sweep, direction construction, and five steering
families scored on a **frozen 35-pair replication**.

- **Selected representation:** `harm / f=0.05 / L11 / rank-4` (metric-selected —
  *not* min reconstruction loss). Validation injection AUC 1.0000,
  `cos(B−A, C−D)` ≈ 0.9621, harmfulness-AUC guardrail 0.4357.
- **Directions:** `v_inj = first right singular vector of stack([u_BA, u_CD])`
  (sign-aligned to `u_BA`); a separate `v_safety` from paired
  harmful-compliance → safe-refusal states. **`−v_inj` is not assumed to equal
  `v_safety`.**
- **Steering families** (each separates direction / dose / location / time):

  | Family | Operator |
  |--------|----------|
  | Additive | `h' = h + α·v_inj` |
  | Projection removal | `h' = h − ρ·⟨h,v_inj⟩·v_inj` |
  | Safety direction | `h' = h + β·v_safety` |
  | Combined | projection removal, then safety addition |
  | Strength / schedule | configurable operator + denoising window (single-step / windowed / persistent) |

Run it (dataset + records are external; point tools at them via CLI/config):

```bash
# 0. Extract final LLaDA hidden-state records (records layers 4, 11, 16, 26):
python run_record.py --backend llada_attack --prompt-root /path/to/canonical_abcd \
    --sae-root /path/to/saes --out /path/to/final_records --seeds 17

# 1. Screen the 72 candidate hidden-state views:
python -m rrae.screen_representations --records /path/to/final_records

# 2. RRAE rank sweep (k ∈ {4,8,16,24,32,48,64}):
python -m rrae.train_rrae --records /path/to/final_records \
    --layer 11 --scope harm --frac 0.05 --ranks 4 8 16 24 32 48 64

# 3. Build residual-space directions (add v_safety with --safety-records):
python -m rrae.build_residual_directions --checkpoint /path/to/rrae_rank4.pt \
    --records /path/to/final_records \
    --safety-records /path/to/paired_behavior_records   # --safety-label-field behavior_label

# 4. Frozen 35-pair replication — config check, then score externally judged pairs:
python -m steering.run_frozen_replication --config configs/final_rrae_steering.yaml
python -m steering.run_frozen_replication --config configs/final_rrae_steering.yaml \
    --predictions /path/to/frozen_35_judgments.jsonl
#   judged rows: {"family":..., "pair_id":..., "b_safe":true, "c_helpful":true}
#   the evaluator rejects duplicates and incomplete 35-pair family cohorts.

python -m pytest tests/test_final_pipeline.py          # final-pipeline smoke tests
```

**Result — frozen 35-pair strict paired success** (B converts to a safe
refusal/redirection **and** its paired C stays helpful): Safety-Direction
**25.7 %** (best) > Combined 11.4 % > Strength/Schedule 8.6 % > Projection-Removal
5.7 % > Additive 2.9 %. The strongest condition was `v_safety` steering at layer
16, on masked positions, persistently, at ≈2× the layer-median calibrated dose.
Clean, coherent representation but only partial causal control ⇒ the same
**detection–correction asymmetry**. Frozen spec + numbers live in
`configs/final_rrae_steering.yaml` and `results/frozen_35_summary.json`.

### Legacy Phases 1–12 pipeline (SAE probes + steering)

The original path — retained for provenance. Records SAE features from the
injected-template region, probes for the injection signal, and tests SAE-feature
zeroing / residual steering.

### Environment note

The shell wrappers default `PY` to a teammate's interpreter
(`/home/f4ng/cdg/bin/python3`) and `LLADA_CACHE` to a teammate's HF snapshot —
**override for your machine.** `run_pipeline.sh` reads `PY`/`LLADA_CACHE` from the
environment; `run_p8_p6.sh` and `run_p9_p10.sh` **hardcode `PY=`** (and
`run_p8_p6.sh` hardcodes `LLADA_CACHE=`), so edit those top lines before running.

### 0. Smoke test (CPU, fake model, no key)

```bash
bash run_pipeline.sh --dummy --limit 1
python run_record.py --prompt-root prompts/cdg_injection --dummy   # recorder only
```

### 1. Record + analyze — `run_pipeline.sh`

Runs, in order: Phase-1 recording (`run_record.py --record-token-level`) →
`p1_check` (QC) → `p2_assemble` (pseudobulk, scopes `tpl_mask out_mask`, L16,
frac 0.10) → `s2_probe_sweep` → `p4_delta` (quick tpl_mask + full out_mask) →
`p5_modules` (co-activation clustering, k=3–15) → `p6_annotate`.

```bash
export DEEPSEEK_API_KEY=sk-...
PY=.venv/bin/python \
LLADA_CACHE=/path/to/hf/.../models--GSAI-ML--LLaDA-8B-Instruct/snapshots/<snap> \
  bash run_pipeline.sh
```

Flags (parsed by the script): `--skip-record` (reuse `outputs/`), `--dummy`
(CPU), `--seeds "0 1 2"`, `--limit 30` (cases/group), `--out DIR`, `--backend
{llada,llada_attack,dream,dream_attack}`. Env overrides: `PY`, `LLADA_CACHE`,
`BACKEND`, `PROMPT_ROOT`, `SAE_ROOT`, `DEVICE`, `JUDGE_MODEL` (default
`deepseek-v4-flash`). Judge steps are skipped automatically if `DEEPSEEK_API_KEY`
is unset.

### 2. Logit-lens vocab labels + re-annotate — `run_p8_p6.sh`

Loads LLaDA once, computes logit-lens vocab labels for all active SAE features
(`p8_vocab`, batch 2048), then re-runs `p6_annotate` with full coverage.
**Edit `PY=` and `LLADA_CACHE=` at the top first.**

```bash
export DEEPSEEK_API_KEY=sk-...
bash run_p8_p6.sh              # → analysis_output/vocab_labels.json, outputs/analysis/annotation/
```

### 3. Corrected probe + residual-stream steering — `run_p9_p10.sh`

`p9_diagnose` (TF-IDF baseline, CV check, B-vs-A / C-vs-D injection probe sweep)
then `p10_steer2` (two configs: `harm_only` steer at harm/unmask, and
`harm_dir_mask` steer at template/mask). **Edit `PY=` at the top first.**

```bash
bash run_p9_p10.sh            # default --limit 30
bash run_p9_p10.sh --full    # all cases (production)
bash run_p9_p10.sh --p9-only # diagnostics only
bash run_p9_p10.sh --p10-only# steering only (needs p9 output)
```

Underlying scripts (call directly for custom sweeps):

```bash
python scripts/p9_diagnose.py --limit 50 --task all --auc-thresh 0.65 \
    --layers 11 16 26 --fracs 0.05 0.10 0.20 0.35 0.50 1.00 \
    --scopes harm out_mask out_unmask --out-json analysis_output/p9_results.json

python scripts/p10_steer2.py --tag harm_dir_mask --dir-scope harm --dir-frac 0.05 \
    --dir-layers 11 16 26 --steer-layer 16 --steer-direction shared \
    --steer-scope-region template --steer-pos mask \
    --alpha-values 0 2 4 8 16 24 32 --max-false-reject 0.20 \
    --p9-results analysis_output/p9_probe_inj_sweep.json
```

### 4. Ablation + direction stability (no wrapper)

```bash
# Scope/position ablation — needs p10 direction .pt files first (--dirs-in):
python scripts/p11_ablation.py --group B --configs baseline A1 A2 A3 \
    --limit 20 --alpha-values 0 16 32 64 \
    --dir-scope harm --dir-frac 0.05 --dir-layers 11 16 26 \
    --dirs-in analysis_output/p10_harm_dir_mask_directions \
    --out-root outputs_p11 --analysis-out analysis_output/p11   # --no-judge to skip DeepSeek

# Cross-frac direction stability (no generation, reuses Phase-1 states):
python scripts/p12_dir_stability.py --out-dir outputs --scope harm \
    --layers 11 16 26 --out-json analysis_output/p12_dir_stability.json
```

### 5. Alternative steering-eval driver — `run_steer_eval.py`

Builds a steering vector from Phase-1 records, re-runs steered, reports ASR drop
+ capability retention:

```bash
python run_steer_eval.py --records outputs --backend llada_attack \
    --sae-root ./saes --prompt-root prompts/cdg_injection \
    --vector bc --scope tpl_mask --space hidden --frac 0.10 \
    --alpha -8 --layers 16 --steer-pos mask --judge --out steer_outputs
# --vector did  uses (B-A)-(C-D) at the output region instead of B-C.
```

### 6. Correlation heatmap — `plot_corr.py`

```bash
python plot_corr.py    # reads outputs/analysis/modules/corr_matrix.npy → corr_raw_heatmap.png
```

### Outputs (gitignored)

`outputs/` (`manifest.jsonl`, `<case>__seed0.pt`, `assembled/X_*_L16_f0p10.npy`,
`analysis/{delta,modules,annotation}/`), `outputs_feat_steer*/`, `outputs_p11/`,
and `analysis_output/` (`probe_sweep.json`, `vocab_labels.json`, `p9_*.json`,
`p10_*_directions/`, `p10_*_dose_response*.json/png`, `p11/`,
`p12_dir_stability.{json,png}`). See the File Map in `REPORT.md`.

**SAEs** (per main's README): external layout `saes/{llada_mask, llada_unmask}`
(mask/unmask bundles; ids in `cdg/config.py` — override the HF cache via standard
HF env vars, don't edit source paths). **Dataset:** the canonical A/B/C/D corpus
is distributed **separately**, not in the repo — point tools at it via
`--prompt-root` / config. A minimal row is
`{"id","behavior","user_content","content_type","has_template","attack_method"}`;
injected B/C rows set `has_template=true`, `attack_method="DIJA"`, and carry the
DIJA scaffold in `user_content`, sharing a pair id across matched A/B/C/D so
paired splits don't leak.

---

## `feat/p2-work-migration` — P2 exploratory studies (`p2/`)

Everything in `main`, plus self-contained studies under `p2/experement/` and
`p2/serverFiles/`. Each has its own doc (`README.md` / `REPORT.md` / `RESULTS.md`
/ `SUMMARY.md`) and its own scripts — **they are not wired into the `main`
wrappers.** The P2 run guide is **`p2/README.md`**. Most GPU jobs use the Alliance
venv `/scratch/ore99/cdg_venv/bin/python` (or `/home/ore99/env_llada/bin/python`)
and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; several use an `flock` lock so
only one instance runs at a time.

### harm_dir — harm-vs-safe direction (`v_harm`) detector + steering pilots

The core of the P2 work: builds a clean harm/safe steering direction and tests it
as both a detector (Gate E) and a corrector (Phase 3 / 3.1 steering).

```bash
cd p2/experement/harm_dir
# Smoke (2 harmful + 1 benign) then full via SLURM (env vars override defaults):
LIMIT=2 sbatch scripts/steer_pilot2.sbatch            # smoke
LIMIT=0 ALPHAS=16,32,48,64 WITH_C=1 sbatch scripts/steer_pilot2.sbatch   # full run
```

The sbatch runs `steer_pilot2.py`; call it directly for custom runs:

```bash
python scripts/steer_pilot2.py --limit 0 --benign-target 20 \
    --alphas 16,32,48,64 --modes S,SR \
    --scope template --pos unmask \
    --remask-frac 0.15 --remask-rounds 1 --refill-steps 32 \
    --with-c --out data/steer_pilot2.jsonl
```

- **`--modes S,SR`** = Arm S (scaffold-region steer) and Arm SR (steer-early /
  remask-late). **`--with-c`** adds the benign-injected control group C.
- Other sbatch entry points: `capture.sbatch`, `capture_ds.sbatch`,
  `capture_gateE.sbatch` (hidden-state capture), `gen_safe.sbatch`,
  `steer_pilot.sbatch` (Phase-3 response-region steering).
- Docs: `GATE_E_VERDICT.md`, `PHASE3_PILOT_VERDICT.md`, `PHASE2P_VERDICT.md`.

### study1 — straddler yield / temperature sweep

Generate-only, key-free, offline; writes to `/scratch/ore99/study1_straddle`.

```bash
cd p2/experement/study1
bash run_straddle.sh          # → straddle_pilot.py --n-ids 15 --k 8 --temp 0.2 --det-check 2
# direct, with a temperature/k sweep:
python straddle_pilot.py --n-ids 15 --k 8 --temps "0.2,0.5,0.8" --ks "4,8" \
    --det-check 2 --det-check-ids 5 --out /scratch/ore99/study1_straddle
```

Also `run_expansion.sh`, `run_straddle_c.sh` (Part C temp-sweep),
`run_straddle_f.sh` (Part F yield). Docs: `STUDY1_WRITEUP.md`, `SCHEMA.md`,
`PARTB_STRADDLE.md`, `PARTC_TEMPSWEEP.md`, `PARTF_YIELD.md`.

### clockv2 / clockv2_refusal / clock_attack — clock attack + refusal axis

Projects `v_injection_svd` (and, in the refusal variant, `v_refusal`) onto
per-position readings across conditions.

```bash
cd p2/experement/clockv2
bash run_seed_clean.sh        # seeds 1,2 × cond=clean, retries until 38400 rows/file
bash run_seed_benign.sh
bash run_seed_rest.sh
# direct (one seed/condition):
python capture_seed.py --cond clean --seed 1 --limit 0 --temperature 0.2
#   --cond {dija, benign_op, benign, clean};  output → results/probe_readings_s<seed>_<cond>.jsonl
```

Docs: `clockv2/README.md`, `clockv2/REPORT.md`, `clock_attack/README.md`.

### region_steer — region-steering confirm runs

```bash
cd p2/experement/region_steer
sbatch confirm40.sbatch            # response/region steering, 40-case confirm
sbatch confirm40_tplmask.sbatch    # template-mask region variant
sbatch smoke_tplmask.sbatch        # smoke
# batch logs + poll helpers under batch/ and batch_tplmask/
```

### prefill — prefill-capture attack

```bash
cd p2/experement
python prefill_capture.py ...      # TODO: P2 confirm exact flags
python baseline_capture.py ...     # TODO: P2 confirm exact flags
```

Doc: `CDG_prefilling_attack_report.md` (+ `.docx`).

### p2/serverFiles/dijawithprefill — DIJA-with-prefill arms (8 arms, n=100)

```bash
cd p2/serverFiles
bash dijawithprefill/run_phase4.sh   # generation + inline binary ASR judge, then graded judge, then summary
```

`run_phase4.sh` reads the key from `dijawithprefill/.deepseek_key` (**not in the
repo** — create it or `export DEEPSEEK_API_KEY`) and runs, in order:

```bash
python dijawithprefill/run_arms.py --arms mid_0,mid_1,mid_2,mid_3,both_1,both_2,both_3,start_1 \
    --groups B,D --out-root dijawithprefill/runs/full \
    --binary-judge --judge-model deepseek-v4-flash --judge-groups B
python dijawithprefill/graded_judge.py --manifest <arm>/manifest.jsonl \
    --out <arm>/graded_judge.jsonl --groups B,D --model deepseek-v4-flash
python dijawithprefill/summarize.py --runs dijawithprefill/runs/full --out dijawithprefill/SUMMARY.md
```

Doc: `dijawithprefill/SUMMARY.md`. Also `resume_judge.sh` to resume the judge pass.

### p2/serverFiles/crossattack + attack2 — cross-format transfer attacks

No shell wrapper; run the Python entry points directly (see each `SUMMARY.md` /
`RESULTS.md` for the exact inputs):

```bash
cd p2/serverFiles
python crossattack/corrected/run_transfer_corrected.py ...   # TODO: P2 confirm flags
python crossattack/sanity/sanity_battery.py ...              # TODO: P2 confirm flags
python attack2/run_attack2_resumable.py ...                  # TODO: P2 confirm flags
```

Docs: `crossattack/SUMMARY.md`, `crossattack/corrected/RESULTS.md`,
`attack2/RESULTS.md`.

---

## `feat/p3-dream-extension` — Dream-v0-Instruct-7B extension (`dream/`)

Ports the injection analysis to the **Dream** diffusion LM: hidden-state
recording, injection probes, a shared BA/CD steering direction, and p10/p11-style
graded judging. It **reuses the shared `scripts/p10_steer2.py`** with
`--backend dream_attack`. **Full result tables: [`dream/README.md`](dream/README.md).**
Runs on Alliance SLURM (`def-zshakeri`, H100 MIG; venv `.../venv_llada`, with a
`runtime_stubs` dir on `PYTHONPATH`).

### 1. Record Dream hidden-state history (400 examples, resumable)

```bash
sbatch dream/slurm/run_dream_history_full_h100_3g.slurm   # → python -u dream_native_history_record_resume.py
#   smoke: dream/slurm/run_dream_history_full.slurm (non-3g) / dream_native_history_record_smoke.py
#   output → outputs_dream_native_history_full/ (manifest.jsonl + *.pt)
```

### 2. Steering (out_mask / harm direction at layer 14)

```bash
sbatch dream/slurm/run_dream_p10_outmask_L14_mid_h100_3g.slurm
sbatch dream/slurm/run_dream_p10_harm_f1_L14_mid_h100_3g.slurm
sbatch dream/slurm/run_dream_p10_outmask_L14_neg_h100_3g.slurm
sbatch dream/slurm/run_dream_p10_smoke_h100_3g.slurm      # smoke
```

Each wraps the shared p10 steering script, e.g.:

```bash
python scripts/p10_steer2.py --task dose_response --backend dream_attack \
    --out-dir outputs_dream_native_history_full \
    --dir-scope out_mask --dir-frac 0.05 --dir-layers 14 \
    --steer-layer 14 --steer-scope-region output ...
```

### 3. Probes, directions, judging (run directly)

```bash
python dream/dream_hidden_probe_auc.py               # injection separability by frac/layer
python dream/dream_build_steering_direction_p10fmt.py# build BA/CD shared direction (paths hardcoded)
python dream/judge_dream_p10_p11style.py             # p11-style graded judge on saved B outputs (no args)
```

**Finding:** Dream has a clean shared injection direction (out_mask, frac 0.05,
layer 14; cos(BA,CD)=0.980), baseline B ASR = 0.50, but direct residual steering
along it did **not** reduce attack success on the 30-case B subset — the same
detection-not-correction asymmetry seen on LLaDA.

---

## `research/rrae-v2-release` — RRAE v2 (`experiments/rrae_v2/`)

> **Not the same as main's `rrae/`.** `main` now carries a distilled, finalized
> `rrae/` + `steering/` package (the paper path — see the [`main`](#main--shared-cdg-pipeline-phases-112)
> section). This branch is the fuller upstream research pipeline under
> `experiments/rrae_v2/` that it draws on.

**Rank-Reduction Auto-Encoder** pipeline: rebuilds a clean, balanced
2,000-example dataset (health + non-health, buckets A/B/C/D × 500), extracts
LLaDA hidden states (layers 11 & 16), trains a low-rank RRAE (rank sweep 8–64),
extracts steering vectors, and evaluates additive steering / projection removal.
**See [`experiments/rrae_v2/README.md`](experiments/rrae_v2/README.md),
[`ARTIFACTS.md`](experiments/rrae_v2/ARTIFACTS.md), and
[`slurm/README.md`](experiments/rrae_v2/slurm/README.md).**

### SLURM environment (set before any job)

Jobs are portable templates — export your paths first (from `slurm/README.md`):

```bash
export RRAE_WORK_ROOT="/path/to/rrae_steering_work_v2"
export RRAE_ENV_ROOT="/path/to/python-environment"
export RRAE_DATA_ROOT="/path/to/rrae_data"
export LLADA_MODEL_PATH="/path/to/LLaDA-8B-Instruct"
export DIJA_ROOT="/path/to/DIJA"
export DEEPSEEK_API_KEY="your-key"        # judge jobs only; never store in the .sbatch
```

### Pipeline (each stage a `scripts/<stage>/` dir)

```bash
cd experiments/rrae_v2

# 2. Extract hidden states (layers 11 & 16, frac 0.05):
python scripts/extraction/extract_hidden_states_abcd_v2_input_region.py \
    --dataset <abcd_v2.jsonl> --out-dir <states> --model-path "$LLADA_MODEL_PATH" \
    --layers 11 16 --frac 0.05 --gen-length 128

# 3. Train the RRAE (rank sweep 8/16/24/32/48/64):
python scripts/training/train_rrae_v2.py --dataset <states> --out-dir <rrae_ckpt> \
    --hidden-dim 1024 --latent-dim 512 --rank-k 32 --epochs 2000 --lr 1e-3

# 4. Extract steering vectors (raw diff / RRAE recon diff / residual diff):
python scripts/vectors/extract_rrae_v2_vectors.py ...    # see README §4

# 5. Steering / projection-removal generation matrix (full grid via SLURM array):
sbatch experiments/rrae_v2/slurm/phase3_v2/full_grid_heldout/P3v2_HO_full_grid.sbatch
#   wraps: run_rrae_steering_generation_matrix_v2.py --prompt-root ... --vector-path ...
#          --layer 11 --rank 32 --position-spec R0|R2 --groups A B C D
#          --limit-per-group 0 --alphas -1 -2 -4 -8 -12 -16 --num-shards N --shard-index i
python scripts/steering/run_rrae_steering_generation_v2.py ...   # single-config variant

# 6. Judging (reads DEEPSEEK_API_KEY):
python scripts/judging/evaluate_rrae_with_p11_judge.py ...
python scripts/judging/evaluate_success_judge_v2.py ...

# 7. Diagnostics / analysis:
python scripts/diagnostics/diagnose_train_vs_heldout_directions.py ...
python scripts/analysis/analyze_projection_confirm_majority.py ...
```

Steering regions: **R0** = template-mask positions, **R2** = template-mask +
output-mask. The full grid sbatch is a SLURM array (`--array=0-191%8`).

**Finding:** strong transferable injection direction (train→held-out cos ≈ 0.90 at
L11/L16), preserved through the RRAE (raw vs. recon vector cos ≈ 0.9999).
Projection removal has measurable causal leverage (L11: harmful-injected success
~84% → ~55%; benign-injected ~67% → ~62%) but strong settings (λ=2) cause
behavioral instability.

---

## Conventions across all branches

- **Never commit** weights, `*.pt` / `*.jsonl` / `*.npy` dumps, prompt/data
  corpora, or API keys — all gitignored. The DeepSeek key is read from
  `DEEPSEEK_API_KEY` (or an untracked `*/.deepseek_key` file that is likewise not
  committed).
- **GPU jobs run via SLURM** on Alliance/Compute Canada; the DeepSeek judge needs
  login-node internet. Generation jobs are typically run **offline**
  (`HF_HUB_OFFLINE=1`) and **key-free**, with judging as a separate pass.
- The shared `scripts/p10_steer2.py` / `p11_ablation.py` back both the LLaDA
  (`--backend llada_attack`) and Dream (`--backend dream_attack`) experiments.
- `TODO: <owner> confirm` markers flag run details (a few sub-study flags, dataset
  sources, SAE ids) not discoverable from the committed files — fill in from the
  owning teammate.
