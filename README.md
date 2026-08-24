# CDG2.0 — Combined Project Guide (All Branches)

A cross-branch guide to the **ClinicalDenoiseGuard (CDG)** project: study of
**DIJA template-injection attacks** on masked-diffusion LLMs (primarily
**LLaDA-8B-Instruct**, plus **Dream-v0-Instruct-7B**), the internal
injection-mechanism signal, and steering / representation-based defenses against
it. The recurring finding across branches is a **detection–correction
asymmetry**: the injection footprint is easy to detect in hidden states, but hard
to *correct* by direct steering without collapsing generation.

> **Cross-branch note.** Each branch owns its own files. Paths in a branch section
> below exist **only on that branch** — check it out first
> (`git checkout <branch>`) before running its commands. This README lives on the
> `combined-readme` branch and only documents the others; it does not bundle their
> code.

---

## Branches at a glance

| Branch | Owner area | Adds on top of `main` | Primary entry point |
|--------|-----------|-----------------------|---------------------|
| **`main`** | Shared pipeline | — (the base: `cdg/`, `scripts/`, `run_*.sh`) | `run_pipeline.sh` → `run_p8_p6.sh` → `run_p9_p10.sh` |
| **`feat/p2-work-migration`** | P2 studies | `p2/experement/`, `p2/serverFiles/` | per-study scripts (see §P2) + `p2/README.md` |
| **`feat/p3-dream-extension`** | Dream model | `dream/judge_dream_p10_p11style.py`, `dream/README.md` (dir also on `main`) | `sbatch dream/slurm/*.slurm` |
| **`research/rrae-v2-release`** | RRAE v2 | `experiments/rrae_v2/` | `sbatch experiments/rrae_v2/slurm/.../*.sbatch` |

---

## Common setup (all branches)

- **Python 3.11**, **PyTorch ≥ 2.1**, **transformers** (LLaDA validated on
  `4.38.2`, Dream on `4.46.2`; Alliance pin `4.46.3+computecanada`). Plus
  `safetensors numpy requests huggingface_hub`.
- Create a venv and install:
  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt          # root: LLaDA/Dream shared deps
  # On Alliance clusters, use the pinned wheelhouse instead:
  #   pip install --no-index -r p2/experement/requirements.txt   (+computecanada wheels)
  ```
- **Models (not in repo):** `GSAI-ML/LLaDA-8B-Instruct` (`cdg/config.py`) and, on
  the Dream branch, `Dream-v0-Instruct-7B`. Downloaded from HuggingFace into the
  standard HF cache (`$HF_HOME` / `~/.cache/huggingface`).
- **Secret — env only:** the DeepSeek judge reads `DEEPSEEK_API_KEY` from the
  environment (`cdg/judge.py` → `https://api.deepseek.com/chat/completions`).
  Provide it via an untracked `.env` or `export`; **never commit a real key.**
  ```bash
  export DEEPSEEK_API_KEY=sk-...            # or an untracked .env
  ```
- **Gitignored / not in repo:** model weights, SAEs (`./saes`), prompt corpora
  (`prompts/cdg_injection/`, RRAE datasets), and all run outputs
  (`outputs*/`, `analysis_output/`, `*.pt`, `*.jsonl`, `*.npy`).
- **Cluster:** GPU work runs on **Alliance / Compute Canada** SLURM (accounts
  `def-zshakeri` / `def-zshakeri_gpu`, H100 MIG slice `3g.40gb`,
  `module load StdEnv/2023`). **Outbound internet is only on login nodes** —
  pre-download weights and run the DeepSeek judge where there is network; keep
  venvs/caches on `/scratch` (tight `/home` quota).

The 2×2 factorial design is shared across every branch:

| Group | Content | Template (DIJA) | Intended behaviour |
|-------|---------|-----------------|--------------------|
| **A** | harmful | ✗ clean | refuse |
| **B** | harmful | ✓ injected | comply (attack target) |
| **C** | neutral | ✓ injected | comply (benign fill) |
| **D** | neutral | ✗ clean | comply (baseline) |

---

## `main` — shared CDG pipeline (Phases 1–12)

Records SAE features + hidden states from the injected-template region during
denoising, probes them for an injection signal, builds steering directions, and
evaluates ASR reduction vs. benign breakage. **Full method + results:
[`REPORT.md`](REPORT.md).**

The shell wrappers default `PY` to a teammate's interpreter
(`/home/f4ng/cdg/bin/python3`) and `LLADA_CACHE` to a teammate's HF snapshot —
**override for your machine.** `run_pipeline.sh` reads `PY`/`LLADA_CACHE` from the
environment; `run_p8_p6.sh` and `run_p9_p10.sh` **hardcode `PY=`** (edit the line).

```bash
# 0. Smoke test (CPU, fake model, no key):
bash run_pipeline.sh --dummy --limit 1

# 1. Record + analyze  (record → p1_check → p2_assemble → s2_probe → p4_delta → p5_modules → p6_annotate)
export DEEPSEEK_API_KEY=sk-...
PY=.venv/bin/python LLADA_CACHE=/path/to/hf/.../LLaDA-8B-Instruct/snapshots/<snap> \
  bash run_pipeline.sh                 # flags: --skip-record  --seeds "0 1 2"  --limit 30  --backend llada_attack

# 2. Logit-lens vocab labels + re-annotate  (edit PY= and LLADA_CACHE= at top first)
bash run_p8_p6.sh

# 3. Corrected injection probe + residual-stream steering  (edit PY= at top first)
bash run_p9_p10.sh                     # or --full | --p9-only | --p10-only

# 4. Manual (no wrapper):
python scripts/p11_ablation.py         # TODO: P2 confirm exact flags (config, alpha, limit)
python scripts/p12_dir_stability.py --out-dir outputs --scope harm --layers 11 16 26

# 5. Alternative steering-eval driver:
python run_steer_eval.py --records outputs --backend llada_attack \
    --sae-root ./saes --prompt-root prompts/cdg_injection \
    --alpha -8 --layers 16 --judge --out steer_outputs

# 6. Correlation heatmap:
python plot_corr.py                    # reads outputs/analysis/modules/corr_matrix.npy
```

**Outputs (gitignored):** `outputs/` (manifest.jsonl, `*.pt`, `assembled/`,
`analysis/{delta,modules,annotation}`), `outputs_feat_steer*/`, `outputs_p11/`,
and `analysis_output/` (`probe_sweep.json`, `vocab_labels.json`, `p9_*`, `p10_*`,
`p11/`). See the File Map in `REPORT.md`.

`TODO: P2 confirm` — SAE repo id(s) to place under `saes/`, and how to obtain the
`prompts/cdg_injection/{A,B,C,D}` corpus (JSON is gitignored, not in the repo).

---

## `feat/p2-work-migration` — P2 exploratory studies (`p2/`)

Everything in `main`, plus self-contained studies under `p2/experement/` and
`p2/serverFiles/`. Each study has its own `README.md` / `REPORT.md` /
`RESULTS.md` / `SUMMARY.md` and its own scripts — **they are not wired into the
`main` wrappers.** See **`p2/README.md`** (the P2 run guide) for the per-study
detail. Real entry points:

| Study (`p2/experement/…` unless noted) | Run | Docs |
|----------------------------------------|-----|------|
| `harm_dir/` — harm-vs-safe direction (`v_harm`) detector + steering pilots | `sbatch scripts/steer_pilot2.sbatch` → `steer_pilot2.py --limit --benign-target --alphas --out` (also `capture*.sbatch`, `gen_safe.sbatch`) | `GATE_E_VERDICT.md`, `PHASE3_PILOT_VERDICT.md` |
| `study1/` — straddle / temp-sweep / yield | `bash run_straddle.sh` → `straddle_pilot.py` (also `run_expansion.sh`, `run_straddle_{c,f}.sh`) | `STUDY1_WRITEUP.md`, `SCHEMA.md` |
| `clockv2/`, `clockv2_refusal/`, `clock_attack/` — clock attack + refusal axis | `bash run_seed_clean.sh` → `capture_seed.py --cond --seed --limit` (also `run_seed_{benign,rest}.sh`) | `clockv2/README.md`, `REPORT.md` |
| `region_steer/` — region-steering confirm runs | `sbatch confirm40.sbatch` / `confirm40_tplmask.sbatch` | (batch logs in `batch*/`) |
| `prefill/`, `prefill_capture.py`, `baseline_capture.py` — prefill capture | `python prefill_capture.py …` `TODO: P2 confirm flags` | `CDG_prefilling_attack_report.md` |
| `p2/serverFiles/dijawithprefill/` — DIJA-with-prefill arms | `bash run_phase4.sh` → `run_arms.py` → `graded_judge.py` → `summarize.py` | `SUMMARY.md` |
| `p2/serverFiles/crossattack/` — cross-format transfer | `python corrected/run_transfer_corrected.py …` / `sanity/sanity_battery.py` `TODO: P2 confirm flags` | `SUMMARY.md`, `corrected/RESULTS.md` |
| `p2/serverFiles/attack2/` — attack-2 corpus + run | `python run_attack2_resumable.py …` `TODO: P2 confirm flags` | `RESULTS.md` |

All P2 GPU jobs use the Alliance venv `/scratch/ore99/cdg_venv/bin/python` and the
`def-zshakeri_gpu` account (see `*.sbatch` headers).

---

## `feat/p3-dream-extension` — Dream-v0-Instruct-7B extension (`dream/`)

Ports the injection analysis to the **Dream** diffusion LM: hidden-state
recording, injection probes, shared BA/CD steering direction, and p10/p11-style
graded judging. **See [`dream/README.md`](dream/README.md)** for the full result
tables. Runs on Alliance SLURM (`def-zshakeri`, H100 MIG; venv
`.../venv_llada`).

```bash
# Record Dream hidden-state history (400 examples, resumable):
sbatch dream/slurm/run_dream_history_full_h100_3g.slurm   # → dream_native_history_record_resume.py

# Steering runs (harm / out_mask directions at layer 14):
sbatch dream/slurm/run_dream_p10_outmask_L14_mid_h100_3g.slurm
sbatch dream/slurm/run_dream_p10_harm_f1_L14_mid_h100_3g.slurm
# smoke: dream/slurm/run_dream_p10_smoke_h100_3g.slurm

# Probes / directions / judging (run directly, see dream/README.md for args):
python dream/dream_hidden_probe_auc.py
python dream/dream_build_steering_direction_p10fmt.py
python dream/judge_dream_p10_p11style.py            # p11-style graded judge on saved B outputs
```

**Finding (from `dream/README.md`):** Dream has a clean shared injection direction
(out_mask, frac 0.05, layer 14; cos(BA,CD)=0.980), but direct residual steering
along it did **not** reduce attack success on the 30-case B subset — same
detection-not-correction asymmetry seen on LLaDA.

---

## `research/rrae-v2-release` — RRAE v2 (`experiments/rrae_v2/`)

**Rank-Reduction Auto-Encoder** pipeline: rebuilds a clean, balanced 2,000-example
dataset (health + non-health, buckets A/B/C/D × 500), extracts LLaDA hidden
states (layers 11 & 16), trains a low-rank RRAE (rank sweep 8–64), extracts
steering vectors, and evaluates additive steering / projection removal.
**See [`experiments/rrae_v2/README.md`](experiments/rrae_v2/README.md),
[`ARTIFACTS.md`](experiments/rrae_v2/ARTIFACTS.md), and
[`slurm/README.md`](experiments/rrae_v2/slurm/README.md).**

SLURM jobs are templates — set your paths first (from `slurm/README.md`), then
submit:

```bash
export RRAE_WORK_ROOT="/path/to/rrae_steering_work_v2"
export RRAE_ENV_ROOT="/path/to/python-environment"
export RRAE_DATA_ROOT="/path/to/rrae_data"
export LLADA_MODEL_PATH="/path/to/LLaDA-8B-Instruct"
export DIJA_ROOT="/path/to/DIJA"
export DEEPSEEK_API_KEY="your-key"        # for judge jobs; never store in the .sbatch
sbatch experiments/rrae_v2/slurm/<path>/<job>.sbatch
```

Pipeline stages (each a `scripts/<stage>/` directory):

1. **data/** — build/audit/finalize buckets, held-out sets (SALAD, MedQuAD,
   XSTest, CARES, EGIDA), blocklist, convert to CDG prompt roots.
2. **extraction/** — `extract_hidden_states_abcd_v2_input_region.py` (layers 11, 16).
3. **training/** — `train_rrae_v2.py` (rank sweep).
4. **vectors/** — `extract_rrae_v2_vectors.py` (raw diff / RRAE recon diff / residual diff).
5. **steering/** — `run_rrae_steering_generation_matrix_v2.py`,
   `run_projection_removal_generation_matrix_v2.py` (regions R0 = template-mask,
   R2 = template-mask + output-mask; alpha/lambda/rank/layer sweeps).
6. **judging/** — `evaluate_rrae_with_p11_judge.py`, `evaluate_success_judge_v2.py`
   (reads `DEEPSEEK_API_KEY`).
7. **diagnostics/ + analysis/** — train→held-out transfer, signal preservation,
   projection-removal effects.

**Finding (from README):** strong transferable injection direction (train→held-out
cos ≈ 0.90 at L11/L16), preserved through the RRAE (raw vs. recon vector cos
≈ 0.9999). Projection removal has measurable causal leverage (L11: harmful-injected
success ~84% → ~55%; benign-injected ~67% → ~62%) but strong settings (λ=2)
cause behavioral instability.

---

## Conventions across all branches

- **Never commit** weights, `*.pt`/`*.jsonl`/`*.npy` dumps, prompt/data corpora,
  or API keys — all gitignored. The DeepSeek key is read from `DEEPSEEK_API_KEY`
  only.
- GPU jobs run via SLURM on Alliance/Compute Canada; the DeepSeek judge needs
  login-node internet.
- `TODO: <owner> confirm` markers above flag run details (exact flags, dataset
  sources, SAE ids) not discoverable from the committed files — fill in from the
  owning teammate.
