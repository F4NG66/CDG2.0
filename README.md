# ClinicalDenoiseGuard (CDG)

Mechanistic-interpretability study of **DIJA template-injection attacks** on the
masked-diffusion LLM **LLaDA-8B-Instruct**, and residual-stream **steering
defenses** against them. The pipeline records SAE features + hidden states from
the injected-template region during denoising, probes them for an
injection-mechanism signal, builds steering directions, and evaluates whether
steering reduces attack success (ASR) without breaking benign generation.

Full method + all results for the shared pipeline are in **[`REPORT.md`](REPORT.md)**
(Phases 1–12). This README is the run guide.

---

## Requirements

- **Python 3.11** (cluster wheels are pinned for 3.11; see `p2/experement/requirements.txt`).
- **PyTorch ≥ 2.1** and **transformers** — note the version split recorded in
  [`requirements.txt`](requirements.txt): LLaDA backends were validated on
  `transformers==4.38.2`, Dream backends on `4.46.2` (the P2 Alliance pin is
  `4.46.3+computecanada`). Pick the version matching the backend you run.
- Python deps: `torch`, `transformers`, `safetensors`, `numpy`, `requests`
  (DeepSeek judge), `huggingface_hub` (SAE download). Install via
  `pip install -r requirements.txt`.
- **GPU** for any real run (LLaDA-8B). CPU is only for the `--dummy` smoke path.

### Models & weights (not in the repo)

- **LLaDA-8B-Instruct** — `model_id = "GSAI-ML/LLaDA-8B-Instruct"`
  (`cdg/config.py`). Downloaded from HuggingFace into the standard HF cache
  (`$HF_HOME` / `~/.cache/huggingface`). The shell wrappers accept a
  `LLADA_CACHE` env var / hardcoded path pointing at a local HF snapshot for the
  tokenizer/model in the logit-lens and annotation steps.
- **SAEs** — expected under `./saes` (`--sae-root`), pulled via `huggingface_hub`.
  `TODO: P2 confirm` the exact SAE repo id(s) to place under `saes/` (DLM-Scope
  TopK-SAE, k=80, layers {11,16,26}, per `REPORT.md`).
- **Prompt corpus** — `prompts/cdg_injection/{A_harmful_clean, B_harmful_injected,
  C_neutral_injected, D_neutral_clean}`. **Not tracked** (JSON is gitignored).
  `TODO: P2 confirm` how teammates obtain the corpus (shared drive / generator).

### Secrets — env only, never committed

The DeepSeek judge reads its key from the environment
(`cdg/judge.py` → `os.environ.get("DEEPSEEK_API_KEY")`, POSTs to
`https://api.deepseek.com/chat/completions`). Provide it via an untracked `.env`
(the repo `.gitignore` excludes `.env`, `*.env`, `.env.*`) or `export` it:

```bash
# .env  (never commit this file)
DEEPSEEK_API_KEY=sk-...        # your key; do NOT paste a real key into git
```

Without a key, recording still runs but judging / LLM annotation are skipped.

---

## Setup

```bash
git clone git@github.com:F4NG66/CDG2.0.git
cd CDG2.0

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or the pinned p2/experement/requirements.txt on Alliance

export DEEPSEEK_API_KEY=sk-...          # or put it in an untracked .env
```

Large / generated data is **gitignored and not in the repo** — create or fetch it
locally: model weights → HF cache; SAEs → `./saes/`; prompt corpus →
`prompts/cdg_injection/`; all run outputs → `outputs/`, `outputs_*/`,
`analysis_output/` (see **Outputs**).

---

## How to run

All shell wrappers do `cd "$(dirname "$0")"` first, so run them from anywhere.
They default `PY` to a teammate's interpreter
(`/home/f4ng/cdg/bin/python3`) and `LLADA_CACHE` to a teammate's HF snapshot —
**override these for your machine.** `run_pipeline.sh` reads `PY`/`LLADA_CACHE`
from the environment; `run_p8_p6.sh` and `run_p9_p10.sh` hardcode `PY=` (and
`run_p8_p6.sh` hardcodes `LLADA_CACHE=`), so **edit those lines** before running
them.

### 0. Smoke test (CPU, no model / no key)

```bash
# Exercises the full record→analyze path with a tiny fake model:
bash run_pipeline.sh --dummy --limit 1
# Or just the recorder:
python run_record.py --prompt-root prompts/cdg_injection --dummy
```

### 1. End-to-end pipeline — record + analyze (`run_pipeline.sh`)

Runs, in order: Phase-1 recording (`run_record.py`, with token-level SAE) →
`p1_check` (QC) → `p2_assemble` (pseudobulk) → `s2_probe_sweep` → `p4_delta`
(quick + full) → `p5_modules` (co-activation clustering) → `p6_annotate`.

```bash
export DEEPSEEK_API_KEY=sk-...
PY=.venv/bin/python \
LLADA_CACHE=/path/to/hf/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/<snap> \
  bash run_pipeline.sh                 # full run, seed 0, all cases

# Useful flags (parsed by the script):
bash run_pipeline.sh --skip-record     # reuse existing outputs/
bash run_pipeline.sh --seeds "0 1 2"   # multiple seeds
bash run_pipeline.sh --limit 30        # cases per group
bash run_pipeline.sh --backend llada_attack   # {llada,llada_attack,dream,dream_attack}
```

### 2. Logit-lens vocab labels + re-annotate (`run_p8_p6.sh`)

Loads LLaDA once, computes logit-lens vocab labels for all active SAE features
(`p8_vocab`), then re-runs `p6_annotate` with full coverage.
**Edit the `PY=` and `LLADA_CACHE=` lines at the top first.**

```bash
export DEEPSEEK_API_KEY=sk-...
bash run_p8_p6.sh
```

### 3. Corrected probe + residual-stream steering (`run_p9_p10.sh`)

`p9_diagnose` (TF-IDF baseline, CV check, B-vs-A / C-vs-D injection probe sweep)
then `p10_steer2` (two steering configs: harm-position and template/mask-position).
**Edit the `PY=` line at the top first.**

```bash
bash run_p9_p10.sh              # default: --limit 30
bash run_p9_p10.sh --full      # all cases (production)
bash run_p9_p10.sh --p9-only   # diagnostics only
bash run_p9_p10.sh --p10-only  # steering only (needs p9 output)
```

### 4. Ablation & direction stability (run manually — no wrapper)

```bash
# Scope/position ablation (baseline/A1/A2/A3, groups B+C):
python scripts/p11_ablation.py     # TODO: P2 confirm exact flags (config, alpha, limit)

# Cross-frac direction stability (pending run, per REPORT.md):
python scripts/p12_dir_stability.py --out-dir outputs --scope harm --layers 11 16 26
```

### 5. Alternative steering-eval driver (`run_steer_eval.py`)

Builds a steering vector from Phase-1 records, re-runs steered, reports ASR drop
+ capability retention:

```bash
export DEEPSEEK_API_KEY=sk-...
python run_steer_eval.py --records outputs --backend llada_attack \
    --sae-root ./saes --prompt-root prompts/cdg_injection \
    --alpha -8 --layers 16 --judge --out steer_outputs
```

### 6. Correlation heatmap (`plot_corr.py`)

```bash
python plot_corr.py    # reads outputs/analysis/modules/corr_matrix.npy → *_raw_heatmap.png
```

### P2 sub-studies (`p2/experement/`, `p2/serverFiles/`)

Self-contained exploratory studies, each with its own scripts and a local
`README.md` / `REPORT.md` / `RESULTS.md`. They are **not wired into the wrappers
above** — read the study's own doc for its run command:

- `p2/experement/harm_dir/` — harm-vs-safe direction (`v_harm`) detector + steering
  pilots (`GATE_E_VERDICT.md`, `PHASE3_PILOT_VERDICT.md`; sbatch in `scripts/`).
- `p2/experement/region_steer/` — region-steering confirm runs (`confirm40*.sbatch`).
- `p2/experement/clockv2/`, `clockv2_refusal/`, `clock_attack/` — clock attack +
  refusal-axis studies (`run_seed_*.sh`).
- `p2/experement/study1/` — straddle / temp-sweep / yield studies (`run_*.sh`, `SCHEMA.md`).
- `p2/experement/prefill/`, `prefill_capture.py`, `baseline_capture.py` — prefill capture.
- `p2/serverFiles/crossattack/`, `attack2/`, `dijawithprefill/` — cross-format &
  DIJA-with-prefill attacks (see each `SUMMARY.md` / `run_phase4.sh`).

`TODO: P2 confirm` a canonical entry order for the sub-studies if one is needed.

---

## Outputs

All output dirs are **gitignored** (not committed). Produced by a run (see the
File Map in `REPORT.md`):

```
outputs/
  manifest.jsonl                 Phase-1 records: gen metadata + judge verdicts
  <case>__seed0.pt               per-generation record (hidden + SAE + entropy)
  assembled/                     X_*_L16_f0p10.npy pseudobulk matrices
  analysis/
    delta/                       Δ vectors, DE tables, cosine atlas (delta_atlas.png)
    modules/                     feature_module.csv, corr_matrix.npy, corr_heatmap.png
    annotation/                  module_labels.csv, per-module trace plots

outputs_feat_steer*/ , outputs_p11/   Phase-7 / Phase-11 steering records

analysis_output/
  probe_sweep.json               p3: B-vs-C probe grid
  vocab_labels.json              p8: logit-lens labels
  p9_results.json , p9_probe_inj_sweep.json , p9_probe_inj_heatmap_*.png
  p10_*_directions/ , p10_*_dose_response*.json/png
  p11/  (baseline/A1/A2/A3 json + _summary.json)
```

---

## Cluster notes

GPU work runs on **Alliance / Compute Canada** HPC (SLURM), not on login nodes.
SLURM job files live in `dream/slurm/*.slurm` and under
`p2/experement/**/*.sbatch` (e.g. `harm_dir/scripts/*.sbatch`,
`region_steer/*.sbatch`). A representative header:

```bash
#SBATCH --account=def-zshakeri_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1   # H100 MIG slice
#SBATCH --cpus-per-task=6  --mem=32G  --time=03:00:00
module load StdEnv/2023
PY=/scratch/ore99/cdg_venv/bin/python                # venv on /scratch, not /home
```

- Build the venv from the pinned Alliance wheels (`--no-index` wheelhouse):
  `p2/experement/requirements.txt` (torch 2.6.0, transformers 4.46.3, all
  `+computecanada`).
- **Internet is only available on login nodes.** Pre-download HF weights/SAEs and
  set `DEEPSEEK_API_KEY` on the login node; compute nodes have no outbound
  network, so the DeepSeek judge must run where there is internet (login node or
  a separate judging pass), not inside a GPU job.
- Keep large caches/venvs on `/scratch` — `/home` has a tight quota.

---

## Branches

| Branch | Contents | Entry point |
|--------|----------|-------------|
| `main` | Shared CDG pipeline, Phases 1–12: recording, SAE probes, Δ/DE analysis, co-activation modules, residual-stream steering. | `run_pipeline.sh` → `run_p8_p6.sh` → `run_p9_p10.sh` |
| `feat/p2-work-migration` *(this branch)* | Everything in `main`, plus P2's exploratory studies under `p2/`: harm-direction detector (`harm_dir`), region steering, clock / refusal-axis, prefill capture, cross-format & DIJA-with-prefill attacks. | `p2/experement/*/` (per-study scripts + docs) |
| `feat/p3-dream-extension` | Dream-model extension: adds `dream/judge_dream_p10_p11style.py` + `dream/README.md` (p10/p11-style judging for the Dream diffusion LM). | `dream/` |
| `research/rrae-v2-release` | RRAE v2 release under `experiments/rrae_v2/`. | `experiments/rrae_v2/` |
