#!/usr/bin/env python

from pathlib import Path


WORK_ROOT = Path("/path/to/rrae_steering_work_v2")
MODEL_PATH = Path("/path/to/LLaDA-8B-Instruct")
ENV_PATH = Path("/path/to/llada-env")

PROMPT_ROOT = (
    WORK_ROOT
    / "data/phase3_prompt_roots/abcd_v2_in_sample_full"
)

VECTOR_PATH = (
    WORK_ROOT
    / "steering_vectors_v2"
    / "rrae_v_injection__scope-input_region"
      "__layer-L16"
      "__rank-r48"
      "__source-residual_BA_plus_CD"
      "__abcd_v2_clean_N2000.pt"
)

GEN_SCRIPT = (
    WORK_ROOT
    / "scripts/phase3_v2/run_rrae_steering_generation_matrix_v2.py"
)

JOBS_ROOT = WORK_ROOT / "jobs/phase3_v2/smoke"
LOG_ROOT = WORK_ROOT / "logs/phase3_v2/smoke"
OUT_ROOT = WORK_ROOT / "steering_runs/phase3_v2/smoke"

CONFIGS = [
    ("R0_template_mask", "template:mask"),
    (
        "R2_template_mask_plus_output_mask",
        "template:mask+output:mask",
    ),
]


def main():
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not PROMPT_ROOT.exists():
        raise FileNotFoundError(PROMPT_ROOT)

    if not VECTOR_PATH.exists():
        raise FileNotFoundError(VECTOR_PATH)

    created = []

    for config_name, position_spec in CONFIGS:
        out_dir = OUT_ROOT / "L16_rank48" / config_name
        out_dir.mkdir(parents=True, exist_ok=True)

        job_name = (
            "P3v2_smoke_R0"
            if config_name.startswith("R0")
            else "P3v2_smoke_R2"
        )

        sbatch_path = JOBS_ROOT / f"{job_name}.sbatch"

        text = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=def-zshakeri
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output={LOG_ROOT}/{job_name}_%j.out
#SBATCH --error={LOG_ROOT}/{job_name}_%j.err

set -euo pipefail

WORK_ROOT="{WORK_ROOT}"
ENV_PATH="{ENV_PATH}"

cd "$WORK_ROOT"
source "$ENV_PATH/bin/activate"

export PYTHONPATH="/path/to/rrae_data:${{PYTHONPATH:-}}"

echo "============================================================"
echo "Phase 3 v2 smoke generation"
echo "config: {config_name}"
echo "position_spec: {position_spec}"
echo "vector: L16 rank48"
echo "groups: A B C D"
echo "limit_per_group: 2"
echo "steered alpha: -1.0"
echo "baseline: generated automatically"
echo "host: $(hostname)"
echo "time: $(date)"
echo "============================================================"

python "{GEN_SCRIPT}" \\
  --prompt-root "{PROMPT_ROOT}" \\
  --model-path "{MODEL_PATH}" \\
  --vector-path "{VECTOR_PATH}" \\
  --out-dir "{out_dir}" \\
  --layer 16 \\
  --rank 48 \\
  --position-spec "{position_spec}" \\
  --groups A B C D \\
  --limit-per-group 2 \\
  --alphas -1.0 \\
  --seed 0 \\
  --num-shards 1 \\
  --shard-index 0 \\
  --device cuda

echo "============================================================"
echo "DONE"
echo "time: $(date)"
echo "============================================================"
"""

        sbatch_path.write_text(text)
        created.append(sbatch_path)

    submit_path = WORK_ROOT / "submit_phase3_v2_smoke.sh"

    with submit_path.open("w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(f'cd "{WORK_ROOT}"\n\n')

        for path in created:
            f.write(f'sbatch "{path}"\n')

    submit_path.chmod(0o755)

    print("Created jobs:")
    for path in created:
        print(" ", path)

    print("\nSubmit script:")
    print(" ", submit_path)


if __name__ == "__main__":
    main()
