#!/usr/bin/env python

from pathlib import Path


WORK_ROOT = Path("/path/to/rrae_steering_work_v2")
ENV_PATH = Path("/path/to/llada-env")
MODEL_PATH = Path("/path/to/LLaDA-8B-Instruct")

PROMPT_ROOT = (
    WORK_ROOT
    / "data/phase3_prompt_roots/abcd_test_heldout_v1"
)

VECTOR_ROOT = (
    WORK_ROOT
    / "steering_vectors_v2/vector_source_ablation"
)

GEN_SCRIPT = (
    WORK_ROOT
    / "scripts/phase3_v2/"
      "run_rrae_steering_generation_matrix_v2.py"
)

JOBS_ROOT = (
    WORK_ROOT
    / "jobs/phase3_v2/vector_source_ablation"
)

LOG_ROOT = (
    WORK_ROOT
    / "logs/phase3_v2/vector_source_ablation"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/vector_source_ablation"
)

LAYERS = [11, 16]

SOURCES = [
    "raw_BA_plus_CD",
    "reconstruction_BA_plus_CD",
    "residual_BA_plus_CD",
]

RANK = 24
LIMIT_PER_GROUP = 20
ALPHAS = [-1.0, -2.0, -4.0, -8.0]

POSITION_NAME = "R0_template_mask"
POSITION_SPEC = "template:mask"


def vector_path(layer: int, source: str) -> Path:
    return (
        VECTOR_ROOT
        / (
            "rrae_vector_ablation"
            f"__source-{source}"
            f"__layer-L{layer}"
            f"__rank-r{RANK}"
            "__train-v2.pt"
        )
    )


def short_source(source: str) -> str:
    mapping = {
        "raw_BA_plus_CD": "raw",
        "reconstruction_BA_plus_CD": "recon",
        "residual_BA_plus_CD": "resid",
    }
    return mapping[source]


def main():
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    required = [
        PROMPT_ROOT,
        GEN_SCRIPT,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    jobs = []

    for layer in LAYERS:
        for source in SOURCES:
            vec_path = vector_path(layer, source)

            if not vec_path.exists():
                raise FileNotFoundError(vec_path)

            source_short = short_source(source)

            run_name = (
                f"L{layer}_r{RANK}_{source_short}_R0"
            )

            job_name = (
                f"VS_{layer}_{source_short}"
            )

            out_dir = (
                OUT_ROOT
                / run_name
                / POSITION_NAME
            )

            out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            sbatch_path = (
                JOBS_ROOT
                / f"{run_name}.sbatch"
            )

            alpha_text = " ".join(
                str(alpha)
                for alpha in ALPHAS
            )

            text = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=def-zshakeri
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output={LOG_ROOT}/{run_name}_%j.out
#SBATCH --error={LOG_ROOT}/{run_name}_%j.err

set -euo pipefail

WORK_ROOT="{WORK_ROOT}"
ENV_PATH="{ENV_PATH}"

cd "$WORK_ROOT"
source "$ENV_PATH/bin/activate"

export PYTHONPATH="/path/to/rrae_data:${{PYTHONPATH:-}}"

echo "============================================================"
echo "Vector-source steering ablation"
echo "run: {run_name}"
echo "dataset: held-out v1"
echo "layer: L{layer}"
echo "rank label: {RANK}"
echo "vector source: {source}"
echo "position: {POSITION_SPEC}"
echo "groups: A B C D"
echo "limit per group: {LIMIT_PER_GROUP}"
echo "alphas: {alpha_text}"
echo "seed: 0"
echo "host: $(hostname)"
echo "start: $(date)"
echo "============================================================"

python "{GEN_SCRIPT}" \\
  --prompt-root "{PROMPT_ROOT}" \\
  --model-path "{MODEL_PATH}" \\
  --vector-path "{vec_path}" \\
  --out-dir "{out_dir}" \\
  --layer {layer} \\
  --rank {RANK} \\
  --position-spec "{POSITION_SPEC}" \\
  --groups A B C D \\
  --limit-per-group {LIMIT_PER_GROUP} \\
  --alphas {alpha_text} \\
  --seed 0 \\
  --num-shards 1 \\
  --shard-index 0 \\
  --device cuda

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
"""

            sbatch_path.write_text(
                text,
                encoding="utf-8",
            )

            jobs.append(sbatch_path)

    submit_all = (
        WORK_ROOT
        / "submit_vector_source_ablation_all.sh"
    )

    with submit_all.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        handle.write(f'cd "{WORK_ROOT}"\n\n')

        for job in jobs:
            handle.write(f'sbatch "{job}"\n')

    submit_all.chmod(0o755)

    # Separate submission scripts allow one layer at a time.
    for layer in LAYERS:
        submit_layer = (
            WORK_ROOT
            / f"submit_vector_source_ablation_L{layer}.sh"
        )

        with submit_layer.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("#!/usr/bin/env bash\n")
            handle.write("set -euo pipefail\n\n")
            handle.write(f'cd "{WORK_ROOT}"\n\n')

            for job in jobs:
                if f"L{layer}_r{RANK}_" in job.name:
                    handle.write(f'sbatch "{job}"\n')

        submit_layer.chmod(0o755)

    print("Created jobs:")

    for job in jobs:
        print(" ", job)

    print("\nSubmit all:")
    print(" ", submit_all)

    for layer in LAYERS:
        print(
            f"Submit L{layer}:",
            WORK_ROOT
            / f"submit_vector_source_ablation_L{layer}.sh",
        )


if __name__ == "__main__":
    main()
