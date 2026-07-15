#!/usr/bin/env python

from pathlib import Path


WORK_ROOT = Path("/path/to/rrae_steering_work_v2")
ENV_PATH = Path("/path/to/llada-env")

GEN_SCRIPT = (
    WORK_ROOT
    / "scripts/phase3_v2/run_projection_removal_generation_matrix_v2.py"
)

PROMPT_ROOT = (
    WORK_ROOT
    / "data/phase3_prompt_roots/abcd_test_heldout_v1"
)

VECTOR_ROOT = (
    WORK_ROOT
    / "steering_vectors_v2/vector_source_ablation"
)

JOBS_ROOT = (
    WORK_ROOT
    / "jobs/phase3_v2/projection_removal_confirm"
)

LOG_ROOT = (
    WORK_ROOT
    / "logs/phase3_v2/projection_removal_confirm"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/projection_removal_confirm"
)

LAYERS = [11, 16]
RANK = 24
LAMBDAS = [1.0, 1.5, 2.0]


def find_raw_vector(layer: int) -> Path:
    candidates = sorted(
        path
        for path in VECTOR_ROOT.rglob("*.pt")
        if f"L{layer}" in str(path)
        and "raw" in str(path).lower()
    )

    if len(candidates) != 1:
        print(f"\nRaw-vector candidates for L{layer}:")
        for path in candidates:
            print(" ", path)

        raise RuntimeError(
            f"Expected exactly one raw vector for L{layer}, "
            f"found {len(candidates)}"
        )

    return candidates[0]


def main():
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = []

    for layer in LAYERS:
        vector_path = find_raw_vector(layer)

        run_name = f"L{layer}_r{RANK}_raw_projection_R0_N100"

        output_dir = OUT_ROOT / run_name
        job_path = JOBS_ROOT / f"{run_name}.sbatch"

        lambdas_text = " ".join(str(x) for x in LAMBDAS)

        text = f"""#!/usr/bin/env bash
#SBATCH --job-name=PR100_L{layer}
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --output={LOG_ROOT}/{run_name}_%j.out
#SBATCH --error={LOG_ROOT}/{run_name}_%j.err

set -euo pipefail

cd "{WORK_ROOT}"
source "{ENV_PATH}/bin/activate"

export PYTHONPATH="/path/to/rrae_data:${{PYTHONPATH:-}}"

echo "============================================================"
echo "Projection-removal confirmation"
echo "layer: L{layer}"
echo "rank: {RANK}"
echo "groups: B C"
echo "limit per group: 100"
echo "lambdas: {lambdas_text}"
echo "vector: {vector_path}"
echo "output: {output_dir}"
echo "start: $(date)"
echo "============================================================"

python "{GEN_SCRIPT}" \\
  --prompt-root "{PROMPT_ROOT}" \\
  --vector-path "{vector_path}" \\
  --out-dir "{output_dir}" \\
  --layer {layer} \\
  --rank {RANK} \\
  --position-spec template:mask \\
  --groups B C \\
  --limit-per-group 100 \\
  --alphas {lambdas_text} \\
  --seed 0 \\
  --num-shards 1 \\
  --shard-index 0 \\
  --device cuda

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
"""

        job_path.write_text(text, encoding="utf-8")
        jobs.append(job_path)

        print("Created:", job_path)
        print("Vector:", vector_path)
        print("Output:", output_dir)

    submit_path = (
        WORK_ROOT
        / "submit_projection_removal_confirm.sh"
    )

    with submit_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        handle.write(f'cd "{WORK_ROOT}"\n\n')

        for job in jobs:
            handle.write(f'sbatch "{job}"\n')

    submit_path.chmod(0o755)

    print("\nSubmit script:")
    print(submit_path)


if __name__ == "__main__":
    main()
