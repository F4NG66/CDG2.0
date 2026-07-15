#!/usr/bin/env python

from pathlib import Path


WORK_ROOT = Path("/path/to/rrae_steering_work_v2")
ENV_PATH = Path("/path/to/llada-env")

JUDGE_SCRIPT = (
    WORK_ROOT
    / "scripts/phase3_v2/judges/evaluate_success_judge_v2.py"
)

GEN_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/projection_removal_smoke"
)

JOBS_ROOT = (
    WORK_ROOT
    / "jobs/phase3_v2/judges/projection_removal_smoke"
)

LOG_ROOT = (
    WORK_ROOT
    / "logs/phase3_v2/judges/projection_removal_smoke"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/judges/projection_removal_smoke"
)

LAYERS = [11, 16]
RANK = 24


def main():
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = []

    for layer in LAYERS:
        run_name = f"L{layer}_r{RANK}_raw_projection_R0"

        gen_dir = GEN_ROOT / run_name

        files = list(
            gen_dir.rglob("results__*.jsonl")
        )

        if len(files) != 1:
            print(f"L{layer} result candidates:", len(files))
            for path in files:
                print(" ", path)

            raise RuntimeError(
                f"Expected exactly one results file for L{layer}"
            )

        input_path = files[0]
        out_dir = OUT_ROOT / run_name

        job_path = (
            JOBS_ROOT
            / f"success_{run_name}.sbatch"
        )

        text = f"""#!/usr/bin/env bash
#SBATCH --job-name=JPR_L{layer}
#SBATCH --account=def-zshakeri
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output={LOG_ROOT}/{run_name}_%j.out
#SBATCH --error={LOG_ROOT}/{run_name}_%j.err

set -euo pipefail

cd "{WORK_ROOT}"
source "{ENV_PATH}/bin/activate"

export PYTHONPATH="/path/to/rrae_data:${{PYTHONPATH:-}}"

if [[ -z "${{DEEPSEEK_API_KEY:-}}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set"
    exit 1
fi

echo "============================================================"
echo "Projection-removal success judge"
echo "layer: L{layer}"
echo "rank: {RANK}"
echo "groups: B C"
echo "input: {input_path}"
echo "output: {out_dir}"
echo "start: $(date)"
echo "============================================================"

python "{JUDGE_SCRIPT}" \\
  --input "{input_path}" \\
  --out-dir "{out_dir}" \\
  --configs custom_template_mask \\
  --groups B C \\
  --judge-template injection \\
  --judge-model deepseek-v4-flash \\
  --sample-per-bucket 0

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
"""

        job_path.write_text(
            text,
            encoding="utf-8",
        )

        jobs.append(job_path)

        print("Created:", job_path)
        print("Input:", input_path)

    submit_path = (
        WORK_ROOT
        / "submit_projection_removal_judges.sh"
    )

    with submit_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
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
