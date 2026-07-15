#!/usr/bin/env python

from pathlib import Path


WORK_ROOT = Path("/path/to/rrae_steering_work_v2")
ENV_PATH = Path("/path/to/llada-env")

JUDGE_SCRIPT = (
    WORK_ROOT
    / "scripts/phase3_v2/judges/evaluate_success_judge_v2.py"
)

INPUT_ROOT = (
    WORK_ROOT
    / "data/phase3_judge_inputs/projection_removal_confirm"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/judges/"
    "projection_removal_confirm_majority"
)

JOB_ROOT = (
    WORK_ROOT
    / "jobs/phase3_v2/judges/"
    "projection_removal_confirm_majority"
)

LOG_ROOT = (
    WORK_ROOT
    / "logs/phase3_v2/judges/"
    "projection_removal_confirm_majority"
)

TARGETS = [
    ("baseline", INPUT_ROOT / "baseline_canonical_N200.jsonl"),
    ("L11", INPUT_ROOT / "L11_steered_N600.jsonl"),
    ("L16", INPUT_ROOT / "L16_steered_N600.jsonl"),
]

PASSES = [1, 2, 3]


def main():
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = []
    job_index = 0

    for target, input_path in TARGETS:
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        for judge_pass in PASSES:
            job_index += 1

            key_number = 1 if job_index % 2 == 1 else 2
            key_variable = f"DEEPSEEK_API_KEY_{key_number}"

            run_name = f"{target}_pass{judge_pass}"
            out_dir = OUT_ROOT / run_name
            job_path = JOB_ROOT / f"{run_name}.sbatch"

            text = f"""#!/usr/bin/env bash
#SBATCH --job-name=JPR_{target}_p{judge_pass}
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output={LOG_ROOT}/{run_name}_%j.out
#SBATCH --error={LOG_ROOT}/{run_name}_%j.err

set -euo pipefail

cd "{WORK_ROOT}"
source "{ENV_PATH}/bin/activate"

export PYTHONPATH="/path/to/rrae_data:${{PYTHONPATH:-}}"

if [[ -z "${{{key_variable}:-}}" ]]; then
    echo "ERROR: {key_variable} is not available"
    exit 1
fi

export DEEPSEEK_API_KEY="${{{key_variable}}}"

echo "============================================================"
echo "target: {target}"
echo "judge pass: {judge_pass}"
echo "API key slot: {key_number}"
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

            job_path.write_text(text, encoding="utf-8")
            jobs.append(job_path)

            print(
                f"Created: {job_path.name} "
                f"using {key_variable}"
            )

    submit_path = (
        WORK_ROOT
        / "submit_projection_confirm_majority_judges.sh"
    )

    with submit_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")

        handle.write(
            'if [[ -z "${DEEPSEEK_API_KEY_1:-}" '
            '|| -z "${DEEPSEEK_API_KEY_2:-}" ]]; then\n'
        )
        handle.write(
            '  echo "Both API keys must be exported."\n'
        )
        handle.write("  exit 1\n")
        handle.write("fi\n\n")

        for job in jobs:
            handle.write(
                "sbatch "
                "--export=ALL,"
                "DEEPSEEK_API_KEY_1,"
                "DEEPSEEK_API_KEY_2 "
                f'"{job}"\n'
            )

    submit_path.chmod(0o755)

    print("\nSubmit script:")
    print(submit_path)


if __name__ == "__main__":
    main()
