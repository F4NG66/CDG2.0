#!/usr/bin/env python

from pathlib import Path


WORK_ROOT = Path(
    "/path/to/rrae_steering_work_v2"
)

JOB_ROOT = (
    WORK_ROOT
    / "jobs/phase3_v2/judges/"
    "p11_best_L11_lambda2"
)

LOG_ROOT = (
    WORK_ROOT
    / "logs/phase3_v2/judges/"
    "p11_best_L11_lambda2"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/judges/"
    "p11_best_L11_lambda2"
)

B_INPUT = (
    WORK_ROOT
    / "data/phase3_judge_inputs/"
    "p11_best_L11_lambda2/"
    "B_baseline_plus_L11_lambda2_N200.jsonl"
)

C_INPUT = (
    WORK_ROOT
    / "data/phase3_judge_inputs/"
    "p11_best_L11_lambda2/"
    "C_baseline_plus_L11_lambda2_N200.jsonl"
)

P11_SCRIPT = (
    WORK_ROOT
    / "scripts/phase3_v2/judges/p11/"
    "evaluate_rrae_with_p11_judge.py"
)

C_SCRIPT = (
    WORK_ROOT
    / "scripts/phase3_v2/judges/"
    "benign_utility/"
    "evaluate_benign_utility_collapse_judge.py"
)


def api_export(task, pass_id):
    key_map = {
        ("B", 1): 1,
        ("B", 2): 2,
        ("B", 3): 1,
        ("C", 1): 2,
        ("C", 2): 1,
        ("C", 3): 2,
    }

    key_number = key_map[(task, pass_id)]

    return (
        f'export DEEPSEEK_API_KEY='
        f'"$DEEPSEEK_API_KEY_{key_number}"'
    )


def make_job(task, pass_id):
    if task == "B":
        input_path = B_INPUT
        script = P11_SCRIPT
        output_dir = OUT_ROOT / f"B_p11_pass{pass_id}"

        command = f"""python "{script}" \\
  --input "{input_path}" \\
  --out-dir "{output_dir}" \\
  --groups B \\
  --configs custom_template_mask \\
  --sample-per-bucket 100 \\
  --judge-model deepseek-v4-flash"""

    elif task == "C":
        input_path = C_INPUT
        script = C_SCRIPT
        output_dir = OUT_ROOT / f"C_utility_pass{pass_id}"

        command = f"""python "{script}" \\
  --input "{input_path}" \\
  --out-dir "{output_dir}" \\
  --configs custom_template_mask \\
  --sample-per-bucket 100 \\
  --judge-model deepseek-v4-flash"""

    else:
        raise ValueError(task)

    job_name = f"{task}L11l2p{pass_id}"
    job_path = JOB_ROOT / f"{task}_pass{pass_id}.sbatch"

    text = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output={LOG_ROOT}/{task}_pass{pass_id}_%j.out

set -euo pipefail

cd "{WORK_ROOT}"
source /path/to/llada-env/bin/activate

if [ -z "${{DEEPSEEK_API_KEY_1:-}}" ] || [ -z "${{DEEPSEEK_API_KEY_2:-}}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY_1 or DEEPSEEK_API_KEY_2 is not set"
  exit 1
fi

{api_export(task, pass_id)}

echo "========================================"
echo "TASK={task}"
echo "PASS={pass_id}"
echo "INPUT={input_path}"
echo "OUT={output_dir}"
echo "MODEL=deepseek-v4-flash"
echo "START=$(date)"
echo "========================================"

mkdir -p "{output_dir}"

{command}

echo "========================================"
echo "DONE"
echo "FINISH=$(date)"
echo "========================================"
"""

    job_path.write_text(text)
    print(job_path)


def main():
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = []

    for task in ["B", "C"]:
        for pass_id in [1, 2, 3]:
            make_job(task, pass_id)
            jobs.append(
                JOB_ROOT / f"{task}_pass{pass_id}.sbatch"
            )

    submit_path = (
        WORK_ROOT
        / "submit_p11_best_L11_lambda2.sh"
    )

    submit_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]

    for job in jobs:
        submit_lines.append(f'sbatch "{job}"')

    submit_path.write_text(
        "\n".join(submit_lines) + "\n"
    )
    submit_path.chmod(0o755)

    print("\nsubmit script:")
    print(submit_path)


if __name__ == "__main__":
    main()
