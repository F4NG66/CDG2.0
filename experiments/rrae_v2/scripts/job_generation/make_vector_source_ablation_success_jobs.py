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
    / "steering_runs/phase3_v2/vector_source_ablation"
)

JOBS_ROOT = (
    WORK_ROOT
    / "jobs/phase3_v2/judges/vector_source_ablation"
)

LOG_ROOT = (
    WORK_ROOT
    / "logs/phase3_v2/judges/vector_source_ablation"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/judges/vector_source_ablation"
)

LAYERS = [11, 16]

SOURCES = {
    "raw": "raw",
    "recon": "recon",
    "resid": "resid",
}

RANK = 24
POSITION_DIR = "R0_template_mask"
RESULT_FILENAME = (
    "results__custom_template_mask"
    "__rank24__L{layer}.jsonl"
)


def main():
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not JUDGE_SCRIPT.exists():
        raise FileNotFoundError(JUDGE_SCRIPT)

    created = []

    for layer in LAYERS:
        for source_name, run_suffix in SOURCES.items():
            input_path = (
                GEN_ROOT
                / f"L{layer}_r{RANK}_{run_suffix}_R0"
                / POSITION_DIR
                / RESULT_FILENAME.format(layer=layer)
            )

            if not input_path.exists():
                raise FileNotFoundError(input_path)

            run_name = (
                f"L{layer}_r{RANK}_{source_name}_R0"
            )

            job_name = (
                f"JVS{layer}_{source_name}"
            )

            out_dir = (
                OUT_ROOT
                / run_name
            )

            out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            sbatch_path = (
                JOBS_ROOT
                / f"success_{run_name}.sbatch"
            )

            text = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
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

INPUT="{input_path}"
OUT_DIR="{out_dir}"

echo "============================================================"
echo "Vector-source success judge"
echo "layer: L{layer}"
echo "rank: {RANK}"
echo "source: {source_name}"
echo "configuration: custom_template_mask"
echo "groups: B C"
echo "input: $INPUT"
echo "output: $OUT_DIR"
echo "model: deepseek-v4-flash"
echo "host: $(hostname)"
echo "start: $(date)"
echo "============================================================"

python "{JUDGE_SCRIPT}" \\
  --input "$INPUT" \\
  --out-dir "$OUT_DIR" \\
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

            sbatch_path.write_text(
                text,
                encoding="utf-8",
            )

            created.append(sbatch_path)

    submit_path = (
        WORK_ROOT
        / "submit_vector_source_ablation_judges_L11.sh"
    )

    with submit_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        handle.write(f'cd "{WORK_ROOT}"\n\n')

        for job in created:
            handle.write(f'sbatch "{job}"\n')

    submit_path.chmod(0o755)

    print("Created jobs:")

    for job in created:
        print(" ", job)

    print("\nSubmit script:")
    print(" ", submit_path)


if __name__ == "__main__":
    main()
