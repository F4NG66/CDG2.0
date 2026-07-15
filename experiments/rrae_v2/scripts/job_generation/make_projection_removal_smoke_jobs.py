#!/usr/bin/env python

from pathlib import Path


WORK_ROOT = Path(
    "/path/to/rrae_steering_work_v2"
)

ENV_PATH = Path(
    "/path/to/llada-env"
)

GEN_SCRIPT = (
    WORK_ROOT
    / "scripts/phase3_v2/"
    "run_projection_removal_generation_matrix_v2.py"
)

PROMPT_ROOT = (
    WORK_ROOT
    / "data/phase3_prompt_roots/"
    "abcd_test_heldout_v1"
)

VECTOR_ROOT = (
    WORK_ROOT
    / "steering_vectors_v2/"
    "vector_source_ablation"
)

JOBS_ROOT = (
    WORK_ROOT
    / "jobs/phase3_v2/"
    "projection_removal_smoke"
)

LOG_ROOT = (
    WORK_ROOT
    / "logs/phase3_v2/"
    "projection_removal_smoke"
)

OUT_ROOT = (
    WORK_ROOT
    / "steering_runs/phase3_v2/"
    "projection_removal_smoke"
)

LAYERS = [11, 16]
RANK = 24

# داخل سكربت التوليد اسمها alpha،
# لكن في هذه التجربة تمثل lambda.
LAMBDAS = [0.5, 1.0, 1.5]


def find_raw_vector(layer: int) -> Path:
    patterns = [
        f"*L{layer}*raw*.pt",
        f"*raw*L{layer}*.pt",
        f"*L{layer}*.pt",
    ]

    candidates = []

    for pattern in patterns:
        found = sorted(VECTOR_ROOT.rglob(pattern))

        # نفضل فقط الملفات التي يحتوي اسمها أو مسارها على raw
        raw_found = [
            path
            for path in found
            if "raw" in str(path).lower()
        ]

        if raw_found:
            candidates = raw_found
            break

    # إزالة التكرار مع الحفاظ على الترتيب
    unique = []
    seen = set()

    for path in candidates:
        resolved = path.resolve()

        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)

    if len(unique) != 1:
        print(f"\nL{layer} raw-vector candidates: {len(unique)}")

        for path in unique:
            print(" ", path)

        raise RuntimeError(
            f"Expected exactly one raw vector for L{layer}. "
            "Inspect steering_vectors_v2/vector_source_ablation."
        )

    return unique[0]


def make_job(layer: int, vector_path: Path) -> Path:
    run_name = f"L{layer}_r{RANK}_raw_projection_R0"

    out_dir = OUT_ROOT / run_name

    job_path = (
        JOBS_ROOT
        / f"{run_name}.sbatch"
    )

    lambda_args = " ".join(
        str(value)
        for value in LAMBDAS
    )

    text = f"""#!/usr/bin/env bash
#SBATCH --job-name=PR_L{layer}r{RANK}
#SBATCH --account=def-zshakeri
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --output={LOG_ROOT}/{run_name}_%j.out
#SBATCH --error={LOG_ROOT}/{run_name}_%j.err

set -euo pipefail

cd "{WORK_ROOT}"
source "{ENV_PATH}/bin/activate"

export PYTHONPATH="/path/to/rrae_data:${{PYTHONPATH:-}}"

echo "============================================================"
echo "Projection-removal smoke"
echo "layer: L{layer}"
echo "rank label: {RANK}"
echo "vector source: raw"
echo "position: template:mask"
echo "lambdas: {lambda_args}"
echo "limit per group: 20"
echo "groups: A B C D"
echo "vector: {vector_path}"
echo "output: {out_dir}"
echo "host: $(hostname)"
echo "start: $(date)"
echo "============================================================"

python "{GEN_SCRIPT}" \\
  --prompt-root "{PROMPT_ROOT}" \\
  --vector-path "{vector_path}" \\
  --out-dir "{out_dir}" \\
  --position-spec "template:mask" \\
  --layer {layer} \\
  --rank {RANK} \\
  --groups A B C D \\
  --limit-per-group 20 \\
  --alphas {lambda_args} \\
  --seed 0 \\
  --num-shards 1 \\
  --shard-index 0 \\
  --device cuda

echo "============================================================"
echo "DONE"
echo "finish: $(date)"
echo "============================================================"
"""

    job_path.write_text(
        text,
        encoding="utf-8",
    )

    return job_path


def main():
    JOBS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not GEN_SCRIPT.exists():
        raise FileNotFoundError(
            f"Generation script missing: {GEN_SCRIPT}"
        )

    if not PROMPT_ROOT.exists():
        raise FileNotFoundError(
            f"Prompt root missing: {PROMPT_ROOT}"
        )

    if not VECTOR_ROOT.exists():
        raise FileNotFoundError(
            f"Vector root missing: {VECTOR_ROOT}"
        )

    jobs = []

    for layer in LAYERS:
        vector_path = find_raw_vector(layer)
        job_path = make_job(layer, vector_path)

        jobs.append(job_path)

        print("\n" + "=" * 80)
        print(f"L{layer}")
        print("vector:", vector_path)
        print("job:", job_path)

    submit_path = (
        WORK_ROOT
        / "submit_projection_removal_smoke.sh"
    )

    with submit_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        handle.write(f'cd "{WORK_ROOT}"\n\n')

        for job in jobs:
            handle.write(
                f'sbatch "{job}"\n'
            )

    submit_path.chmod(0o755)

    print("\n" + "=" * 80)
    print("Submit script:")
    print(submit_path)


if __name__ == "__main__":
    main()
