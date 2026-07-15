# SLURM Environment Setup

The included SLURM jobs are cluster templates. Before submitting a job, define
the paths used by your local installation.

Example:

    export RRAE_WORK_ROOT="/path/to/rrae_steering_work_v2"
    export RRAE_ENV_ROOT="/path/to/python-environment"
    export RRAE_DATA_ROOT="/path/to/rrae_data"
    export LLADA_MODEL_PATH="/path/to/LLaDA-8B-Instruct"
    export DIJA_ROOT="/path/to/DIJA"

For jobs using the DeepSeek judge:

    export DEEPSEEK_API_KEY="your-key"

Then submit the required job:

    sbatch experiments/rrae_v2/slurm/path/to/job.sbatch

API keys and credentials must never be stored directly in SLURM files or
committed to Git.
