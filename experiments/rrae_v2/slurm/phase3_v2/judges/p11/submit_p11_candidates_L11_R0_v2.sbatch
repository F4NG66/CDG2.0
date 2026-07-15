#!/usr/bin/env bash
#SBATCH --job-name=P11v2L11
#SBATCH --account=def-zshakeri
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-3%2
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd ${RRAE_WORK_ROOT:?Set RRAE_WORK_ROOT}

bash scripts/phase3_v2/judges/p11/run_p11_candidate_from_tsv.sh "$TSV"
