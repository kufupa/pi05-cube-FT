#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s11
#SBATCH --partition=gpucluster3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage11_orch_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

bash "/vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh" stage11_slurm_orchestration

