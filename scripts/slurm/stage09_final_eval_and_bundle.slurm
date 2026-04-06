#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s09
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage09_report_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
PIPE="${REPO_ROOT}/scripts/smolvla_vggflow/run_stage.sh"
bash "${PIPE}" phase13_post_train_eval
bash "${PIPE}" phase12_reporting
