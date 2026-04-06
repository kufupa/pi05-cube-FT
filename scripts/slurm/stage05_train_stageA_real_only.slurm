#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s05
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/stage05_train_a_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
export SMOLVLA_TRAIN_VARIANT=a
bash "${REPO_ROOT}/scripts/smolvla_vggflow/run_stage.sh" stage10_train_loop
