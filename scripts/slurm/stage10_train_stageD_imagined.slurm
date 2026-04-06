#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s10d
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/stage10_train_d_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
# Optional lane: VGG auxiliary on imagined-heavy LeRobot root (default SMOLVLA_STAGE_D_DATA_ROOT=.../val).
# Same base checkpoint as A/B/C; not chained from StageC weights. Not part of the default 11-stage DAG.
export SMOLVLA_TRAIN_VARIANT=d
export SMOLVLA_ENABLE_VGG=0
export SMOLVLA_ENABLE_TRAIN_D=1
bash "${REPO_ROOT}/scripts/smolvla_vggflow/run_stage.sh" stage10_train_loop
