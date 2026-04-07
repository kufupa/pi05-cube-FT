#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s03
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=logs/stage03_install_jepa_%j.log

set -euo pipefail
# Defaults come from scripts/smolvla_vggflow/config.sh (WM + SmolVLA export enabled). Do not force "safe"
# overrides here; that hid real export paths. If a node stack segfaults on VLM load, WM hub load, or
# CUDA+pygame render, opt in before sbatch, for example:
#   export SMOLVLA_JEPA_EXPORT_USE_HEURISTIC_EXEC=1
#   export SMOLVLA_JEPA_EXPORT_WM_ON_CPU=1
#   export SMOLVLA_JEPA_EXPORT_SKIP_WM=1
#   export SMOLVLA_JEPA_EXPORT_POLICY_LOAD_VLM_WEIGHTS=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
bash "${REPO_ROOT}/scripts/smolvla_vggflow/run_stage.sh" stage03_install_jepa_wms
