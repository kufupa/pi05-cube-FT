#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s01b-mw
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/stage01b_metaworld_%j.log

# CPU-only: pip install Meta-World into the lerobot env (needed for phase07 export + baselines).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
bash "${REPO_ROOT}/scripts/smolvla_vggflow/run_stage.sh" stage01b_install_metaworld
