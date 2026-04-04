#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s03
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=logs/stage03_install_jepa_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
bash "/vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh" stage03_install_jepa_wms
