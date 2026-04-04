#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s01
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/stage01_install_lerobot_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
bash "/vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh" stage01_install_lerobot_mw
