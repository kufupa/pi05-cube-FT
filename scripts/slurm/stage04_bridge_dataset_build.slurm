#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s04
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/stage04_bridge_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
bash "/vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh" stage04_bridge_dataset_build
