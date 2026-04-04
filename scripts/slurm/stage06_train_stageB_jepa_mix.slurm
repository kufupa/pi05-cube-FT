#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s06
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/stage06_train_b_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
bash "/vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh" stage10_train_loop
