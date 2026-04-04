#!/usr/bin/env bash
#SBATCH --job-name=pi05-train-smoke
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/pi05_train_smoke_%j.log

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

mkdir -p logs "${ROOT}/.cache/jax/xla_gpu_per_fusion_autotune_cache_dir"

source "${ROOT}/stable-worldmodel/.venv/bin/activate"

export PYTHONPATH="${ROOT}"
export XDG_CACHE_HOME="${ROOT}/.cache"
export OPENPI_DATA_HOME="${XDG_CACHE_HOME}/openpi"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export JAX_COMPILATION_CACHE_DIR="${XDG_CACHE_HOME}/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=${XDG_CACHE_HOME}/jax/xla_gpu_per_fusion_autotune_cache_dir --xla_gpu_strict_conv_algorithm_picker=false"

EXP_NAME="${EXP_NAME:-pi05_train_smoke_${SLURM_JOB_ID:-manual}}"

echo "[smoke] root=${ROOT}"
echo "[smoke] exp_name=${EXP_NAME}"
echo "[smoke] cuda devices:"
nvidia-smi || true

WANDB_MODE=disabled python external/openpi/scripts/train.py \
  debug_pi05 \
  --exp-name="${EXP_NAME}" \
  --overwrite \
  --no-wandb-enabled

echo "[smoke] PASS"
