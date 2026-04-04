#!/usr/bin/env bash
#SBATCH --job-name=pi05-droid-ft
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pi05_droid_finetune_%j.log

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

CONFIG_NAME="${CONFIG_NAME:-pi05_droid_finetune}"
DATASET_REPO_ID="${DATASET_REPO_ID:-}"
EXP_NAME="${EXP_NAME:-pi05_droid_ft_${SLURM_JOB_ID:-manual}}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WEIGHTS_PATH="${WEIGHTS_PATH:-}"
ASSETS_DIR="${ASSETS_DIR:-}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"

if [[ "${CONFIG_NAME}" == "pi05_droid_finetune" ]]; then
  if [[ -z "${DATASET_REPO_ID}" ]]; then
    echo "FATAL: set DATASET_REPO_ID for pi05_droid_finetune"
    echo "Example:"
    echo "  DATASET_REPO_ID=myorg/my_droid_dataset sbatch scripts/slurm/run_pi05_droid_finetune.slurm"
    exit 2
  fi
  if [[ -z "${WEIGHTS_PATH}" ]]; then
    WEIGHTS_PATH="gs://openpi-assets/checkpoints/pi05_droid/params"
  fi
  if [[ -z "${ASSETS_DIR}" ]]; then
    ASSETS_DIR="gs://openpi-assets/checkpoints/pi05_droid/assets"
  fi
fi
PRECHECK_DATASET="${PRECHECK_DATASET:-1}"

WANDB_FLAG="--no-wandb-enabled"
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  WANDB_FLAG="--wandb-enabled"
fi

echo "[ft] root=${ROOT}"
echo "[ft] config=${CONFIG_NAME}"
echo "[ft] exp_name=${EXP_NAME}"
if [[ -n "${DATASET_REPO_ID}" ]]; then
  echo "[ft] dataset_repo_id=${DATASET_REPO_ID}"
fi
echo "[ft] num_train_steps=${NUM_TRAIN_STEPS} batch_size=${BATCH_SIZE} fsdp_devices=${FSDP_DEVICES}"
if [[ -n "${WEIGHTS_PATH}" ]]; then
  echo "[ft] weights=${WEIGHTS_PATH}"
fi
if [[ -n "${ASSETS_DIR}" ]]; then
  echo "[ft] assets=${ASSETS_DIR}"
fi
echo "[ft] cuda devices:"
nvidia-smi || true

if [[ "${CONFIG_NAME}" == "pi05_droid_finetune" && "${PRECHECK_DATASET}" == "1" ]]; then
  echo "[ft] dataset compatibility precheck..."
  python scripts/check_pi05_dataset_compat.py --repo-id="${DATASET_REPO_ID}"
fi

TRAIN_ARGS=(
  "${CONFIG_NAME}"
  --exp-name="${EXP_NAME}"
  --overwrite
  "${WANDB_FLAG}"
  --num-train-steps="${NUM_TRAIN_STEPS}"
  --batch-size="${BATCH_SIZE}"
  --fsdp-devices="${FSDP_DEVICES}"
  --num-workers="${NUM_WORKERS}"
  --save-interval="${SAVE_INTERVAL}"
  --log-interval="${LOG_INTERVAL}"
)

if [[ -n "${DATASET_REPO_ID}" ]]; then
  TRAIN_ARGS+=(--data.repo-id="${DATASET_REPO_ID}")
fi
if [[ -n "${ASSETS_DIR}" ]]; then
  TRAIN_ARGS+=(--data.assets.assets-dir="${ASSETS_DIR}")
fi
if [[ -n "${WEIGHTS_PATH}" ]]; then
  TRAIN_ARGS+=(--weight-loader.params-path="${WEIGHTS_PATH}")
fi

python external/openpi/scripts/train.py "${TRAIN_ARGS[@]}"

echo "[ft] DONE"
