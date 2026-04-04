#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT}/stable-worldmodel/.venv}"
PY="${VENV_PATH}/bin/python"
ROOT_CACHE="${ROOT}/.cache"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_CACHE}}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${XDG_CACHE_HOME}/openpi}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${OPENPI_DATA_HOME}"

if [[ ! -x "${PY}" ]]; then
  echo "FATAL: missing ${PY}. Run: bash scripts/bootstrap_hpc_env.sh"
  exit 2
fi

"${PY}" "${ROOT}/scripts/download_models.py"
