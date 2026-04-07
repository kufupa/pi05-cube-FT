#!/usr/bin/env bash
set -euo pipefail

# Bootstraps the repo into the path expected by current rollout/PBS scripts:
#   stable-worldmodel/.venv/bin/python

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT}/stable-worldmodel/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
ROOT_CACHE="${ROOT}/.cache"

echo "[bootstrap] repo root: ${ROOT}"
echo "[bootstrap] venv path: ${VENV_PATH}"
echo "[bootstrap] python bin: ${PYTHON_BIN}"
echo "[bootstrap] cache root: ${ROOT_CACHE}"

# Prefer repo-local caches (override XDG_CACHE_HOME if you use a shared workspace volume).
mkdir -p "${ROOT_CACHE}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_CACHE}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${XDG_CACHE_HOME}/openpi}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${OPENPI_DATA_HOME}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "FATAL: ${PYTHON_BIN} not found in PATH"
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] uv not found; installing repo-local uv bootstrap env"
  UV_BOOTSTRAP="${ROOT}/.tools/uv-bootstrap"
  "${PYTHON_BIN}" -m venv "${UV_BOOTSTRAP}"
  "${UV_BOOTSTRAP}/bin/pip" install --upgrade pip
  "${UV_BOOTSTRAP}/bin/pip" install uv
  export PATH="${UV_BOOTSTRAP}/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "FATAL: uv still unavailable after installation attempt"
  exit 2
fi

mkdir -p "$(dirname "${VENV_PATH}")"
if [[ -d "${VENV_PATH}" ]]; then
  echo "[bootstrap] existing venv found, reusing: ${VENV_PATH}"
else
  uv venv "${VENV_PATH}" --python "${PYTHON_BIN}"
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

echo "[bootstrap] upgrading packaging tools"
uv pip install --upgrade pip setuptools wheel

echo "[bootstrap] installing repo package (editable)"
uv pip install -e "${ROOT}"

echo "[bootstrap] ensuring vendored openpi editable install"
uv pip install -e "${ROOT}/external/openpi"

echo "[bootstrap] pinning transformers for OpenPI compatibility"
uv pip install "transformers==4.53.2"

echo "[bootstrap] applying OpenPI transformers patch files"
TRANSFORMERS_SITE="$("${VENV_PATH}/bin/python" - <<'PY'
import pathlib
import transformers
print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)"
cp -r "${ROOT}/external/openpi/src/openpi/models_pytorch/transformers_replace/"* "${TRANSFORMERS_SITE}/"

echo "[bootstrap] ensuring ogbench + render stack"
uv pip install \
  ogbench \
  gymnasium \
  mujoco \
  imageio \
  imageio-ffmpeg \
  certifi

echo
echo "[bootstrap] complete"
echo "Activate with:"
echo "  source \"${VENV_PATH}/bin/activate\""
echo "Then export:"
echo "  export PYTHONPATH=\"${ROOT}\""
