#!/usr/bin/env bash
set -euo pipefail

# Bootstraps the repo into the path expected by current rollout/PBS scripts:
#   stable-worldmodel/.venv/bin/python

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT}/stable-worldmodel/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "[bootstrap] repo root: ${ROOT}"
echo "[bootstrap] venv path: ${VENV_PATH}"
echo "[bootstrap] python bin: ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "FATAL: ${PYTHON_BIN} not found in PATH"
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] uv not found, installing with pip --user"
  "${PYTHON_BIN}" -m pip install --user --upgrade uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "FATAL: uv still unavailable after installation attempt"
  exit 2
fi

mkdir -p "$(dirname "${VENV_PATH}")"
uv venv "${VENV_PATH}" --python "${PYTHON_BIN}"

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

echo "[bootstrap] upgrading packaging tools"
uv pip install --upgrade pip setuptools wheel

echo "[bootstrap] installing repo package (editable)"
uv pip install -e "${ROOT}"

echo "[bootstrap] ensuring vendored openpi editable install"
uv pip install -e "${ROOT}/external/openpi"

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
