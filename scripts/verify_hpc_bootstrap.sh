#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT}/stable-worldmodel/.venv}"
MODE="${MODE:-quick}"  # quick|full
CHECKPOINT="${PI05_OGBENCH_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_base}"
ROOT_CACHE="${ROOT}/.cache"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

if [[ "${MODE}" != "quick" && "${MODE}" != "full" ]]; then
  echo "FATAL: --mode must be quick or full"
  exit 2
fi

PY="${VENV_PATH}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "FATAL: missing ${PY}. Run: bash scripts/bootstrap_hpc_env.sh"
  exit 2
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_CACHE}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${XDG_CACHE_HOME}/openpi}"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${OPENPI_DATA_HOME}"

echo "[verify] root=${ROOT}"
echo "[verify] mode=${MODE}"
echo "[verify] python=${PY}"

echo "[verify] import checks"
"${PY}" - <<'PY'
import importlib
mods = ["numpy", "torch", "gymnasium", "mujoco", "ogbench", "imageio", "certifi"]
for m in mods:
    importlib.import_module(m)
print("imports: PASS")
PY

if [[ "${MODE}" == "full" ]]; then
  echo "[verify] OpenPI import checks"
  "${PY}" - <<'PY'
import importlib
mods = ["openpi", "jax", "flax"]
for m in mods:
    importlib.import_module(m)
print("openpi imports: PASS")
PY
fi

run_headless_if_needed() {
  if [[ -n "${DISPLAY:-}" || -n "${MUJOCO_GL:-}" ]]; then
    "$@"
    return
  fi
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "FATAL: DISPLAY/MUJOCO_GL unset and xvfb-run not found"
    exit 2
  fi
  MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1280x1024x24" "$@"
}

echo "[verify] data generation preflight"
run_headless_if_needed \
  "${PY}" \
  "${ROOT}/cube_dataset/finetune_pi05_cube_single_v1/render_cube_single_dualcam.py" \
  --preflight

if [[ "${MODE}" == "full" ]]; then
  echo "[verify] rollout smoke with OpenPI checkpoint"
  run_headless_if_needed \
    "${PY}" \
    "${ROOT}/cube_dataset/run_pi05_base_ur5e_rollouts.py" \
    --smoke-test \
    --require-openpi \
    --checkpoint "${CHECKPOINT}"
else
  echo "[verify] quick mode skips --require-openpi rollout smoke"
fi

echo "[verify] PASS (${MODE})"
