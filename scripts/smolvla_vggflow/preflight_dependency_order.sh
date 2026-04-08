#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

FAIL_COUNT=0
PASS_COUNT=0

CHECK_01_SLURM="CHECK_01_SLURM"
CHECK_02_ENVS="CHECK_02_ENVS"
CHECK_03_CUDA_RENDER="CHECK_03_CUDA_RENDER"
CHECK_04_JEPA="CHECK_04_JEPA"
CHECK_05_SMOLVLA="CHECK_05_SMOLVLA"
CHECK_06_EXPORT="CHECK_06_EXPORT"
CHECK_07_BRIDGE="CHECK_07_BRIDGE"

log_header() {
  local marker="$1"
  local label="$2"
  printf "\n[%s] %s\n" "${marker}" "${label}"
}

mark_pass() {
  local message="$1"
  PASS_COUNT=$((PASS_COUNT + 1))
  printf "  PASS %s\n" "${message}"
}

mark_fail() {
  local message="$1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf "  FAIL %s\n" "${message}" >&2
}

require_cmd() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    mark_pass "command available: ${cmd}"
  else
    mark_fail "missing required command: ${cmd}"
  fi
}

require_file() {
  local file_path="$1"
  if [[ -f "${file_path}" ]]; then
    mark_pass "file present: ${file_path}"
  else
    mark_fail "missing required file: ${file_path}"
  fi
}

require_exec() {
  local exec_path="$1"
  if [[ -x "${exec_path}" ]]; then
    mark_pass "executable present: ${exec_path}"
  else
    mark_fail "missing executable: ${exec_path}"
  fi
}

require_dir_or_creatable() {
  local dir_path="$1"
  if [[ -d "${dir_path}" ]]; then
    mark_pass "directory present: ${dir_path}"
    return
  fi
  if mkdir -p "${dir_path}" >/dev/null 2>&1; then
    mark_pass "directory created: ${dir_path}"
    return
  fi
  mark_fail "directory missing and cannot be created: ${dir_path}"
}

check_python_module() {
  local python_bin="$1"
  local module_name="$2"
  local label="$3"
  if [[ ! -x "${python_bin}" ]]; then
    mark_fail "${label}: python missing at ${python_bin}"
    return
  fi
  if "${python_bin}" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module_name}")
PY
  then
    mark_pass "${label}: import ${module_name}"
  else
    mark_fail "${label}: failed import ${module_name}"
  fi
}

check_01_slurm() {
  log_header "${CHECK_01_SLURM}" "scheduler prerequisites"
  require_cmd sbatch
  require_cmd squeue
  require_cmd sacct
  require_cmd sinfo
}

check_02_envs() {
  log_header "${CHECK_02_ENVS}" "python environment prerequisites"
  require_exec "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
  require_exec "${SMOLVLA_JEPA_ENV_DIR}/bin/python"
  require_exec "${SMOLVLA_VGG_ENV_DIR}/bin/python"
}

check_03_cuda_render() {
  log_header "${CHECK_03_CUDA_RENDER}" "cuda and render backend prerequisites"
  require_cmd nvidia-smi
  require_cmd xvfb-run

  if [[ -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
    if "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
      mark_pass "lerobot env reports torch.cuda.is_available()"
    else
      mark_fail "lerobot env reports CUDA unavailable"
    fi
  else
    mark_fail "cannot check CUDA without ${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
  fi

  if [[ "${SMOLVLA_JEPA_MUJOCO_GL:-}" == "egl" ]]; then
    mark_pass "SMOLVLA_JEPA_MUJOCO_GL=egl"
  else
    mark_fail "SMOLVLA_JEPA_MUJOCO_GL must be egl for headless rendering"
  fi

  if [[ "${SMOLVLA_JEPA_PYOPENGL_PLATFORM:-}" == "egl" ]]; then
    mark_pass "SMOLVLA_JEPA_PYOPENGL_PLATFORM=egl"
  else
    mark_fail "SMOLVLA_JEPA_PYOPENGL_PLATFORM must be egl for headless rendering"
  fi
}

check_04_jepa() {
  log_header "${CHECK_04_JEPA}" "jepa workflow prerequisites"
  require_file "${SCRIPT_DIR}/jepa_cem_paired_pushv3_export.py"
  require_file "${SCRIPT_DIR}/jepa_smoke_check.py"
  require_exec "${SMOLVLA_JEPA_ENV_DIR}/bin/python"
  check_python_module "${SMOLVLA_JEPA_ENV_DIR}/bin/python" "torch" "jepa env"

  if [[ -n "${SMOLVLA_JEPA_SOURCE:-}" ]]; then
    mark_pass "SMOLVLA_JEPA_SOURCE is configured: ${SMOLVLA_JEPA_SOURCE}"
  else
    mark_fail "SMOLVLA_JEPA_SOURCE is empty"
  fi
}

check_05_smolvla() {
  log_header "${CHECK_05_SMOLVLA}" "smolvla workflow prerequisites"
  require_file "${SCRIPT_DIR}/train_smolvla_vggflow.py"
  require_file "${SCRIPT_DIR}/run_stage.sh"
  require_file "${SCRIPT_DIR}/submit_workflow.sh"
  check_python_module "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" "torch" "smolvla env"

  if [[ -n "${SMOLVLA_INIT_CHECKPOINT:-}" ]]; then
    mark_pass "SMOLVLA_INIT_CHECKPOINT is configured: ${SMOLVLA_INIT_CHECKPOINT}"
  else
    mark_fail "SMOLVLA_INIT_CHECKPOINT is empty"
  fi
}

check_06_export() {
  log_header "${CHECK_06_EXPORT}" "export prerequisites"
  require_file "${SCRIPT_DIR}/jepa_cem_paired_pushv3_export.py"
  require_dir_or_creatable "${SMOLVLA_JEPA_EXPORT_OUT}"

  if [[ "${SMOLVLA_JEPA_EXPORT_EPISODES}" =~ ^[0-9]+$ ]] && (( SMOLVLA_JEPA_EXPORT_EPISODES > 0 )); then
    mark_pass "SMOLVLA_JEPA_EXPORT_EPISODES is a positive integer"
  else
    mark_fail "SMOLVLA_JEPA_EXPORT_EPISODES must be a positive integer"
  fi

  if [[ "${SMOLVLA_JEPA_EXPORT_MAX_STEPS}" =~ ^[0-9]+$ ]] && (( SMOLVLA_JEPA_EXPORT_MAX_STEPS > 0 )); then
    mark_pass "SMOLVLA_JEPA_EXPORT_MAX_STEPS is a positive integer"
  else
    mark_fail "SMOLVLA_JEPA_EXPORT_MAX_STEPS must be a positive integer"
  fi
}

check_07_bridge() {
  log_header "${CHECK_07_BRIDGE}" "bridge dataset prerequisites"
  require_file "${SCRIPT_DIR}/bridge_builder.py"
  require_dir_or_creatable "${SMOLVLA_DATA_ROOT}"
  require_dir_or_creatable "${SMOLVLA_DATA_ROOT}/train"
  require_dir_or_creatable "${SMOLVLA_DATA_ROOT}/val"
}

main() {
  printf "smolvla dependency-order preflight\n"
  printf "repo=%s\n" "${SMOLVLA_REPO_ROOT}"
  printf "workspace=%s\n" "${SMOLVLA_WORKSPACE_ROOT}"

  check_01_slurm
  check_02_envs
  check_03_cuda_render
  check_04_jepa
  check_05_smolvla
  check_06_export
  check_07_bridge

  printf "\nsummary: %s passed, %s failed\n" "${PASS_COUNT}" "${FAIL_COUNT}"
  if (( FAIL_COUNT > 0 )); then
    printf "result: FAIL\n" >&2
    return 1
  fi
  printf "result: PASS\n"
  return 0
}

main "$@"
