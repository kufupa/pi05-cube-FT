#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMOLVLA_PIPE_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

source "${SCRIPT_DIR}/config.sh"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SMOLVLA_CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${SMOLVLA_HF_HOME}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SMOLVLA_PIP_CACHE}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SMOLVLA_UV_CACHE}}"

mkdir -p \
  "${SMOLVLA_REPORT_ROOT}" \
  "${SMOLVLA_ARTIFACT_ROOT}/smolvla/smoke" \
  "${SMOLVLA_ARTIFACT_ROOT}/eval_videos/push_v3" \
  "${SMOLVLA_ARTIFACT_ROOT}/checkpoints" \
  "${SMOLVLA_DATA_ROOT}" \
  "${SMOLVLA_DATA_ROOT}/train" \
  "${SMOLVLA_DATA_ROOT}/val" \
  "${SMOLVLA_LOG_ROOT}" \
  "${SMOLVLA_LOCK_ROOT}"

log() {
  local level="$1"
  shift
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[${ts}] [${level}] $*" | tee -a "${SMOLVLA_LOG_ROOT}/workflow.log"
}

log_info() {
  log INFO "$*"
}

log_warn() {
  log WARN "$*"
}

log_error() {
  log ERROR "$*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    log_error "Missing required command: ${cmd}"
    return 1
  fi
}

run_logged() {
  local log_file="$1"
  shift
  log_info "Run: $*"
  "$@" >"${log_file}" 2>&1
}

write_yaml() {
  local path="$1"
  shift
  local key="$1"
  local value="$2"
  {
    printf "%s: %s\n" "${key}" "${value}"
  } >>"${path}"
}

append_decision() {
  local action="$1"
  local reason="$2"
  local outcome="$3"
  local when
  when="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  {
    echo ""
    echo "## ${when}"
    echo "- **Action**: ${action}"
    echo "- **Reason**: ${reason}"
    echo "- **Outcome**: ${outcome}"
  } >>"${SMOLVLA_DECISION_LOG}"
}

require_env() {
  local var="$1"
  if [[ -z "${!var-}" ]]; then
    log_error "${var} is not set"
    return 1
  fi
}

with_venv() {
  local env_path="$1"
  local cmd="$2"
  shift 2
  if [[ ! -x "${env_path}/bin/python" ]]; then
    log_error "Python not found in ${env_path}"
    return 1
  fi

  # shellcheck disable=SC1090
  source "${env_path}/bin/activate"
  export PATH="${env_path}/bin:${PATH}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
  "$cmd" "$@"
}

emit_lock() {
  local env_name="$1"
  local env_dir="$2"
  local lock_file="${SMOLVLA_LOCK_ROOT}/${env_name}_pip_freeze.txt"
  if [[ ! -x "${env_dir}/bin/python" ]]; then
    return 1
  fi
  # shellcheck disable=SC1090
  source "${env_dir}/bin/activate"
  "${env_dir}/bin/python" -V >"${lock_file}.version"
  "${env_dir}/bin/pip" freeze >>"${lock_file}"
}

readlink_safe() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  else
    echo "$1"
  fi
}

