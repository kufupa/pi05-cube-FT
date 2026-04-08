#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMOLVLA_PIPE_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

source "${SCRIPT_DIR}/config.sh"

_base_report_root="${SMOLVLA_REPORT_ROOT}"
_base_artifact_root="${SMOLVLA_ARTIFACT_ROOT}"
_base_log_root="${SMOLVLA_LOG_ROOT}"
_base_data_root="${SMOLVLA_DATA_ROOT}"
_base_jepa_source="${SMOLVLA_JEPA_SOURCE}"
_base_jepa_export_out="${SMOLVLA_JEPA_EXPORT_OUT}"

_warn_scope_reuse_nonblocking() {
  local label="$1"
  local path="$2"
  echo "smolvla: ${label} already exists (SMOLVLA_FAIL_ON_PATH_REUSE=1): ${path} -- non-blocking for staged execution; stage-level output guards enforce non-overwrite" >&2
}

if [[ -n "${SMOLVLA_RUN_SCOPE_ID:-}" ]]; then
  _scope_tag="run_${SMOLVLA_RUN_SCOPE_ID}"
  _scoped_art="${_base_artifact_root}/${_scope_tag}"
  _scoped_data_root="${_base_data_root}/${_scope_tag}"
  _scoped_jepa_export_out="${_base_jepa_export_out}/${_scope_tag}"
  if [[ "${SMOLVLA_FAIL_ON_PATH_REUSE:-0}" == "1" ]] && [[ -e "${_scoped_art}" ]]; then
    _warn_scope_reuse_nonblocking "scoped artifact root" "${_scoped_art}"
  fi
  if [[ "${SMOLVLA_FAIL_ON_PATH_REUSE:-0}" == "1" ]] && [[ -e "${_scoped_data_root}" ]]; then
    _warn_scope_reuse_nonblocking "scoped data root" "${_scoped_data_root}"
  fi
  if [[ "${SMOLVLA_FAIL_ON_PATH_REUSE:-0}" == "1" ]] && [[ -e "${_scoped_jepa_export_out}" ]]; then
    _warn_scope_reuse_nonblocking "scoped JEPA export root" "${_scoped_jepa_export_out}"
  fi
  SMOLVLA_ARTIFACT_ROOT="${_scoped_art}"
  SMOLVLA_REPORT_ROOT="${_base_report_root}/${_scope_tag}"
  SMOLVLA_LOG_ROOT="${_base_log_root}/${_scope_tag}"
  SMOLVLA_DATA_ROOT="${_scoped_data_root}"
  SMOLVLA_JEPA_EXPORT_OUT="${_scoped_jepa_export_out}"
  if [[ "${SMOLVLA_VGG_GATE_JSON}" == "${_base_report_root}"/* ]]; then
    SMOLVLA_VGG_GATE_JSON="${SMOLVLA_REPORT_ROOT}${SMOLVLA_VGG_GATE_JSON#${_base_report_root}}"
  fi
  if [[ "${SMOLVLA_DECISION_LOG}" == "${_base_report_root}"/* ]]; then
    SMOLVLA_DECISION_LOG="${SMOLVLA_REPORT_ROOT}${SMOLVLA_DECISION_LOG#${_base_report_root}}"
  fi
  if [[ "${SMOLVLA_PRESET_REPORT}" == "${_base_report_root}"/* ]]; then
    SMOLVLA_PRESET_REPORT="${SMOLVLA_REPORT_ROOT}${SMOLVLA_PRESET_REPORT#${_base_report_root}}"
  fi
  if [[ "${SMOLVLA_STAGE_D_DATA_ROOT}" == "${_base_data_root}"/* ]]; then
    SMOLVLA_STAGE_D_DATA_ROOT="${SMOLVLA_DATA_ROOT}${SMOLVLA_STAGE_D_DATA_ROOT#${_base_data_root}}"
  fi
  if [[ "${SMOLVLA_JEPA_SOURCE}" == "${_base_jepa_source}" ]]; then
    SMOLVLA_JEPA_SOURCE="${SMOLVLA_JEPA_EXPORT_OUT}"
  elif [[ "${SMOLVLA_JEPA_SOURCE}" == "${_base_jepa_source}"/* ]]; then
    SMOLVLA_JEPA_SOURCE="${SMOLVLA_JEPA_EXPORT_OUT}${SMOLVLA_JEPA_SOURCE#${_base_jepa_source}}"
  fi
fi

SMOLVLA_TMP_ROOT="${SMOLVLA_TMP_ROOT:-${SMOLVLA_CACHE_ROOT}/tmp}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SMOLVLA_CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${SMOLVLA_HF_HOME}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SMOLVLA_PIP_CACHE}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SMOLVLA_UV_CACHE}}"
export TMPDIR="${TMPDIR:-${SMOLVLA_TMP_ROOT}}"

mkdir -p \
  "${SMOLVLA_REPORT_ROOT}" \
  "${SMOLVLA_ARTIFACT_ROOT}/smolvla/smoke" \
  "${SMOLVLA_ARTIFACT_ROOT}/eval_videos/push_v3" \
  "${SMOLVLA_ARTIFACT_ROOT}/checkpoints" \
  "${SMOLVLA_DATA_ROOT}" \
  "${SMOLVLA_DATA_ROOT}/train" \
  "${SMOLVLA_DATA_ROOT}/val" \
  "${SMOLVLA_JEPA_EXPORT_OUT}" \
  "${SMOLVLA_LOG_ROOT}" \
  "${SMOLVLA_LOCK_ROOT}" \
  "${SMOLVLA_TMP_ROOT}"

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

smolvla_run_id() {
  local override="${SMOLVLA_TRAIN_RUN_ID:-}"
  if [[ -n "${override}" ]]; then
    echo "${override}"
    return 0
  fi

  local ts host user random_tag
  ts="$(date -u +"%Y%m%dT%H%M%SZ")"
  host="$(hostname -s 2>/dev/null || echo unknown)"
  user="${USER:-unknown}"
  random_tag="${RANDOM:-0}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "${ts}_${user}_${host}_${SLURM_JOB_ID}"
  else
    echo "${ts}_${user}_${host}_l${random_tag}"
  fi
}

emit_run_manifest() {
  local manifest_path="$1"
  local run_stage="$2"
  local run_id="$3"
  local variant="$4"
  local stage_label="$5"
  local output_dir="$6"
  local init_ckpt="$7"
  local command_str="$8"
  local dataset_root="$9"
  local max_steps="$10"
  local log_steps="$11"
  local save_steps="$12"

  python3 - "${manifest_path}" "${run_stage}" "${run_id}" "${variant}" "${stage_label}" "${output_dir}" "${init_ckpt}" "${command_str}" "${dataset_root}" "${max_steps}" "${log_steps}" "${save_steps}" <<'PY'
import json
import re
import shlex
import socket
import subprocess
from pathlib import Path
import os
from datetime import datetime, timezone


def _git_rev(path: str) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8", "replace")
            .strip()
        )
    except Exception:
        return ""


def _git_dirty(path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", path, "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return bool((result.stdout or "").strip())
    except Exception:
        return False


manifest_path, run_stage, run_id, variant, stage_label, output_dir, init_ckpt, command_str, dataset_root, max_steps, log_steps, save_steps = __import__("sys").argv[1:]


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _extract_max_steps_from_cmd(command: str) -> int | None:
    try:
        tokens = shlex.split(command)
    except Exception:
        return None
    for idx, tok in enumerate(tokens):
        if tok == "--max-steps" and idx + 1 < len(tokens):
            try:
                return int(tokens[idx + 1])
            except Exception:
                return None
        if tok.startswith("--max-steps="):
            try:
                return int(tok.split("=", 1)[1].strip("'\""))
            except Exception:
                return None
    return None

repo_root = os.environ.get("SMOLVLA_REPO_ROOT", "")

_max_steps = _safe_int(max_steps, None)
if _max_steps is None:
    _max_steps = _extract_max_steps_from_cmd(command_str)
if _max_steps is None:
    _max_steps = 0

payload = {
    "run_id": run_id,
    "run_stage": run_stage,
    "variant": variant,
    "stage_label": stage_label,
    "started_by": os.environ.get("USER", "unknown"),
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "git_rev": _git_rev(repo_root),
    "git_dirty": _git_dirty(repo_root),
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "max_steps": _max_steps,
    "log_steps": _safe_int(log_steps, 100),
    "save_steps": _safe_int(save_steps, 1000),
    "torch": None,
    "dataset_root": dataset_root,
    "checkpoint_from": init_ckpt,
    "output_dir": output_dir,
    "metrics_jsonl": f"{output_dir.rstrip('/')}/metrics.jsonl",
    "smolvla_train_cmd": command_str,
    "smolvla_train_report_to": os.environ.get("SMOLVLA_TRAIN_REPORT_TO", ""),
    "smolvla_wandb_mode": os.environ.get("SMOLVLA_WANDB_MODE", ""),
    "smolvla_train_env": {
        "smolvla_train_variant": os.environ.get("SMOLVLA_TRAIN_VARIANT", ""),
        "smolvla_enable_vgg": os.environ.get("SMOLVLA_ENABLE_VGG", ""),
        "smolvla_enable_train_d": os.environ.get("SMOLVLA_ENABLE_TRAIN_D", ""),
        "smolvla_jepa_source": os.environ.get("SMOLVLA_JEPA_SOURCE", ""),
        "slurm_partition": os.environ.get("SMOLVLA_DEFAULT_PARTITION", ""),
    },
    "manifest_schema_version": "smolvla_run_manifest_v1",
    "checkpoint_policy_mode": os.environ.get("SMOLVLA_CHECKPOINT_POLICY", "base_init_compare"),
    "run_scope_id": os.environ.get("SMOLVLA_RUN_SCOPE_ID", ""),
    "stage_a_dataset_root": os.environ.get("SMOLVLA_MANIFEST_STAGE_A_DATA_ROOT", ""),
    "stage_c_dataset_root": os.environ.get("SMOLVLA_MANIFEST_STAGE_C_DATA_ROOT", ""),
    "stage_b_jepa_dataset_root": os.environ.get("SMOLVLA_MANIFEST_STAGE_B_JEPA_ROOT", ""),
    "stage_b_mixed_dataset_root": os.environ.get("SMOLVLA_MANIFEST_STAGE_B_MIXED_ROOT", ""),
    "stage_d_dataset_root": os.environ.get("SMOLVLA_MANIFEST_STAGE_D_DATA_ROOT", ""),
}
try:
    import torch  # type: ignore
    payload["torch"] = torch.__version__
except Exception:
    payload["torch"] = os.environ.get("TORCH_VERSION", "")
Path(manifest_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

