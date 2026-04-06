#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

STAGE="${1:?Usage: $0 <stage_id>}"
REPORT_FILE="${SMOLVLA_REPORT_ROOT}/${STAGE}_status.md"

log_capture() {
  local label="$1"
  shift
  local _log_capture_ec=0
  # Capture exit status without tripping set -e inside the group (so callers can `if ! log_capture ...`).
  set +e
  {
    echo
    echo "## ${label}"
    echo '```'
    "$@"
    _log_capture_ec=$?
    echo '```'
  } >>"${REPORT_FILE}"
  set -e
  return "${_log_capture_ec}"
}

append_passfail() {
  local status="$1"
  local text="$2"
  {
    echo
    echo "- [${status}] ${text}"
    echo "  - $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  } >>"${REPORT_FILE}"
}

env_python_version() {
  local env_dir="$1"
  "${env_dir}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

env_site_package_root() {
  local env_dir="$1"
  "${env_dir}/bin/python" -c 'import site; print(site.getsitepackages()[0])'
}

run_env_python_siteclean() {
  local env_dir="$1"
  shift
  local site_pkg
  site_pkg="$(env_site_package_root "${env_dir}")"
  (cd / && PYTHONPATH="${site_pkg}" "${env_dir}/bin/python" "$@")
}

resolve_python_binary() {
  local target="${1}"
  local avoid_prefix="${2:-}"
  local candidate
  local fallback=""
  for candidate in "python${target}" "python${target%.*}" "python3" "python"; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      local resolved
      resolved="$(command -v "${candidate}")"
      if [[ -n "${avoid_prefix}" && "${resolved}" == "${avoid_prefix}"* ]]; then
        if [[ -z "${fallback}" ]]; then
          fallback="${resolved}"
        fi
        continue
      fi
      echo "${resolved}"
      return 0
    fi
  done
  if [[ -n "${fallback}" ]]; then
    echo "${fallback}"
    return 0
  fi
  return 1
}

require_slurm_gpu_stage() {
  local stage_label="$1"

  if [[ "${SMOLVLA_REQUIRE_GPU_STAGES:-1}" != "1" ]]; then
    return 0
  fi

  if [[ "${SMOLVLA_ALLOW_CPU_RUN:-0}" == "1" ]]; then
    log_warn "SMOLVLA_ALLOW_CPU_RUN=1 set; bypassing GPU-only guard for ${stage_label}"
    return 0
  fi

  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    log_error "blocked local execution for ${stage_label}; this stage requires Slurm GPU allocation"
    append_decision "${stage_label} execution" "run blocked: must use sbatch/queue for this stage" "FAIL"
    exit 1
  fi

  if [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
    log_error "blocked ${stage_label}: missing ${SMOLVLA_LEROBOT_ENV_DIR}/bin/python for CUDA check"
    append_decision "${stage_label} execution" "run blocked: missing runtime env for guard check" "FAIL"
    exit 1
  fi

  local has_cuda
  if ! has_cuda="$("${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import torch
print(int(torch.cuda.is_available()))
PY
  )"; then
    log_error "blocked ${stage_label}: unable to import torch in ${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
    append_decision "${stage_label} execution" "run blocked: torch import failed in runtime env" "FAIL"
    exit 1
  fi
  if [[ "${has_cuda}" != "1" ]]; then
    log_error "blocked ${stage_label}: CUDA unavailable for Slurm job ${SLURM_JOB_ID}"
    append_decision "${stage_label} execution" "run blocked: GPU resources missing in current allocation" "FAIL"
    exit 1
  fi
  log_info "GPU stage ${stage_label} running on Slurm job ${SLURM_JOB_ID} with CUDA available: ${has_cuda}"
}

check_jepa_source_has_trajectory_artifacts() {
  local source="$1"

  if [[ "${SMOLVLA_JEPA_SKIP_SOURCE_CHECK:-0}" == "1" ]]; then
    return 0
  fi

  local status=0
  if python3 - "$source" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]).expanduser().resolve()
if not source.exists():
    print(f"[bridge source] MISSING: {source}")
    sys.exit(1)

if source.is_file():
    name = source.name.lower()
    suffix = source.suffix.lower()
    if name.endswith(".pth.tar") or name.endswith(".ckpt"):
        print(f"[bridge source] CHECKPOINT_FILE: {source}")
        sys.exit(2)
    if suffix in {".json", ".jsonl", ".npz", ".pt", ".pth", ".pickle", ".pkl"}:
        print(f"[bridge source] FILE_OK: {source}")
        sys.exit(0)
    print(f"[bridge source] UNSUPPORTED_FILE: {source}")
    sys.exit(1)

supported = {".json", ".jsonl", ".npz", ".pth", ".pt", ".pickle", ".pkl"}
for path in sorted(source.rglob("*")):
    if not path.is_file():
        continue
    name = path.name.lower()
    if name.endswith(".pth.tar"):
        continue
    if path.suffix.lower() in supported:
        print(f"[bridge source] FOUND: {path}")
        sys.exit(0)

print(f"[bridge source] NO_TRAJECTORY_FILE_UNDER: {source}")
sys.exit(1)
PY
  then
    return 0
  else
    status=$?
    return "${status}"
  fi
}

read_bridge_summary_counts() {
  local summary_path="$1"
  local counts
  counts="$(python3 - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0 0 0")
    raise SystemExit(0)
data = json.loads(path.read_text(encoding="utf-8"))
print(
    int(data.get("train_records", 0)),
    int(data.get("val_records", 0)),
    int(1 if data.get("empty_inputs", False) else 0),
)
PY
)"
  echo "${counts}"
}

stage00_inventory() {
  log_info "phase00_inventory: collecting system, scheduler and rendering preflight"
  cat >"${REPORT_FILE}" <<EOF
# phase00_inventory
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
  {
    echo "## Host / runtime"
    echo "ROOT=${SMOLVLA_WORKSPACE_ROOT}"
    echo "USER=$(id -un)"
    echo "HOST=$(hostname)"
    uname -a
  } >>"${REPORT_FILE}"

  log_capture "Scheduler probes" bash -c '
    if command -v sinfo >/dev/null 2>&1; then
      sinfo -a
    else
      echo "sinfo: command not found"
    fi
    if command -v scontrol >/dev/null 2>&1; then
      scontrol show config | head -n 80
    else
      echo "scontrol: command not found"
    fi
  '

  log_capture "GPU and CUDA probes" bash -c '
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi
    else
      echo "nvidia-smi: command not found"
    fi
    python3 - <<PY
import importlib
mods = ["torch", "numpy", "gymnasium", "mujoco"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m}: PASS")
    except Exception as exc:
        print(f"{m}: FAIL {exc}")
PY
  '

  log_capture "Offline/IO probes" bash -c '
    command -v ffmpeg >/dev/null && ffmpeg -version | head -n 1 || echo "ffmpeg: missing"
    command -v xvfb-run >/dev/null || echo "xvfb-run: missing"
    python3 - <<PY
import os
print("pwd", os.getcwd())
PY
  '

  append_passfail "PASS" "preflight report saved to ${REPORT_FILE}"
  append_decision "phase00 inventory collection" "required for safe execution on gpucluster3" "PASS; report written"
}

create_uv_env() {
  local env_dir="$1"
  local python_ver="$2"
  local created_py=""
  local requested_bin=""
  if [[ -x "${env_dir}/bin/python" ]]; then
    if [[ "$(env_python_version "${env_dir}")" == "${python_ver}" ]]; then
      log_info "env exists, matching python ${python_ver}: ${env_dir}"
      return 0
    fi
    if [[ "${SMOLVLA_ALLOW_PYTHON_FALLBACK:-1}" == "1" ]]; then
      log_warn "env python mismatch for ${env_dir}; reusing fallback interpreter by policy"
      return 0
    fi
    log_warn "env python mismatch for ${env_dir}; recreating"
    rm -rf "${env_dir}"
  fi
  log_info "creating env: ${env_dir} (python ${python_ver})"
  mkdir -p "$(dirname "${env_dir}")"

  if ! requested_bin="$(resolve_python_binary "${python_ver}" "${env_dir}")"; then
    log_error "failed to resolve any python launcher for request ${python_ver}"
    return 1
  fi
  if [[ "${requested_bin}" != "$(command -v "python${python_ver}")" ]]; then
    log_warn "requested python ${python_ver} binary not found; falling back to ${requested_bin}"
  fi

  if command -v uv >/dev/null 2>&1; then
    uv venv "${env_dir}" --python "${requested_bin}"
  else
    "${requested_bin}" -m venv "${env_dir}"
  fi

  created_py="$(env_python_version "${env_dir}")"
  if [[ "${created_py}" != "${python_ver}" ]]; then
    if [[ "${SMOLVLA_ALLOW_PYTHON_FALLBACK:-1}" == "1" ]]; then
      log_warn "env ${env_dir} created with python ${created_py} (requested ${python_ver}); continuing by policy"
      log_info "export SMOLVLA_ALLOW_PYTHON_FALLBACK=0 to fail hard on version mismatch"
    else
      log_error "env ${env_dir} created with python ${created_py} but ${python_ver} was requested"
      log_error "set SMOLVLA_ALLOW_PYTHON_FALLBACK=1 to continue with fallback interpreter"
      return 1
    fi
  else
    log_info "env ${env_dir} created with requested python ${created_py}"
  fi
  return 0
}


install_seed_packages() {
  local env_dir="$1"
  local env_label="$2"
  local index_url="$3"
  shift 3
  local packages=("$@")
  if (( ${#packages[@]} == 0 )); then
    return 0
  fi

  local cache_cmd="mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && source '${env_dir}/bin/activate'"
  local package_list=""
  local package
  for package in "${packages[@]}"; do
    package_list="${package_list} ${package}"
  done

  if command -v uv >/dev/null 2>&1; then
    if [[ -n "${index_url}" ]]; then
      if log_capture "Seed env ${env_label} packages (uv, ${index_url})" bash -lc "${cache_cmd} && uv pip install --no-cache-dir --index-url ${index_url}${package_list}"; then
        return 0
      fi
    fi
    if log_capture "Seed env ${env_label} packages (uv)" bash -lc "${cache_cmd} && uv pip install --no-cache-dir${package_list}"; then
      return 0
    fi
  fi

  if ! log_capture "Seed env ${env_label} ensure pip bootstrap" bash -lc "${cache_cmd} && '${env_dir}/bin/python' -m ensurepip --upgrade --default-pip"; then
    log_error "pip bootstrap failed for ${env_label}"
  fi

  if [[ -n "${index_url}" ]]; then
    if log_capture "Seed env ${env_label} packages (pip, ${index_url})" bash -lc "${cache_cmd} && '${env_dir}/bin/python' -m pip install --no-cache-dir --index-url ${index_url} ${package_list}"; then
      return 0
    fi
  fi

  if log_capture "Seed env ${env_label} packages (pip)" bash -lc "${cache_cmd} && '${env_dir}/bin/python' -m pip install --no-cache-dir ${package_list}"; then
    return 0
  fi

  log_error "Seed dependency install failed for ${env_label}: ${package_list}"
  return 1
}

seed_torch_stack() {
  local env_dir="$1"
  local env_label="$2"
  install_seed_packages "${env_dir}" "${env_label}" "https://download.pytorch.org/whl/cu124" torch torchvision torchaudio
}

seed_core_python_env() {
  local env_dir="$1"
  local env_label="$2"
  local include_torch="$3"
  shift 3
  local -a extras=("$@")

  if [[ "${include_torch}" == "1" ]]; then
    if ! seed_torch_stack "${env_dir}" "${env_label}"; then
      return 1
    fi
  fi

  if (( ${#extras[@]} > 0 )); then
    install_seed_packages "${env_dir}" "${env_label}" "" "${extras[@]}"
    return $?
  fi

  return 0
}

verify_env_imports() {
  local env_dir="$1"
  local env_label="$2"
  shift 2
  local module
  local import_failed=0

  for module in "$@"; do
    if "${env_dir}/bin/python" -c "import ${module}" >/dev/null 2>&1; then
      log_info "import check pass: ${env_label}.${module}"
    else
      log_error "import check fail: ${env_label} missing ${module}"
      import_failed=1
    fi
  done
  if (( import_failed != 0 )); then
    return 1
  fi
  return 0
}

snapshot_env() {
  local env_dir="$1"
  local env_name="$2"
  local freeze_path="${SMOLVLA_LOCK_ROOT}/${env_name}_pip_freeze.txt"
  source "${env_dir}/bin/activate"
  "${env_dir}/bin/python" -m pip freeze > "${freeze_path}"
  "${env_dir}/bin/python" -V >> "${freeze_path}"
  log_info "environment lock written: ${freeze_path}"
}

stage01_env_topology() {
  log_info "phase01_env_topology: creating isolated envs"
  mkdir -p "${SMOLVLA_ENV_ROOT}"

  cat >"${REPORT_FILE}" <<EOF
# phase01_env_topology
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

  create_uv_env "${SMOLVLA_LEROBOT_ENV_DIR}" "3.10"
  create_uv_env "${SMOLVLA_JEPA_ENV_DIR}" "3.10"
  create_uv_env "${SMOLVLA_VGG_ENV_DIR}" "3.11"

  if ! seed_core_python_env "${SMOLVLA_LEROBOT_ENV_DIR}" "lerobot_mw_py310" 1 \
    typing_extensions transformers diffusers accelerate safetensors sentencepiece pillow pyarrow; then
    append_passfail "FAIL" "core package bootstrap failed for lerobot environment"
    append_decision "phase01 env bootstrap" "seed core deps for lerobot stack failed" "FAIL"
    exit 1
  fi
  if ! verify_env_imports "${SMOLVLA_LEROBOT_ENV_DIR}" "lerobot_mw_py310" \
    torch typing_extensions transformers diffusers pyarrow; then
    append_passfail "FAIL" "core import checks failed for lerobot environment"
    append_decision "phase01 env bootstrap" "import guard for seeded lerobot deps failed" "FAIL"
    exit 1
  fi

  if ! seed_core_python_env "${SMOLVLA_JEPA_ENV_DIR}" "jepa_wms_py310" 1 \
    typing_extensions transformers; then
    append_passfail "FAIL" "core package bootstrap failed for jepa environment"
    append_decision "phase01 env bootstrap" "seed core deps for JEPA stack failed" "FAIL"
    exit 1
  fi
  if ! verify_env_imports "${SMOLVLA_JEPA_ENV_DIR}" "jepa_wms_py310" torch typing_extensions transformers; then
    append_passfail "FAIL" "core import checks failed for jepa environment"
    append_decision "phase01 env bootstrap" "import guard for seeded JEPA deps failed" "FAIL"
    exit 1
  fi

  if ! seed_core_python_env "${SMOLVLA_VGG_ENV_DIR}" "vggflow_py311" 1 \
    typing_extensions transformers; then
    append_passfail "FAIL" "core package bootstrap failed for vgg environment"
    append_decision "phase01 env bootstrap" "seed core deps for VGG stack failed" "FAIL"
    exit 1
  fi
  if ! verify_env_imports "${SMOLVLA_VGG_ENV_DIR}" "vggflow_py311" torch typing_extensions transformers; then
    append_passfail "FAIL" "core import checks failed for vgg environment"
    append_decision "phase01 env bootstrap" "import guard for seeded VGG deps failed" "FAIL"
    exit 1
  fi

  snapshot_env "${SMOLVLA_LEROBOT_ENV_DIR}" "lerobot_mw"
  snapshot_env "${SMOLVLA_JEPA_ENV_DIR}" "jepa_wms"
  snapshot_env "${SMOLVLA_VGG_ENV_DIR}" "vggflow"

cat >>"${REPORT_FILE}" <<EOF
## env creation status
- lerobot_mw: ${SMOLVLA_LEROBOT_ENV_DIR}
- jepa_wms: ${SMOLVLA_JEPA_ENV_DIR}
- vggflow: ${SMOLVLA_VGG_ENV_DIR}
EOF
  append_passfail "PASS" "all env roots created"
  append_decision "phase01 env bootstrap" "isolate dependencies across stacks" "PASS"

  if ! ensure_smolvla_stack; then
    append_passfail "FAIL" "SmolVLA stack bootstrap failed during phase01"
    append_decision "phase01 env bootstrap" "smolvla package install failed" "FAIL"
    exit 1
  fi
}

stage02_gpu_compat() {
  log_info "phase02_gpu_compat: checking CUDA + MuJoCo + media runtime"
  cat >"${REPORT_FILE}" <<EOF
# phase02_gpu_compat
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

if [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  log_error "missing env ${SMOLVLA_LEROBOT_ENV_DIR}; run phase01 first"
  exit 1
fi

source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
if "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" -c "import torch" >/dev/null 2>&1; then
  log_info "torch already available"
else
  log_warn "torch missing; installing runtime stack"
  local install_ok=0
  if command -v uv >/dev/null 2>&1; then
    if log_capture "Torch install (uv)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio || uv pip install torch torchvision torchaudio"; then
      install_ok=1
    fi
  else
    if log_capture "Torch install (pip)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m ensurepip --upgrade --default-pip && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio || '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --no-cache-dir torch torchvision torchaudio"; then
      install_ok=1
    fi
  fi
  if [[ "${install_ok}" -eq 0 ]]; then
    append_passfail "WARN" "torch install failed; continuing with degraded phase02"
    append_decision "phase02 gpu compatibility" "torch/runtime unavailable (install blocked)" "DEGRADED"
    return 0
  fi
fi

log_capture "Torch CUDA probe" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
    print("fp16_supported", torch.cuda.is_fp16_supported())
PY

log_capture "MuJoCo/render smoke" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import importlib
import os
mods = ["mujoco", "gymnasium", "ffmpeg"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m}: PASS")
    except Exception as exc:
        print(f"{m}: {type(exc).__name__}: {exc}")
print("MUJOCO_GL=", os.environ.get("MUJOCO_GL", "<unset>"))
PY

if log_capture "Headless render check" bash -lc '
if command -v xvfb-run >/dev/null 2>&1; then
  MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1024x768x24" \
    python3 - <<PY
import mujoco
print("render_probe_ok", bool(mujoco))
PY
else
  echo "xvfb-run missing; skip render check"
fi
'; then
  : 
else
  log_warn "headless render check failed (non-fatal)"
fi

append_passfail "PASS" "compatibility probes captured in ${REPORT_FILE}"
append_decision "phase02 gpu compatibility" "required guardrails before installs/evals" "PASS"
}

stage03_smolvla_install() {
  log_info "phase03_smolvla_install: install LeRobot / SmolVLA dependencies"
  cat >"${REPORT_FILE}" <<EOF
# phase03_smolvla_install
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

if [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  log_error "missing env ${SMOLVLA_LEROBOT_ENV_DIR}; run phase01 first"
  exit 1
fi

source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
smolvla_site_pkg="$(env_site_package_root "${SMOLVLA_LEROBOT_ENV_DIR}")"
if command -v uv >/dev/null 2>&1; then
  log_capture "Base package install (uv)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && uv pip install --upgrade pip setuptools wheel && uv pip install --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir torch torchvision torchaudio || uv pip install --no-cache-dir torch torchvision torchaudio && uv pip install --no-cache-dir \"git+https://github.com/huggingface/lerobot.git\""
else
  log_capture "Base package install (pip)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m ensurepip --upgrade --default-pip && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --upgrade pip && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --ignore-installed --no-deps setuptools==82.0.1 wheel && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio || '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --no-cache-dir torch torchvision torchaudio && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --no-cache-dir 'git+https://github.com/huggingface/lerobot.git'"
fi
if command -v uv >/dev/null 2>&1; then
    log_capture "SmolVLA dependency seed (uv)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && uv pip install --no-cache-dir --upgrade transformers diffusers==0.35.1 accelerate safetensors sentencepiece pillow pyarrow"
else
  log_capture "SmolVLA dependency seed (pip)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m ensurepip --upgrade --default-pip && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --no-cache-dir --upgrade transformers diffusers==0.35.1 accelerate safetensors sentencepiece pillow pyarrow"
fi
if [[ -f "${smolvla_site_pkg}/lerobot/policies/__init__.py" ]]; then
  log_capture "Compat shim for lerobot policies/__init__" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import pathlib
import site

init_py = pathlib.Path(site.getsitepackages()[0]) / "lerobot" / "policies" / "__init__.py"
text = init_py.read_text(encoding="utf-8")
if "try:\n    from .groot.configuration_groot import GrootConfig as GrootConfig\nexcept Exception:" not in text:
    text = text.replace(
        "from .groot.configuration_groot import GrootConfig as GrootConfig",
        "try:\n    from .groot.configuration_groot import GrootConfig as GrootConfig\nexcept Exception:\n    GrootConfig = None",
    1,
    )
    init_py.write_text(text, encoding="utf-8")
PY
else
  log_warn "skipping lerobot compat shim; policies/__init__.py absent before patching"
fi

if run_env_python_siteclean "${SMOLVLA_LEROBOT_ENV_DIR}" - <<'PY'
import torch
from lerobot.policies.smolvla import modeling_smolvla  # type: ignore
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
print("lerobot:", torch.__version__)
print("smolvla import:", modeling_smolvla.__name__)
print("has VLAFlowMatching:", callable(VLAFlowMatching))
PY
then
  log_capture "SmolVLA smoke import" bash -lc "cd / && PYTHONPATH='${smolvla_site_pkg}' '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -c 'import torch; from lerobot.policies.smolvla import modeling_smolvla; from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching; print(\"lerobot:\", torch.__version__); print(\"smolvla import:\", modeling_smolvla.__name__); print(\"has VLAFlowMatching:\", callable(VLAFlowMatching))'"
else
  log_capture "SmolVLA smoke import (failed first pass)" bash -lc "cd / && PYTHONPATH='${smolvla_site_pkg}' '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -c 'import torch; from lerobot.policies.smolvla import modeling_smolvla; from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching; print(\"lerobot:\", torch.__version__); print(\"smolvla import:\", modeling_smolvla.__name__); print(\"has VLAFlowMatching:\", callable(VLAFlowMatching))'"
  log_warn "SmolVLA smoke import failed; installing dependency fixes"
  if command -v uv >/dev/null 2>&1; then
    log_capture "SmolVLA dependency repair (uv)" bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && uv pip install transformers diffusers==0.35.1 accelerate safetensors sentencepiece pillow pyarrow"
  else
    log_capture "SmolVLA dependency repair (pip)" bash -lc "'${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install --upgrade transformers diffusers==0.35.1 accelerate safetensors sentencepiece pillow pyarrow"
  fi
  log_capture "SmolVLA smoke import (retry)" bash -lc "cd / && PYTHONPATH='${smolvla_site_pkg}' '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -c 'import torch; from lerobot.policies.smolvla import modeling_smolvla; from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching; print(\"lerobot:\", torch.__version__); print(\"smolvla import:\", modeling_smolvla.__name__); print(\"has VLAFlowMatching:\", callable(VLAFlowMatching))'"
  if ! run_env_python_siteclean "${SMOLVLA_LEROBOT_ENV_DIR}" - <<'PY'
import torch
from lerobot.policies.smolvla import modeling_smolvla  # type: ignore
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
print("lerobot:", torch.__version__)
print("smolvla import:", modeling_smolvla.__name__)
print("has VLAFlowMatching:", callable(VLAFlowMatching))
PY
then
  append_passfail "FAIL" "SmolVLA smoke import still failing after repairs"
  append_decision "phase03 SmolVLA install" "smolvla repair did not satisfy runtime import contract" "FAIL"
  return 1
fi
fi

snapshot_env "${SMOLVLA_LEROBOT_ENV_DIR}" "lerobot_mw"
append_passfail "PASS" "SmolVLA base stack smoke checks succeeded"
append_decision "phase03 SmolVLA install" "build flow policy runtime foundation" "PASS"
}

ensure_smolvla_stack() {
  if run_env_python_siteclean "${SMOLVLA_LEROBOT_ENV_DIR}" - <<'PY'
import torch
from lerobot.policies.smolvla import modeling_smolvla  # type: ignore
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
print("smolvla-stack-ok")
PY
  then
    log_info "SmolVLA stack probe succeeded in ${SMOLVLA_LEROBOT_ENV_DIR}"
    return 0
  fi

  log_warn "SmolVLA stack unavailable; triggering phase03_smolvla_install repair flow"
  local prev_report="${REPORT_FILE}"
  REPORT_FILE="${SMOLVLA_REPORT_ROOT}/stage03_smolvla_install_status.md"
  if ! stage03_smolvla_install; then
    REPORT_FILE="${prev_report}"
    log_error "SmolVLA stack repair failed"
    return 1
  fi
  REPORT_FILE="${prev_report}"
  return 0
}

stage04_metaworld_install() {
  log_info "phase04_metaworld_install: install/verify Meta-World and push-v3"
  cat >"${REPORT_FILE}" <<EOF
# phase04_metaworld_install
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
if [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  log_error "missing env ${SMOLVLA_LEROBOT_ENV_DIR}; run phase01 first"
  exit 1
fi

source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
if command -v uv >/dev/null 2>&1; then
  log_capture "Meta-World install (uv)" bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && uv pip install \"metaworld>=0.1.0\""
else
  log_capture "Meta-World install (pip)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m ensurepip --upgrade --default-pip && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -m pip install metaworld>=0.1.0"
fi
log_capture "push-v3 env probe" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import metaworld

ml1 = metaworld.ML1("push-v3")
env = ml1.train_classes["push-v3"]()
print("env class:", env.__class__.__name__)
print("has reset:", hasattr(env, "reset"))
env.close()
PY
append_passfail "PASS" "Meta-World install + push-v3 reset smoke passed"
append_decision "phase04 Meta-World install" "required for baseline and rollout collection" "PASS"
}

stage05_model_pull() {
  log_info "phase05_model_pull: pull and verify HF checkpoint"
  cat >"${REPORT_FILE}" <<EOF
# phase05_model_pull
Checkpoint: ${SMOLVLA_INIT_CHECKPOINT}
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

if [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  log_error "missing env ${SMOLVLA_LEROBOT_ENV_DIR}; run phase01 first"
  exit 1
fi

source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
log_capture "Hugging Face snapshot download" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="${SMOLVLA_INIT_CHECKPOINT}", allow_patterns="*")
print(path)
PY

append_passfail "PASS" "checkpoint snapshot downloaded"
append_decision "phase05 model pull" "baseline policy contract fixed to init checkpoint" "PASS"
}

stage06_baseline_eval() {
  log_info "phase06_baseline_eval: run push-v3 baseline packet"
  cat >"${REPORT_FILE}" <<EOF
# phase06_baseline_eval
Episodes: ${SMOLVLA_BASELINE_EPISODES}
Device: ${SMOLVLA_BASELINE_DEVICE}
Video: ${SMOLVLA_BASELINE_VIDEO} (len=${SMOLVLA_BASELINE_VIDEO_LENGTH}, interval=${SMOLVLA_BASELINE_VIDEO_INTERVAL})
Seed: ${SMOLVLA_BASELINE_SEED}
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
  require_slurm_gpu_stage "phase06_baseline_eval"

cmd="${SMOLVLA_BASELINE_CMD:-}"
if [[ -z "${cmd}" ]]; then
  if [[ -x "${SMOLVLA_PIPE_ROOT}/run_baseline_eval.sh" ]]; then
    cmd="bash '${SMOLVLA_PIPE_ROOT}/run_baseline_eval.sh' \
      --episodes '${SMOLVLA_BASELINE_EPISODES}' \
      --device '${SMOLVLA_BASELINE_DEVICE}' \
      --seed '${SMOLVLA_BASELINE_SEED}' \
      --video '${SMOLVLA_BASELINE_VIDEO}' \
      --video-length '${SMOLVLA_BASELINE_VIDEO_LENGTH}' \
      --video-interval '${SMOLVLA_BASELINE_VIDEO_INTERVAL}'"
  else
    append_warn="1"
    echo "No SMOLVLA_BASELINE_CMD provided." >>"${REPORT_FILE}"
    echo "Set SMOLVLA_BASELINE_CMD with a concrete evaluator command." >>"${REPORT_FILE}"
    append_passfail "PASS" "baseline command was intentionally skipped (placeholder)"
    append_decision "phase06 baseline eval" "waiting for evaluator entrypoint" "SKIP/placeholder"
    return 0
  fi
fi
  baseline_log="${SMOLVLA_REPORT_ROOT}/phase06_baseline_eval.log"
  if ! log_capture "Baseline push-v3 run" bash -lc "cd ${SMOLVLA_REPO_ROOT} && ${cmd} > ${baseline_log} 2>&1"; then
    append_passfail "FAIL" "baseline command failed"
    append_decision "phase06 baseline eval" "reproducibility gate before JEPA/VGG stages" "FAIL"
    return 1
  fi
  append_passfail "PASS" "baseline command executed"
  baseline_output_dir="$(sed -n 's/^Baseline eval output directory: //p' "${baseline_log}" | tail -n 1)"
  if [[ -n "${baseline_output_dir}" && -d "${baseline_output_dir}" ]]; then
    echo "Baseline output dir: ${baseline_output_dir}" >>"${REPORT_FILE}"
    append_decision "phase06 baseline eval" "baseline artifact path captured: ${baseline_output_dir}" "PASS"
    eval_info_path="${baseline_output_dir}/eval_info.json"
    if [[ -f "${eval_info_path}" ]]; then
      if eval_summary="$(SMOLVLA_EVAL_INFO_PATH="${eval_info_path}" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SMOLVLA_EVAL_INFO_PATH"])
obj = json.loads(path.read_text(encoding="utf-8"))
overall = obj.get("overall", {})
print(
    str(int(overall.get("n_episodes", -1)))
    + "|"
    + str(overall.get("pc_success", ""))
    + "|"
    + str(overall.get("avg_sum_reward", ""))
    + "|"
    + str(overall.get("avg_max_reward", ""))
    + "|"
    + str(overall.get("eval_ep_s", ""))
)
PY
      )"; then
        IFS='|' read -r baseline_episodes baseline_success baseline_avg_sum baseline_avg_max baseline_eval_ep_s <<< "${eval_summary}"
        echo "Baseline eval summary (n=${baseline_episodes}, success=${baseline_success}, avg_sum_reward=${baseline_avg_sum}, avg_max_reward=${baseline_avg_max}, avg_eval_s_per_ep=${baseline_eval_ep_s})" >>"${REPORT_FILE}"
        if [[ -z "${baseline_episodes}" ]]; then
          append_passfail "WARN" "baseline eval_info.json could not be parsed for episode count"
          append_decision "phase06 baseline eval" "artifact parse incomplete" "WARN"
        else
          append_passfail "PASS" "baseline eval_info parsed (${baseline_episodes} episodes)"
          append_decision "phase06 baseline eval" "reproducibility gate before JEPA/VGG stages" "PASS"
        fi
      else
        append_passfail "WARN" "baseline eval_info.json present but not readable"
        append_decision "phase06 baseline eval" "artifact parse failed" "WARN"
      fi
    else
      append_passfail "WARN" "baseline eval_info.json not found"
      append_decision "phase06 baseline eval" "baseline completed without eval metrics" "WARN"
    fi
  else
    echo "Baseline output dir not found in ${baseline_log}" >>"${REPORT_FILE}"
    append_passfail "WARN" "baseline output path unresolved"
    append_decision "phase06 baseline eval" "baseline succeeded without captured artifact path" "WARN"
  fi
}

stage07_jepa_setup() {
  log_info "phase07_jepa_setup: install JEPA-WM and smoke latent unroll"
  cat >"${REPORT_FILE}" <<EOF
# phase07_jepa_setup
Task: ${SMOLVLA_JEPA_TASK}
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
  require_slurm_gpu_stage "phase07_jepa_setup"

if [[ ! -x "${SMOLVLA_JEPA_ENV_DIR}/bin/python" ]]; then
  log_error "missing env ${SMOLVLA_JEPA_ENV_DIR}; run phase01 first"
  exit 1
fi

  source "${SMOLVLA_JEPA_ENV_DIR}/bin/activate"
  install_failed=0
  jepa_py_version="$("${SMOLVLA_JEPA_ENV_DIR}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
  )"
  install_attempted=0
  if [[ "${jepa_py_version}" != 3.10* ]]; then
    log_warn "JEPA env Python ${jepa_py_version} incompatible with JEPA-WM pyproject constraints; skipping editable install and using torch.hub fallback"
    install_failed=0
    install_attempted=0
  elif command -v uv >/dev/null 2>&1; then
    install_attempted=1
    if ! bash -c "cd '${SMOLVLA_VGG_REPO}/jepa-wms' && uv pip install -e ."; then
      log_warn "JEPA install (uv) failed; attempting pip fallback in jepa env"
      install_failed=1
    else
      log_info "JEPA install (uv) succeeded"
    fi
  else
    install_failed=1
    install_attempted=0
  fi
  if (( install_failed == 1 )); then
    if (( install_attempted == 1 )); then
      if ! bash -c "cd '${SMOLVLA_VGG_REPO}/jepa-wms' && '${SMOLVLA_JEPA_ENV_DIR}/bin/python' -m ensurepip --upgrade --default-pip && '${SMOLVLA_JEPA_ENV_DIR}/bin/python' -m pip install -e ."; then
        log_warn "JEPA install (pip fallback) failed; continuing with torch.hub load from cached assets"
      else
        log_capture "JEPA install (pip fallback)" bash -c "cd '${SMOLVLA_VGG_REPO}/jepa-wms' && '${SMOLVLA_JEPA_ENV_DIR}/bin/python' -m pip install -e ."
      fi
    else
      log_warn "JEPA editable install skipped due environment/tooling constraints"
    fi
  fi

  # Prefer the leRobot env for smoke checks because it already has the required torch build.
  local smoke_py="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
  if [[ ! -x "${smoke_py}" ]]; then
    smoke_py="${SMOLVLA_JEPA_ENV_DIR}/bin/python"
  fi
  smoke_cuda_available="$("${smoke_py}" - <<'PY'
import torch
print(int(torch.cuda.is_available()))
PY
)"

  if [[ "${SMOLVLA_JEPA_SMOKE_FORCE_VALIDATE:-0}" != "1" && "${smoke_cuda_available}" != "1" ]]; then
    append_passfail "WARN" "JEPA smoke unroll skipped due to no CUDA; keeping stage dependency setup only."
    append_decision "phase07 JEPA setup" "smoke rollout blocked by missing CUDA" "WARN"
    {
      echo ""
      echo "## SmolVLA/VJepa smoke status"
      echo "- status: SKIPPED_NO_CUDA"
      echo "- message: smoke unroll requires CUDA acceleration."
      echo "- checked_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    } >>"${REPORT_FILE}"
    return 0
  fi

  if [[ -x "${smoke_py}" ]] && [[ -f "${SMOLVLA_PIPE_ROOT}/jepa_smoke_check.py" ]]; then
    JEPA_LOG_BASE="${SMOLVLA_CACHE_ROOT}/jepa_workflow"
    if [[ ! -d "${JEPA_LOG_BASE}" ]]; then
      JEPA_LOG_BASE="/tmp"
    fi
    log_capture "JEPA smoke unroll" bash -c "
      env \
        JEPAWM_LOGS='${JEPA_LOG_BASE}' \
        JEPAWM_CKPT='${JEPA_LOG_BASE}' \
        '${smoke_py}' '${SMOLVLA_PIPE_ROOT}/jepa_smoke_check.py' \
          --repo '${SMOLVLA_VGG_REPO}/jepa-wms' \
          --ckpt '${SMOLVLA_JEPA_CKPT}' \
          --task '${SMOLVLA_JEPA_TASK}' \
          --pretrained \
          --smoke-steps '${SMOLVLA_JEPA_SMOKE_STEPS}' \
          --device '${SMOLVLA_BASELINE_DEVICE}'
    "
  else
    log_capture "JEPA model smoke (fallback)" bash -c "
${smoke_py} <<'PY'
import torch
print("torch:", torch.__version__)
try:
    import metaworld
    print("metaworld:", metaworld.__version__)
except Exception as exc:
    print("metaworld import error:", exc)
PY
    "
  fi
  append_passfail "PASS" "JEPA-WMS environment prepared"
  append_decision "phase07 JEPA setup" "required differentiable latent objective for VGG gating" "PASS"

  if [[ "${SMOLVLA_JEPA_EXPORT_ENABLED:-0}" == "1" ]]; then
    if [[ "${SMOLVLA_JEPA_SMOKE_FORCE_VALIDATE:-0}" != "1" && "${smoke_cuda_available}" != "1" ]]; then
      append_passfail "WARN" "JEPA rollout export skipped (no CUDA in probe env; set SMOLVLA_JEPA_SMOKE_FORCE_VALIDATE=1 to override)"
      append_decision "phase07 JEPA export" "export requires GPU allocation for consistent pipeline" "WARN"
    elif [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
      append_passfail "WARN" "JEPA rollout export skipped (missing lerobot python)"
    elif [[ ! -f "${SMOLVLA_PIPE_ROOT}/jepa_cem_paired_pushv3_export.py" ]]; then
      append_passfail "WARN" "jepa_cem_paired_pushv3_export.py missing"
    elif ! "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" -c "import metaworld" >/dev/null 2>&1; then
      append_passfail "FAIL" "metaworld not installed in lerobot env; run phase04_metaworld_install (run_stage.sh 4) before export"
      append_decision "phase07 JEPA export" "missing metaworld dependency" "FAIL"
      return 1
    else
      mkdir -p "${SMOLVLA_JEPA_EXPORT_OUT}"
      site_packages="$("${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import site
print(":".join(site.getsitepackages()))
PY
)"
      if log_capture "JEPA CEM paired push-v3 trajectory export" bash -c "
        set -euo pipefail
        export LEAKY=1 MH_REPO=metaworld-v2 METAWORLD_RENDER_MODE=rgb_array
        xvfb-run -a -s '-screen 0 1280x1024x24' env \
          PYTHONPATH='${site_packages}:'\"\${PYTHONPATH:-}\" \
          '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' '${SMOLVLA_PIPE_ROOT}/jepa_cem_paired_pushv3_export.py' \
            --task '${SMOLVLA_JEPA_TASK}' \
            --episodes '${SMOLVLA_JEPA_EXPORT_EPISODES}' \
            --max-steps '${SMOLVLA_JEPA_EXPORT_MAX_STEPS}' \
            --seed '${SMOLVLA_JEPA_EXPORT_SEED}' \
            --out '${SMOLVLA_JEPA_EXPORT_OUT}' \
            --jepa-repo '${SMOLVLA_VGG_REPO}/jepa-wms' \
            --jepa-ckpt '${SMOLVLA_JEPA_CKPT}' \
            --cem-horizon '${SMOLVLA_JEPA_CEM_HORIZON}' \
            --cem-pop '${SMOLVLA_JEPA_CEM_POP}' \
            --cem-iters '${SMOLVLA_JEPA_CEM_ITERS}' \
            --device '${SMOLVLA_BASELINE_DEVICE}'
      "; then
        append_passfail "PASS" "JEPA rollout export written under ${SMOLVLA_JEPA_EXPORT_OUT}"
        append_decision "phase07 JEPA export" "trajectories.pt for bridge_builder" "PASS"
      else
        append_passfail "FAIL" "JEPA rollout export command failed"
        append_decision "phase07 JEPA export" "metaworld export failed" "FAIL"
        return 1
      fi
    fi
  fi
}

stage08_bridge_design() {
  log_info "phase08_bridge_design: build synthetic bridge dataset"
  cat >"${REPORT_FILE}" <<EOF
# phase08_bridge_design
Task: ${SMOLVLA_JEPA_TASK}
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

output_dir="${SMOLVLA_DATA_ROOT}"
mkdir -p "${output_dir}" "${SMOLVLA_DATA_ROOT}/val"

if ! check_jepa_source_has_trajectory_artifacts "${SMOLVLA_JEPA_SOURCE}"; then
  status=$?
  if [[ "${status}" == "2" ]]; then
    log_error "SMOLVLA_JEPA_SOURCE appears to be a checkpoint file: ${SMOLVLA_JEPA_SOURCE}"
    echo "- [FAIL] SMOLVLA_JEPA_SOURCE should be a trajectory artifact path (directory or file), not a checkpoint" >>"${REPORT_FILE}"
    append_decision "phase08 bridge source check" "blocked because source is checkpoint-only" "FAIL"
  else
    log_error "SMOLVLA_JEPA_SOURCE has no trajectory artifacts: ${SMOLVLA_JEPA_SOURCE}"
    echo "- [FAIL] SMOLVLA_JEPA_SOURCE produced no usable trajectory files (json/jsonl/npz/pth/pt/pickle/pkl) during precheck" >>"${REPORT_FILE}"
    append_decision "phase08 bridge source check" "blocked because source contains no trajectory artifacts" "FAIL"
  fi
  return 1
fi

if [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  append_passfail "FAIL" "lerobot env python missing at ${SMOLVLA_LEROBOT_ENV_DIR}; cannot run bridge_builder"
  append_decision "phase08 bridge design" "lerobot env not executable" "FAIL"
  return 1
fi
if [[ ! -f "${SMOLVLA_PIPE_ROOT}/bridge_builder.py" ]]; then
  append_passfail "FAIL" "bridge_builder.py not found at ${SMOLVLA_PIPE_ROOT}/bridge_builder.py"
  append_decision "phase08 bridge design" "missing bridge_builder.py" "FAIL"
  return 1
fi
source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
if ! log_capture "Bridge conversion" bash -lc "python '${SMOLVLA_PIPE_ROOT}/bridge_builder.py' \
  --jepa-source '${SMOLVLA_JEPA_SOURCE}' \
  --out-dir '${SMOLVLA_DATA_ROOT}' \
  --train-ratio '${SMOLVLA_BRIDGE_TRAIN_RATIO}' \
  --min-confidence '${SMOLVLA_BRIDGE_MIN_CONFIDENCE}' \
  --val-ratio '${SMOLVLA_BRIDGE_VAL_RATIO}' \
  --min-action-length '${SMOLVLA_BRIDGE_MIN_ACTION_LEN}'"; then
  append_passfail "FAIL" "bridge_builder.py exited non-zero (see Bridge conversion log above)"
  append_decision "phase08 bridge design" "bridge conversion command failed" "FAIL"
  return 1
fi

if [[ -f "${SMOLVLA_DATA_ROOT}/bridge_summary.json" ]]; then
  summary_counts="$(read_bridge_summary_counts "${SMOLVLA_DATA_ROOT}/bridge_summary.json")"
  read -r train_records val_records empty_inputs <<< "${summary_counts}"
  if [[ "${train_records}" == "" || "${val_records}" == "" ]]; then
    append_passfail "FAIL" "bridge summary parsing failed at ${SMOLVLA_DATA_ROOT}/bridge_summary.json"
    append_decision "phase08 bridge design" "bridge summary malformed" "FAIL"
    return 1
  fi
  if (( train_records + val_records == 0 )); then
    append_passfail "FAIL" "bridge summary is empty (train=0, val=0); no trajectory data ingested"
    append_decision "phase08 bridge design" "empty dataset blocks stage10 training" "FAIL"
    return 1
  fi
  if (( val_records == 0 )); then
    append_passfail "WARN" "bridge summary indicates val=0; stageB may receive no synthetic mix"
  else
    append_passfail "PASS" "bridge summary generated at ${SMOLVLA_DATA_ROOT}/bridge_summary.json"
  fi
else
  append_passfail "FAIL" "bridge_summary.json missing under ${SMOLVLA_DATA_ROOT} after bridge_builder (unexpected)"
  append_decision "phase08 bridge design" "missing bridge summary" "FAIL"
  return 1
fi
append_decision "phase08 bridge design" "bridge required for mixed synthetic training" "PASS"
}

stage09_vgg_gates() {
  log_info "phase09_vgg_gates: validate VGG prerequisites"
  cat >"${REPORT_FILE}" <<EOF
# phase09_vgg_gates
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
  require_slurm_gpu_stage "phase09_vgg_gates"

if [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  log_error "missing env ${SMOLVLA_LEROBOT_ENV_DIR}; run phase01 first"
  exit 1
fi

source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
  if ! ensure_smolvla_stack; then
    append_passfail "FAIL" "smolvla dependency bootstrap failed before gate checks"
    append_decision "phase09 VGG checks" "smolvla dependency repair failed" "FAIL"
    exit 1
  fi
if [[ -f "${SMOLVLA_PIPE_ROOT}/validate_smolvla_vgg_gates.py" ]]; then
  gate_output="${SMOLVLA_VGG_GATE_JSON}"
  gate_ok="0"
  smolvla_site_pkg="$("${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  cuda_available="$("${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import torch
print(int(torch.cuda.is_available()))
PY
)"
  vgg_skip_flow_check=""
  if [[ "${SMOLVLA_VGG_GATE_SKIP_FLOW_CHECK:-0}" == "1" ]]; then
    vgg_skip_flow_check="--skip-flow-check"
  elif [[ "${SMOLVLA_BASELINE_DEVICE}" == "cpu" || "${SMOLVLA_BASELINE_DEVICE}" == "auto" && "${cuda_available}" != "1" ]]; then
    vgg_skip_flow_check="--skip-flow-check"
  fi
  if [[ "${SMOLVLA_VGG_GATE_FORCE_VALIDATE:-0}" != "1" && ("${SMOLVLA_BASELINE_DEVICE}" == "cpu" || "${SMOLVLA_BASELINE_DEVICE}" == "auto" && "${cuda_available}" != "1") ]]; then
    cat >"${gate_output}" <<EOF
{
  "schema_version": "smolvla_gate_v1",
  "emit_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "init_checkpoint": "${SMOLVLA_INIT_CHECKPOINT}",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "velocity_trace_ok": false,
  "contract_ok": false,
  "contract_reasons": [
    "validation_skipped_no_cuda"
  ],
  "velocity_trace_skipped": true,
  "velocity_shape": [],
  "base_flow_diff_max": 0.0,
  "base_flow_diff_mean": 0.0,
  "value_head_ok": false,
  "value_head_grad_norm": 0.0,
  "value_head_value": 0.0,
  "device": "${SMOLVLA_BASELINE_DEVICE}",
  "value_head_grad_ok": false,
  "base_flow_ok": false,
  "gate_ok": false,
  "gate_reasons": [
    "validation_skipped_no_cuda"
  ],
  "skipped_for_cpu": true,
  "skip_flow_check": true
}
EOF
    append_passfail "WARN" "VGG gate validation skipped due to no CUDA; precomputed artifact emitted."
  elif log_capture "SmolVLA velocity/value-gate validation" bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && \
    cd / && \
    PYTHONPATH='${smolvla_site_pkg}' \
    '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' \
    '${SMOLVLA_PIPE_ROOT}/validate_smolvla_vgg_gates.py' \
    --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' \
    --episodes 2 \
    --steps 6 \
    --device '${SMOLVLA_BASELINE_DEVICE}' \
    --output '${gate_output}' \
    ${vgg_skip_flow_check} \
    --emit-trace \
    --trace-max-steps '${SMOLVLA_VGG_TRACE_MAX_STEPS}' \
    --trace-max-batch '${SMOLVLA_VGG_TRACE_MAX_BATCH}' \
    --value-head-grad-min '${SMOLVLA_VGG_GATE_MIN_VALUE_GRAD}' \
    --max-base-flow-diff '${SMOLVLA_VGG_GATE_MAX_BASE_FLOW_DIFF}'"; then
    if [[ -f "${gate_output}" ]]; then
    gate_ok="$(python - <<PY
import json
from pathlib import Path
path = Path("${gate_output}")
if not path.exists():
    print("0")
else:
    data = json.loads(path.read_text(encoding="utf-8"))
    print("1" if bool(data.get("gate_ok", False)) else "0")
PY
)"
    else
      gate_ok="0"
      log_warn "VGG gate JSON missing; command may have terminated before reporting."
      if log_capture "Velocity field check" bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && \
        cd / && \
        PYTHONPATH='${smolvla_site_pkg}' \
        '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -c 'import inspect; from lerobot.policies.smolvla import modeling_smolvla; print(\"has VLAFlowMatching:\", hasattr(modeling_smolvla, \"VLAFlowMatching\")); print(\"VLAFlowMatching methods:\", [m for m in (\"forward\", \"sample_actions\") if hasattr(modeling_smolvla.VLAFlowMatching, m)])'"; then
          :
      else
          log_warn "fallback velocity check failed; skipping introspection."
      fi
      append_passfail "WARN" "Fallback velocity check used; full gate telemetry unavailable"
    fi
else
  gate_ok="0"
  log_warn "validate_smolvla_vgg_gates.py execution failed; fallback check enabled."
  if log_capture "Velocity field check" bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && \
    cd / && \
    PYTHONPATH='${smolvla_site_pkg}' \
    '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -c 'import inspect; from lerobot.policies.smolvla import modeling_smolvla; print(\"has VLAFlowMatching:\", hasattr(modeling_smolvla, \"VLAFlowMatching\")); print(\"VLAFlowMatching methods:\", [m for m in (\"forward\", \"sample_actions\") if hasattr(modeling_smolvla.VLAFlowMatching, m)])'"; then
      :
  else
      log_warn "fallback velocity check failed; skipping introspection."
  fi
  append_passfail "WARN" "Fallback velocity check used; full gate telemetry unavailable"
fi
  if [[ "${gate_ok}" == "1" ]]; then
    append_passfail "PASS" "VGG gate evaluation passed"
  else
    append_passfail "WARN" "VGG gate evaluation failed; StageC may be skipped"
    append_decision "phase09 VGG checks" "value/head-flow gates indicate auxiliary training disabled" "WARN"
  fi
else
  if log_capture "Velocity field check" bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && \
    cd / && \
    PYTHONPATH='${smolvla_site_pkg}' \
    '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' -c 'import inspect; from lerobot.policies.smolvla import modeling_smolvla; print(\"has VLAFlowMatching:\", hasattr(modeling_smolvla, \"VLAFlowMatching\")); print(\"VLAFlowMatching methods:\", [m for m in (\"forward\", \"sample_actions\") if hasattr(modeling_smolvla.VLAFlowMatching, m)])'"; then
    :
  else
    log_warn "fallback velocity check failed; skipping introspection."
  fi
  append_passfail "WARN" "Fallback velocity check used; full gate telemetry unavailable"
fi

if [[ "${gate_output:-}" == "" || ! -f "${gate_output}" || "${gate_ok:-}" != "1" ]]; then
  echo "Gates not fully passed; jobs with SMOLVLA_ENABLE_VGG=1 stop here. Dedicated SMOLVLA_TRAIN_VARIANT=c (stage08) still runs phase10, which re-validates ${SMOLVLA_VGG_GATE_JSON} before training." >>"${REPORT_FILE}"
  append_decision "phase09 VGG checks" "gate prerequisites did not pass" "WARN"
  if [[ "${SMOLVLA_ENABLE_VGG:-0}" == "1" ]]; then
    append_passfail "FAIL" "SMOLVLA_ENABLE_VGG=1 requires gate pass before StageC"
    return 1
  fi
else
  echo "All VGG gates passed." >>"${REPORT_FILE}"
  append_decision "phase09 VGG checks" "enable value-guided stage after gating criteria" "PASS"
fi
}

stage10_train_loop() {
  log_info "phase10_train_loop: staged training sequence"
  cat >"${REPORT_FILE}" <<EOF
# phase10_train_loop
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
  require_slurm_gpu_stage "phase10_train_loop"

  local run_id run_variant
  local train_variant="${SMOLVLA_TRAIN_VARIANT:-auto}"
  run_variant="${train_variant}"
  run_id="$(smolvla_run_id)"
  log_info "stage10 run_id=${run_id}, variant=${run_variant}"

  log_capture "Stage10 torch/CUDA probe" bash -lc "
    source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate'
    export PYTHONPATH=''
    '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY
  "
  if ! ensure_smolvla_stack; then
    append_passfail "FAIL" "smolvla dependency bootstrap failed before staged training"
    append_decision "phase10 train loop" "smolvla dependency repair failed" "FAIL"
    exit 1
  fi

  local stage10_root="${SMOLVLA_ARTIFACT_ROOT}/stage10_${run_variant}_${run_id}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    stage10_root="${stage10_root}_job${SLURM_JOB_ID}"
  fi
  local stage10_stageA_root="${stage10_root}/stageA"
  local stage10_stageB_root="${stage10_root}/stageB"
  local stage10_stageC_root="${stage10_root}/stageC"
  local stage10_stageD_root="${stage10_root}/stageD"
  mkdir -p \
    "${stage10_stageA_root}" \
    "${stage10_stageB_root}" \
    "${stage10_stageB_root}/jepa_mix" \
    "${stage10_stageC_root}" \
    "${stage10_stageC_root}/vgg_aux" \
    "${stage10_stageD_root}" \
    "${stage10_stageD_root}/vgg_aux_imagined"

  emit_stage_manifest() {
    local stage_label="$1"
    local stage_output_dir="$2"
    local stage_cmd="$3"
    local stage_dataset="$4"
    local manifest_path="${stage_output_dir}/run_manifest.json"
    emit_run_manifest \
      "${manifest_path}" \
      "stage10_train_loop" \
      "${run_id}" \
      "${run_variant}" \
      "${stage_label}" \
      "${stage_output_dir}" \
      "${SMOLVLA_INIT_CHECKPOINT}" \
      "${stage_cmd}" \
      "${stage_dataset}" \
      "${SMOLVLA_FIRST_FT_STEPS}" \
      "${SMOLVLA_TRAIN_LOG_STEPS}" \
      "${SMOLVLA_TRAIN_SAVE_STEPS}"
  }

  smolvla_read_vgg_gate_ok() {
    local ok="0"
    if [[ -f "${SMOLVLA_VGG_GATE_JSON}" ]]; then
      ok="$(
        SMOLVLA_VGG_GATE_JSON="${SMOLVLA_VGG_GATE_JSON}" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SMOLVLA_VGG_GATE_JSON"])
if not path.exists():
    print(0)
else:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        gate_ok = bool(payload.get("gate_ok", False))
        contract_ok = bool(payload.get("contract_ok", True))
        ic = str(payload.get("init_checkpoint", "")).strip()
        want = str(os.environ.get("SMOLVLA_INIT_CHECKPOINT", "")).strip()
        if ic and want and ic != want:
            gate_ok = False
        print(1 if (gate_ok and contract_ok) else 0)
    except Exception:
        print(0)
PY
      )"
    fi
    printf '%s' "${ok}"
  }

STAGE_A_CMD="${SMOLVLA_TRAIN_STAGE_A_CMD:-}"
STAGE_B_CMD="${SMOLVLA_TRAIN_STAGE_B_CMD:-}"
STAGE_C_CMD="${SMOLVLA_TRAIN_STAGE_C_CMD:-}"
STAGE_D_CMD="${SMOLVLA_TRAIN_STAGE_D_CMD:-}"
TRAIN_REPORT_TO_ARG=""
TRAIN_OUT_ROOT="${stage10_root}"
TRAIN_OUT_DIR_A="${TRAIN_OUT_ROOT}/stageA"
TRAIN_OUT_DIR_B="${TRAIN_OUT_ROOT}/stageB"
TRAIN_OUT_DIR_C="${TRAIN_OUT_ROOT}/stageC"
TRAIN_OUT_DIR_D="${TRAIN_OUT_ROOT}/stageD"
TRAIN_LOG_STEP_ARG=" --log-steps '${SMOLVLA_TRAIN_LOG_STEPS}'"
TRAIN_SAVE_STEP_ARG=" --save-steps '${SMOLVLA_TRAIN_SAVE_STEPS}'"
if [[ -n "${SMOLVLA_TRAIN_REPORT_TO:-}" ]]; then
  TRAIN_REPORT_TO_ARG=" --policy-report-to '${SMOLVLA_TRAIN_REPORT_TO}'"
fi
if [[ "${SMOLVLA_TRAIN_DRY_RUN}" == "1" ]]; then
  DRY_FLAG="--dry-run"
else
  DRY_FLAG=""
fi
if [[ -z "${STAGE_A_CMD}" && -f "${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py" ]]; then
  STAGE_A_CMD="'${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' '${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py' --mode stageA --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' --lerobot-env '${SMOLVLA_LEROBOT_ENV_DIR}' --real-data-root '${SMOLVLA_DATA_ROOT}/train' --max-steps '${SMOLVLA_FIRST_FT_STEPS}' --output-dir '${TRAIN_OUT_DIR_A}'${TRAIN_REPORT_TO_ARG}${TRAIN_LOG_STEP_ARG}${TRAIN_SAVE_STEP_ARG} ${DRY_FLAG}"
fi
if [[ -z "${STAGE_B_CMD}" && -f "${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py" ]]; then
  STAGE_B_CMD="'${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' '${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py' --mode stageB --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' --lerobot-env '${SMOLVLA_LEROBOT_ENV_DIR}' --real-data-root '${SMOLVLA_DATA_ROOT}/train' --jepa-data-root '${SMOLVLA_DATA_ROOT}/val' --max-steps '${SMOLVLA_FIRST_FT_STEPS}' --output-dir '${TRAIN_OUT_DIR_B}'${TRAIN_REPORT_TO_ARG}${TRAIN_LOG_STEP_ARG}${TRAIN_SAVE_STEP_ARG} ${DRY_FLAG}"
fi
if [[ -z "${STAGE_C_CMD}" && -f "${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py" ]]; then
  STAGE_C_CMD="'${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' '${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py' --mode stageC --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' --lerobot-env '${SMOLVLA_LEROBOT_ENV_DIR}' --real-data-root '${SMOLVLA_DATA_ROOT}/train' --max-steps '${SMOLVLA_FIRST_FT_STEPS}' --output-dir '${TRAIN_OUT_DIR_C}' --gate-json '${SMOLVLA_VGG_GATE_JSON}' --match-weight '${SMOLVLA_VGG_MATCH_WEIGHT}' --match-warmup '${SMOLVLA_VGG_MATCH_WARMUP_STEPS}' --value-head-dim '${SMOLVLA_VGG_VALUE_HEAD_DIM}' --value-head-steps '${SMOLVLA_VGG_VALUE_HEAD_STEPS}' --value-head-seed '${SMOLVLA_VGG_VALUE_HEAD_SEED}' --value-head-min-grad '${SMOLVLA_VGG_VALUE_HEAD_MIN_GRAD}' --trace-cap '${SMOLVLA_VGG_MATCH_TRACE_CAP}'${TRAIN_REPORT_TO_ARG}${TRAIN_LOG_STEP_ARG}${TRAIN_SAVE_STEP_ARG} ${DRY_FLAG}"
fi
if [[ -z "${STAGE_D_CMD}" && -f "${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py" ]]; then
  STAGE_D_CMD="'${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' '${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py' --mode stageD --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' --lerobot-env '${SMOLVLA_LEROBOT_ENV_DIR}' --real-data-root '${SMOLVLA_STAGE_D_DATA_ROOT}' --max-steps '${SMOLVLA_FIRST_FT_STEPS}' --output-dir '${TRAIN_OUT_DIR_D}' --gate-json '${SMOLVLA_VGG_GATE_JSON}' --match-weight '${SMOLVLA_VGG_MATCH_WEIGHT}' --match-warmup '${SMOLVLA_VGG_MATCH_WARMUP_STEPS}' --value-head-dim '${SMOLVLA_VGG_VALUE_HEAD_DIM}' --value-head-steps '${SMOLVLA_VGG_VALUE_HEAD_STEPS}' --value-head-seed '${SMOLVLA_VGG_VALUE_HEAD_SEED}' --value-head-min-grad '${SMOLVLA_VGG_VALUE_HEAD_MIN_GRAD}' --trace-cap '${SMOLVLA_VGG_MATCH_TRACE_CAP}'${TRAIN_REPORT_TO_ARG}${TRAIN_LOG_STEP_ARG}${TRAIN_SAVE_STEP_ARG} ${DRY_FLAG}"
fi

# Reset manifest hooks so a long-lived shell or inherited env cannot stamp skipped stages with old roots.
export SMOLVLA_MANIFEST_STAGE_A_DATA_ROOT=""
export SMOLVLA_MANIFEST_STAGE_B_JEPA_ROOT=""
export SMOLVLA_MANIFEST_STAGE_B_MIXED_ROOT=""
export SMOLVLA_MANIFEST_STAGE_C_DATA_ROOT=""
export SMOLVLA_MANIFEST_STAGE_D_DATA_ROOT=""

if [[ "${train_variant}" == "a" ]]; then
  STAGE_B_CMD=""
  STAGE_C_CMD=""
elif [[ "${train_variant}" == "b" ]]; then
  STAGE_A_CMD=""
  STAGE_C_CMD=""
elif [[ "${train_variant}" == "c" ]]; then
  STAGE_A_CMD=""
  STAGE_B_CMD=""
elif [[ "${train_variant}" == "d" ]]; then
  STAGE_A_CMD=""
  STAGE_B_CMD=""
  STAGE_C_CMD=""
fi

# Variant c is the dedicated StageC Slurm lane; run the VGG auxiliary path even if SMOLVLA_ENABLE_VGG is unset.
# Variant d is TrainD-only: never enter the StageC branch (avoids SMOLVLA_ENABLE_VGG=1 leaked from the environment).
local vgg_run_stage_c="${SMOLVLA_ENABLE_VGG:-0}"
if [[ "${train_variant}" == "c" ]]; then
  vgg_run_stage_c="1"
elif [[ "${train_variant}" == "d" ]]; then
  vgg_run_stage_c="0"
fi

if [[ -z "${STAGE_A_CMD}" ]]; then
  append_warn="1"
  echo "SMOLVLA_TRAIN_STAGE_A_CMD not set. StageA will be skipped." >>"${REPORT_FILE}"
else
  export SMOLVLA_MANIFEST_STAGE_A_DATA_ROOT="${SMOLVLA_DATA_ROOT}/train"
  emit_stage_manifest "stageA" "${TRAIN_OUT_DIR_A}" "${STAGE_A_CMD}" "${SMOLVLA_DATA_ROOT}/train"
  log_capture "StageA (real only)" bash -lc "cd / && ${STAGE_A_CMD}"
fi

if [[ -n "${STAGE_B_CMD}" ]]; then
  export SMOLVLA_MANIFEST_STAGE_B_JEPA_ROOT="${SMOLVLA_DATA_ROOT}/val"
  export SMOLVLA_MANIFEST_STAGE_B_MIXED_ROOT="${TRAIN_OUT_DIR_B}/mixed_lerobot_b"
  emit_stage_manifest "stageB" "${TRAIN_OUT_DIR_B}/jepa_mix" "${STAGE_B_CMD}" "mixed:${SMOLVLA_DATA_ROOT}/train|${SMOLVLA_DATA_ROOT}/val"
  log_capture "StageB (real+JEPA mixed)" bash -lc "cd / && ${STAGE_B_CMD}"
fi

if [[ "${vgg_run_stage_c}" == "1" ]]; then
  if [[ -n "${STAGE_C_CMD}" ]]; then
    gate_ok="$(smolvla_read_vgg_gate_ok)"
    if [[ "${gate_ok}" == "1" ]]; then
      export SMOLVLA_STRICT_VGG_TRAIN=1
      export SMOLVLA_MANIFEST_STAGE_C_DATA_ROOT="${SMOLVLA_DATA_ROOT}/train"
      emit_stage_manifest "stageC" "${TRAIN_OUT_DIR_C}/vgg_aux" "${STAGE_C_CMD}" "${SMOLVLA_DATA_ROOT}/train"
      log_capture "StageC (VGG auxiliary)" bash -lc "cd / && ${STAGE_C_CMD}"
    else
      append_passfail "FAIL" "StageC hard-stop due to gate failure: ${SMOLVLA_VGG_GATE_JSON}"
      append_decision "phase10_train_loop" "vgg gate check failed (variant=${train_variant}, enable_vgg=${SMOLVLA_ENABLE_VGG:-0})" "FAIL"
      return 1
    fi
  else
    append_passfail "FAIL" "StageC VGG path active but SMOLVLA_TRAIN_STAGE_C_CMD is empty."
    append_decision "phase10_train_loop" "StageC command missing while VGG path enabled" "FAIL"
    return 1
  fi
else
  if [[ -n "${STAGE_C_CMD}" ]]; then
    log_warn "StageC command present but VGG StageC path inactive (SMOLVLA_ENABLE_VGG=${SMOLVLA_ENABLE_VGG:-0}, SMOLVLA_TRAIN_VARIANT=${train_variant})."
  fi
  echo "StageC (VGG aux) skipped: path inactive for this job (variant=${train_variant}, SMOLVLA_ENABLE_VGG=${SMOLVLA_ENABLE_VGG:-0})." >>"${REPORT_FILE}"
fi

local run_train_d="0"
if [[ "${train_variant}" == "d" || "${SMOLVLA_ENABLE_TRAIN_D:-0}" == "1" ]]; then
  run_train_d="1"
fi
if [[ "${run_train_d}" == "1" ]]; then
  if [[ -n "${STAGE_D_CMD}" ]]; then
    gate_ok_d="$(smolvla_read_vgg_gate_ok)"
    if [[ "${gate_ok_d}" == "1" ]]; then
      export SMOLVLA_STRICT_VGG_TRAIN=1
      export SMOLVLA_MANIFEST_STAGE_D_DATA_ROOT="${SMOLVLA_STAGE_D_DATA_ROOT}"
      emit_stage_manifest "stageD" "${TRAIN_OUT_DIR_D}/vgg_aux_imagined" "${STAGE_D_CMD}" "${SMOLVLA_STAGE_D_DATA_ROOT}"
      log_capture "StageD (VGG aux imagined)" bash -lc "cd / && ${STAGE_D_CMD}"
    else
      append_passfail "FAIL" "StageD hard-stop due to gate failure: ${SMOLVLA_VGG_GATE_JSON}"
      append_decision "phase10_train_loop" "vgg gate check failed for TrainD" "FAIL"
      return 1
    fi
  else
    append_passfail "FAIL" "TrainD requested (variant=${train_variant}, SMOLVLA_ENABLE_TRAIN_D=${SMOLVLA_ENABLE_TRAIN_D:-0}) but SMOLVLA_TRAIN_STAGE_D_CMD is empty."
    append_decision "phase10_train_loop" "StageD command missing" "FAIL"
    return 1
  fi
fi

append_passfail "PASS" "stage loop driver executed"
append_decision "phase10 train loop" "stages sequenced with manifest-first launch" "PASS"
}

stage11_slurm_orchestration() {
  log_info "phase11_slurm_orchestration: generate watcher + DAG launcher"
  cat >"${REPORT_FILE}" <<EOF
# phase11_slurm_orchestration
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
  log_capture "Workflow script check" bash -lc "python3 '${SMOLVLA_PIPE_ROOT}/smolvla_workflow_launcher.py' --help"
  log_capture "Submit helper check" bash -lc "${SMOLVLA_PIPE_ROOT}/submit_workflow.sh --dry-run"
log_capture "Watcher helper check" bash -lc "python3 '${SMOLVLA_PIPE_ROOT}/watch_workflow.py' --help"

append_passfail "PASS" "orchestration helpers validated"
append_decision "phase11 orchestration" "prepares Slurm DAG + watcher for long run" "PASS"
}

stage13_post_train_eval() {
  # Stable path even when this stage is invoked via legacy alias names.
  REPORT_FILE="${SMOLVLA_REPORT_ROOT}/phase13_post_train_eval_status.md"
  log_info "phase13_post_train_eval: push-v3 eval (post-train checkpoint, same protocol as baseline)"
  cat >"${REPORT_FILE}" <<EOF
# phase13_post_train_eval
Checkpoint: ${SMOLVLA_FINAL_EVAL_CHECKPOINT:-${SMOLVLA_INIT_CHECKPOINT}}
Episodes: ${SMOLVLA_BASELINE_EPISODES}
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
  require_slurm_gpu_stage "phase13_post_train_eval"

  local eval_ckpt="${SMOLVLA_FINAL_EVAL_CHECKPOINT:-${SMOLVLA_INIT_CHECKPOINT}}"
  local out_root="${SMOLVLA_ARTIFACT_ROOT}/phase13_posttrain_eval"
  mkdir -p "${out_root}"

  if [[ ! -x "${SMOLVLA_PIPE_ROOT}/run_baseline_eval.sh" ]]; then
    append_passfail "FAIL" "run_baseline_eval.sh missing"
    return 1
  fi
  local cmd="bash '${SMOLVLA_PIPE_ROOT}/run_baseline_eval.sh' \
    --checkpoint '${eval_ckpt}' \
    --episodes '${SMOLVLA_BASELINE_EPISODES}' \
    --device '${SMOLVLA_BASELINE_DEVICE}' \
    --seed '${SMOLVLA_BASELINE_SEED}' \
    --video '${SMOLVLA_BASELINE_VIDEO}' \
    --video-length '${SMOLVLA_BASELINE_VIDEO_LENGTH}' \
    --video-interval '${SMOLVLA_BASELINE_VIDEO_INTERVAL}' \
    --output-root '${out_root}'"
  local blog="${SMOLVLA_REPORT_ROOT}/phase13_post_train_eval.log"
  if ! log_capture "Post-train eval (push-v3)" bash -lc "cd ${SMOLVLA_REPO_ROOT} && ${cmd} > '${blog}' 2>&1"; then
    append_passfail "FAIL" "phase13 post-train eval command failed"
    append_decision "phase13 post-train eval" "evaluator failed" "FAIL"
    return 1
  fi
  append_passfail "PASS" "phase13 post-train eval driver finished"
  append_decision "phase13 post-train eval" "same protocol as phase06; checkpoint=${eval_ckpt}" "PASS"
}

stage12_reporting() {
  log_info "phase12_reporting: final artifact bundle"
  cat >"${REPORT_FILE}" <<EOF
# phase12_reporting
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
  require_slurm_gpu_stage "phase12_reporting"

{
  echo "## Environment"
  echo "- Workspace: ${SMOLVLA_WORKSPACE_ROOT}"
  echo "- Repo: ${SMOLVLA_REPO_ROOT}"
  echo "- init checkpoint: ${SMOLVLA_INIT_CHECKPOINT}"
  echo ""
  echo "## Commands"
  ls "${SMOLVLA_LOG_ROOT}"/* 2>/dev/null | sed "s#^#- #"
  echo ""
  echo "## Locks"
  ls "${SMOLVLA_LOCK_ROOT}" 2>/dev/null | sed "s#^#- #"
} >>"${REPORT_FILE}"
append_passfail "PASS" "final report packaged"
append_decision "phase12 reporting" "close run with reproducibility metadata" "PASS"
}

case "${STAGE}" in
  0|phase00_inventory|stage00_preflight)
    stage00_inventory
    ;;
  1|phase01_env_topology|stage01_install_lerobot_mw)
    stage01_env_topology
    ;;
  2|phase02_gpu_compat|stage02_gpu_compat)
    stage02_gpu_compat
    ;;
  3|phase03_smolvla_install|stage03_install_smolvla)
    stage03_smolvla_install
    ;;
  4|phase04_metaworld_install|stage04_install_metaworld|stage01b_install_metaworld|phase01b_install_metaworld)
    stage04_metaworld_install
    ;;
  5|phase05_model_pull|stage05_model_pull)
    stage05_model_pull
    ;;
  6|phase06_baseline_eval|stage06_baseline_eval|stage02_baseline_pushv3_eval|stage02)
    stage06_baseline_eval
    ;;
  7|phase07_jepa_setup|stage07_jepa_setup|stage03_install_jepa_wms|stage03)
    stage07_jepa_setup
    ;;
  8|phase08_bridge_design|stage08_bridge_design|stage04_bridge_dataset_build)
    stage08_bridge_design
    ;;
  9|phase09_vgg_gates|stage09_vgg_gates|stage07_vgg_gatecheck)
    stage09_vgg_gates
    ;;
  10|phase10_train_loop|stage10_train_loop|stage05_train_stageA_real_only|stage06_train_stageB_jepa_mix|stage08_train_stageC_vgg_aux|stage10_train_stageD_imagined)
    stage10_train_loop
    ;;
  11|phase11_slurm_orchestration|stage11_slurm_orchestration)
    stage11_slurm_orchestration
    ;;
  12|phase12_reporting|stage12_reporting)
    stage12_reporting
    ;;
  13|phase13_post_train_eval|stage13_post_train_eval|stage09_final_eval_and_bundle)
    stage13_post_train_eval
    ;;
  *)
    log_error "Unknown stage: ${STAGE}"
    exit 2
    ;;
esac

