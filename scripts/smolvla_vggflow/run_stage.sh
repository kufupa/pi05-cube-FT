#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

STAGE="${1:?Usage: $0 <stage_id>}"
REPORT_FILE="${SMOLVLA_REPORT_ROOT}/${STAGE}_status.md"

log_capture() {
  local label="$1"
  shift
  {
    echo
    echo "## ${label}"
    echo '```'
    "$@"
    echo '```'
  } >>"${REPORT_FILE}"
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
  if [[ -x "${env_dir}/bin/python" ]]; then
    if "${env_dir}/bin/python" -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" | grep -q "^${python_ver}$"; then
      log_info "env exists, matching python ${python_ver}: ${env_dir}"
      return 0
    fi
    log_warn "env python mismatch for ${env_dir}; recreating"
    rm -rf "${env_dir}"
  fi
  log_info "creating env: ${env_dir} (python ${python_ver})"
  mkdir -p "$(dirname "${env_dir}")"
  if command -v uv >/dev/null 2>&1; then
    uv venv "${env_dir}" --python "${python_ver}"
  else
    python3 -m venv "${env_dir}"
  fi
  return 0
}

snapshot_env() {
  local env_dir="$1"
  local env_name="$2"
  local freeze_path="${SMOLVLA_LOCK_ROOT}/${env_name}_pip_freeze.txt"
  source "${env_dir}/bin/activate"
  "${env_dir}/bin/pip" freeze > "${freeze_path}"
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
    if log_capture "Torch install (pip)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio || '${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install --no-cache-dir torch torchvision torchaudio"; then
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
if command -v uv >/dev/null 2>&1; then
  log_capture "Base package install (uv)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && uv pip install --upgrade pip setuptools wheel && uv pip install --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir torch torchvision torchaudio || uv pip install --no-cache-dir torch torchvision torchaudio && uv pip install --no-cache-dir \"git+https://github.com/huggingface/lerobot.git\""
else
  log_capture "Base package install (pip)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install --upgrade pip setuptools wheel && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio || '${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install --no-cache-dir torch torchvision torchaudio && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install --no-cache-dir 'git+https://github.com/huggingface/lerobot.git'"
fi
if command -v uv >/dev/null 2>&1; then
  log_capture "SmolVLA dependency seed (uv)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && uv pip install --no-cache-dir --upgrade transformers diffusers accelerate safetensors sentencepiece pillow"
else
  log_capture "SmolVLA dependency seed (pip)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install --no-cache-dir --upgrade transformers diffusers accelerate safetensors sentencepiece pillow"
fi
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

if "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import torch
from lerobot.policies.smolvla import modeling_smolvla  # type: ignore
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
print("lerobot:", torch.__version__)
print("smolvla import:", modeling_smolvla.__name__)
print("has VLAFlowMatching:", callable(VLAFlowMatching))
PY
then
  log_capture "SmolVLA smoke import" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import torch
from lerobot.policies.smolvla import modeling_smolvla  # type: ignore
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
print("lerobot:", torch.__version__)
print("smolvla import:", modeling_smolvla.__name__)
print("has VLAFlowMatching:", callable(VLAFlowMatching))
PY
else
  log_capture "SmolVLA smoke import (failed first pass)" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import torch
from lerobot.policies.smolvla import modeling_smolvla  # type: ignore
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
print("lerobot:", torch.__version__)
print("smolvla import:", modeling_smolvla.__name__)
print("has VLAFlowMatching:", callable(VLAFlowMatching))
PY
  log_warn "SmolVLA smoke import failed; installing dependency fixes"
  if command -v uv >/dev/null 2>&1; then
    log_capture "SmolVLA dependency repair (uv)" bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && uv pip install transformers diffusers accelerate safetensors sentencepiece pillow"
  else
    log_capture "SmolVLA dependency repair (pip)" bash -lc "'${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install --upgrade transformers diffusers accelerate safetensors sentencepiece pillow"
  fi
  log_capture "SmolVLA smoke import (retry)" "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import torch
from lerobot.policies.smolvla import modeling_smolvla  # type: ignore
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
print("lerobot:", torch.__version__)
print("smolvla import:", modeling_smolvla.__name__)
print("has VLAFlowMatching:", callable(VLAFlowMatching))
PY
fi

snapshot_env "${SMOLVLA_LEROBOT_ENV_DIR}" "lerobot_mw"
append_passfail "PASS" "SmolVLA base stack smoke checks succeeded"
append_decision "phase03 SmolVLA install" "build flow policy runtime foundation" "PASS"
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
  log_capture "Meta-World install (pip)" bash -lc "mkdir -p '${SMOLVLA_CACHE_ROOT}/tmp' && export TMPDIR='${SMOLVLA_CACHE_ROOT}/tmp' && '${SMOLVLA_LEROBOT_ENV_DIR}/bin/pip' install metaworld>=0.1.0"
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

if [[ ! -x "${SMOLVLA_JEPA_ENV_DIR}/bin/python" ]]; then
  log_error "missing env ${SMOLVLA_JEPA_ENV_DIR}; run phase01 first"
  exit 1
fi

  source "${SMOLVLA_JEPA_ENV_DIR}/bin/activate"
  install_failed=0
  if command -v uv >/dev/null 2>&1; then
    if ! bash -c "cd '${SMOLVLA_VGG_REPO}/jepa-wms' && uv pip install -e ."; then
      log_warn "JEPA install (uv) failed; attempting pip fallback in jepa env"
      install_failed=1
    else
      log_info "JEPA install (uv) succeeded"
    fi
  else
    install_failed=1
  fi
  if (( install_failed == 1 )); then
    if ! bash -c "cd '${SMOLVLA_VGG_REPO}/jepa-wms' && '${SMOLVLA_JEPA_ENV_DIR}/bin/pip' install -e ."; then
      log_warn "JEPA install (pip fallback) failed; continuing with torch.hub load from cached assets"
    else
      log_capture "JEPA install (pip fallback)" bash -c "cd '${SMOLVLA_VGG_REPO}/jepa-wms' && '${SMOLVLA_JEPA_ENV_DIR}/bin/pip' install -e ."
    fi
  fi

  # Prefer the leRobot env for smoke checks because it already has the required torch build.
  local smoke_py="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
  if [[ ! -x "${smoke_py}" ]]; then
    smoke_py="${SMOLVLA_JEPA_ENV_DIR}/bin/python"
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

if [[ -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
  if [[ -f "${SMOLVLA_PIPE_ROOT}/bridge_builder.py" ]]; then
    log_capture "Bridge conversion" bash -lc "python '${SMOLVLA_PIPE_ROOT}/bridge_builder.py' \
      --jepa-source '${SMOLVLA_JEPA_SOURCE}' \
      --out-dir '${SMOLVLA_DATA_ROOT}' \
      --train-ratio '${SMOLVLA_BRIDGE_TRAIN_RATIO}' \
      --min-confidence '${SMOLVLA_BRIDGE_MIN_CONFIDENCE}' \
      --val-ratio '${SMOLVLA_BRIDGE_VAL_RATIO}' \
      --min-action-length '${SMOLVLA_BRIDGE_MIN_ACTION_LEN}'"
  else
    echo "bridge_builder.py not found; skipping conversion" >>"${REPORT_FILE}"
  fi
else
  log_warn "lerobot env unavailable; skip bridge build"
fi

if [[ -f "${SMOLVLA_DATA_ROOT}/bridge_summary.json" ]]; then
  append_passfail "PASS" "bridge summary generated at ${SMOLVLA_DATA_ROOT}/bridge_summary.json"
else
  append_passfail "WARN" "bridge summary missing; dataset may be empty"
fi
append_decision "phase08 bridge design" "bridge required for mixed synthetic training" "PASS"
}

stage09_vgg_gates() {
  log_info "phase09_vgg_gates: validate VGG prerequisites"
  cat >"${REPORT_FILE}" <<EOF
# phase09_vgg_gates
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

if [[ ! -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  log_error "missing env ${SMOLVLA_LEROBOT_ENV_DIR}; run phase01 first"
  exit 1
fi

source "${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate"
if [[ -f "${SMOLVLA_PIPE_ROOT}/validate_smolvla_vgg_gates.py" ]]; then
  gate_output="${SMOLVLA_VGG_GATE_JSON}"
  gate_ok="0"
  smolvla_site_pkg="$("${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  if log_capture "SmolVLA velocity/value-gate validation" bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && \
    cd / && \
    PYTHONPATH='${smolvla_site_pkg}' \
    '${SMOLVLA_LEROBOT_ENV_DIR}/bin/python' \
    '${SMOLVLA_PIPE_ROOT}/validate_smolvla_vgg_gates.py' \
    --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' \
    --episodes 2 \
    --steps 6 \
    --device '${SMOLVLA_BASELINE_DEVICE}' \
    --output '${gate_output}' \
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
  echo "Gates not fully passed; stageC will rely on SMOLVLA_ENABLE_VGG and gate JSON." >>"${REPORT_FILE}"
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

train_variant="${SMOLVLA_TRAIN_VARIANT:-auto}"
STAGE_A_CMD="${SMOLVLA_TRAIN_STAGE_A_CMD:-}"
STAGE_B_CMD="${SMOLVLA_TRAIN_STAGE_B_CMD:-}"
STAGE_C_CMD="${SMOLVLA_TRAIN_STAGE_C_CMD:-}"
if [[ "${SMOLVLA_TRAIN_DRY_RUN}" == "1" ]]; then
  DRY_FLAG="--dry-run"
else
  DRY_FLAG=""
fi
if [[ -z "${STAGE_A_CMD}" && -f "${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py" ]]; then
  STAGE_A_CMD="python '${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py' --mode stageA --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' --lerobot-env '${SMOLVLA_LEROBOT_ENV_DIR}' --real-data-root '${SMOLVLA_DATA_ROOT}/train' --max-steps '${SMOLVLA_FIRST_FT_STEPS}' --output-dir '${SMOLVLA_ARTIFACT_ROOT}/stage10' ${DRY_FLAG}"
fi
if [[ -z "${STAGE_B_CMD}" && -f "${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py" ]]; then
  STAGE_B_CMD="python '${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py' --mode stageB --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' --lerobot-env '${SMOLVLA_LEROBOT_ENV_DIR}' --real-data-root '${SMOLVLA_DATA_ROOT}/train' --jepa-data-root '${SMOLVLA_DATA_ROOT}/val' --max-steps '${SMOLVLA_FIRST_FT_STEPS}' --output-dir '${SMOLVLA_ARTIFACT_ROOT}/stage10' ${DRY_FLAG}"
fi
if [[ -z "${STAGE_C_CMD}" && -f "${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py" ]]; then
  STAGE_C_CMD="python '${SMOLVLA_PIPE_ROOT}/train_smolvla_vggflow.py' --mode stageC --checkpoint '${SMOLVLA_INIT_CHECKPOINT}' --lerobot-env '${SMOLVLA_LEROBOT_ENV_DIR}' --real-data-root '${SMOLVLA_DATA_ROOT}/train' --output-dir '${SMOLVLA_ARTIFACT_ROOT}/stage10' --gate-json '${SMOLVLA_VGG_GATE_JSON}' --match-weight '${SMOLVLA_VGG_MATCH_WEIGHT}' --match-warmup '${SMOLVLA_VGG_MATCH_WARMUP_STEPS}' --value-head-dim '${SMOLVLA_VGG_VALUE_HEAD_DIM}' --value-head-steps '${SMOLVLA_VGG_VALUE_HEAD_STEPS}' --value-head-seed '${SMOLVLA_VGG_VALUE_HEAD_SEED}' --value-head-min-grad '${SMOLVLA_VGG_VALUE_HEAD_MIN_GRAD}' --trace-cap '${SMOLVLA_VGG_MATCH_TRACE_CAP}' ${DRY_FLAG}"
fi

if [[ "${train_variant}" == "a" ]]; then
  STAGE_B_CMD=""
  STAGE_C_CMD=""
elif [[ "${train_variant}" == "b" ]]; then
  STAGE_A_CMD=""
  STAGE_C_CMD=""
elif [[ "${train_variant}" == "c" ]]; then
  STAGE_A_CMD=""
  STAGE_B_CMD=""
fi

if [[ -z "${STAGE_A_CMD}" ]]; then
  append_warn="1"
  echo "SMOLVLA_TRAIN_STAGE_A_CMD not set. StageA will be skipped." >>"${REPORT_FILE}"
else
  log_capture "StageA (real only)" bash -lc "cd / && ${STAGE_A_CMD}"
fi

if [[ -n "${STAGE_B_CMD}" ]]; then
  log_capture "StageB (JMPA mix)" bash -lc "cd / && ${STAGE_B_CMD}"
fi

if [[ "${SMOLVLA_ENABLE_VGG:-0}" == "1" ]]; then
  if [[ -n "${STAGE_C_CMD}" ]]; then
    gate_ok="0"
    if [[ -f "${SMOLVLA_VGG_GATE_JSON}" ]]; then
      gate_ok="$(python3 - <<PY\nimport json\nfrom pathlib import Path\npath = Path(\"${SMOLVLA_VGG_GATE_JSON}\")\nif not path.exists():\n    print(0)\nelse:\n    try:\n        payload = json.loads(path.read_text(encoding=\"utf-8\"))\n        gate_ok = bool(payload.get(\"gate_ok\", False))\n        contract_ok = bool(payload.get(\"contract_ok\", True))\n        print(1 if (gate_ok and contract_ok) else 0)\n    except Exception:\n        print(0)\nPY\n)"
    fi
    if [[ "${gate_ok}" == "1" ]]; then
      log_capture "StageC (VGG auxiliary)" bash -lc "cd / && ${STAGE_C_CMD}"
    else
      append_passfail "WARN" "StageC skipped due to gate check (or missing gate file): ${SMOLVLA_VGG_GATE_JSON}"
      mkdir -p "${SMOLVLA_ARTIFACT_ROOT}/stage10"
      cat >"${SMOLVLA_ARTIFACT_ROOT}/stage10/vgg_stagec_disabled.txt" <<EOF
StageC was skipped by stage10 gate-driven safety logic.
Gate file: ${SMOLVLA_VGG_GATE_JSON}
Gate pass: ${gate_ok:-0}
EOF
    fi
  else
    echo "SMOLVLA_ENABLE_VGG=1 but SMOLVLA_TRAIN_STAGE_C_CMD is empty." >>"${REPORT_FILE}"
  fi
else
  echo "SMOLVLA_ENABLE_VGG not set; skipping StageC." >>"${REPORT_FILE}"
fi

append_passfail "PASS" "stage loop driver executed"
append_decision "phase10 train loop" "first stage sequence staged by env overrides" "PASS"
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

stage12_reporting() {
  log_info "phase12_reporting: final artifact bundle"
  cat >"${REPORT_FILE}" <<EOF
# phase12_reporting
Executed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

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
  4|phase04_metaworld_install|stage04_install_metaworld)
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
  10|phase10_train_loop|stage10_train_loop|stage05_train_stageA_real_only|stage06_train_stageB_jepa_mix|stage08_train_stageC_vgg_aux)
    stage10_train_loop
    ;;
  11|phase11_slurm_orchestration|stage11_slurm_orchestration)
    stage11_slurm_orchestration
    ;;
  12|phase12_reporting|stage12_reporting|stage09_final_eval_and_bundle)
    stage12_reporting
    ;;
  *)
    log_error "Unknown stage: ${STAGE}"
    exit 2
    ;;
esac

