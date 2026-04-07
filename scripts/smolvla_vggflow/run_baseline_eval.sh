#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/common.sh"

episodes="${SMOLVLA_BASELINE_EPISODES}"
seed="${SMOLVLA_BASELINE_SEED}"
device="${SMOLVLA_BASELINE_DEVICE}"
video="${SMOLVLA_BASELINE_VIDEO}"
video_length="${SMOLVLA_BASELINE_VIDEO_LENGTH}"
video_interval="${SMOLVLA_BASELINE_VIDEO_INTERVAL}"
output_root="${SMOLVLA_ARTIFACT_ROOT}/phase06_baseline"
checkpoint="${SMOLVLA_INIT_CHECKPOINT}"

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --episodes)
      episodes="${2}"
      shift 2
      ;;
    --seed)
      seed="${2}"
      shift 2
      ;;
    --device)
      device="${2}"
      shift 2
      ;;
    --video)
      video="${2}"
      shift 2
      ;;
    --video-length)
      video_length="${2}"
      shift 2
      ;;
    --video-interval)
      video_interval="${2}"
      shift 2
      ;;
    --output-root)
      output_root="${2}"
      shift 2
      ;;
    --checkpoint)
      checkpoint="${2}"
      shift 2
      ;;
    *)
      echo "Unknown arg: ${1}" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${device}" || "${device}" == "auto" ]]; then
  if [[ -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
    resolved_device="$(
      "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<PY
import torch
if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
PY
    )"
    device="${resolved_device}"
  else
    device="cpu"
  fi
fi

mkdir -p "${output_root}"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
output_dir="${output_root}/run_${timestamp}_ep${episodes}_v${video}"
mkdir -p "${output_dir}"

tmp_dir="$(mktemp -d)"
tmp_cleanup() {
  rm -rf "${tmp_dir}"
}
trap tmp_cleanup EXIT
site_packages="$(
  "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import site
print(":".join(site.getsitepackages()))
PY
)"
cat > "${tmp_dir}/sitecustomize.py" <<'PY'
import importlib.util
import pathlib
import site
import sys


def _restore_datasets_module() -> None:
    # Workaround for a known import-shadowing issue where a local `lerobot.datasets`
    # package can mask Hugging Face's external `datasets` package during policy import.
    candidates = []
    for site_dir in site.getsitepackages() + [site.getusersitepackages() or ""]:
        if not site_dir:
            continue
        candidates.append(pathlib.Path(site_dir) / "datasets" / "__init__.py")
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("datasets", str(candidate))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.__file__ = str(candidate)
                sys.modules["datasets"] = module
                break


_restore_datasets_module()
PY

log_info "Running baseline push-v3 eval: episodes=${episodes}, device=${device}, seed=${seed}, video=${video}, checkpoint=${checkpoint}"
xvfb-run -a -s "-screen 0 1280x1024x24" env \
  LEAKY=1 \
  UV_INSECURE_HOST=http://localhost:9999 \
  MH_REPO=metaworld-v2 \
  METAWORLD_RENDER_MODE=rgb_array \
  PYTHONPATH="${tmp_dir}:${site_packages}:${PYTHONPATH:-}" \
  bash -lc "source '${SMOLVLA_LEROBOT_ENV_DIR}/bin/activate' && cd /tmp && lerobot-eval \
  --policy.type smolvla \
  --policy.pretrained_path ${checkpoint} \
  --policy.load_vlm_weights true \
  --policy.vlm_model_name HuggingFaceTB/SmolVLM2-500M-Instruct \
  --policy.expert_width_multiplier 0.5 \
  --policy.self_attn_every_n_layers 0 \
  --policy.n_action_steps 1 \
  --env.type metaworld \
  --env.task push-v3 \
  --eval.n_episodes ${episodes} \
  --eval.batch_size 1 \
  --eval.use_async_envs false \
  --policy.device ${device} \
  --env.multitask_eval false \
  --output_dir '${output_dir}' \
  --seed ${seed}"

echo "Baseline eval output directory: ${output_dir}"
