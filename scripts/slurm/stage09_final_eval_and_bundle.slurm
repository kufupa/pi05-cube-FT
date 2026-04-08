#!/usr/bin/env bash
#SBATCH --job-name=smolvla-s09
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage09_report_%j.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
PIPE="${REPO_ROOT}/scripts/smolvla_vggflow/run_stage.sh"
source "${REPO_ROOT}/scripts/smolvla_vggflow/config.sh"

ARTIFACT_ROOT="${SMOLVLA_ARTIFACT_ROOT}"
if [[ -n "${SMOLVLA_RUN_SCOPE_ID:-}" ]]; then
  ARTIFACT_ROOT="${ARTIFACT_ROOT}/run_${SMOLVLA_RUN_SCOPE_ID}"
fi

PYTHON_BIN=""
if [[ -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
fi

# Prefer the newest trained checkpoint for post-train eval when final eval ckpt is unset
# or still at the base init default from config.sh.
if [[ -z "${SMOLVLA_FINAL_EVAL_CHECKPOINT:-}" || "${SMOLVLA_FINAL_EVAL_CHECKPOINT}" == "${SMOLVLA_INIT_CHECKPOINT}" ]]; then
  auto_ckpt=""
  if [[ -n "${PYTHON_BIN}" ]]; then
    auto_ckpt="$(
    REPO_ROOT="${REPO_ROOT}" ARTIFACT_ROOT="${ARTIFACT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
artifact_root = Path(os.environ["ARTIFACT_ROOT"])
artifacts = artifact_root if artifact_root.is_absolute() else repo_root / artifact_root

stage_patterns = [
    "stage10_c_*_job*/stageC/vgg_aux/train_run/checkpoints/*/pretrained_model",
    "stage10_b_*_job*/stageB/jepa_mix/train_run/checkpoints/*/pretrained_model",
    "stage10_a_*_job*/stageA/train_run/checkpoints/*/pretrained_model",
]

cands: list[Path] = []
for pattern in stage_patterns:
    for p in artifacts.glob(pattern):
        if (p / "config.json").is_file():
            cands.append(p)

if cands:
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(cands[0])
    raise SystemExit(0)

print("")
PY
    )"
  else
    echo "[stage09] WARN: no python interpreter found; skipping checkpoint auto-discovery"
  fi
  if [[ -n "${auto_ckpt}" ]]; then
    export SMOLVLA_FINAL_EVAL_CHECKPOINT="${auto_ckpt}"
    echo "[stage09] auto-selected SMOLVLA_FINAL_EVAL_CHECKPOINT=${SMOLVLA_FINAL_EVAL_CHECKPOINT}"
  fi
fi

bash "${PIPE}" phase13_post_train_eval
bash "${PIPE}" phase12_reporting
