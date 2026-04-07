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

# Prefer the newest trained checkpoint for post-train eval when not explicitly set.
if [[ -z "${SMOLVLA_FINAL_EVAL_CHECKPOINT:-}" ]]; then
  auto_ckpt="$(
    REPO_ROOT="${REPO_ROOT}" python - <<'PY'
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
artifacts = repo_root / "artifacts"

stage_patterns = [
    "stage10_c_*_job*/stageC/vgg_aux/train_run/checkpoints/006000/pretrained_model",
    "stage10_b_*_job*/stageB/jepa_mix/train_run/checkpoints/006000/pretrained_model",
    "stage10_a_*_job*/stageA/train_run/checkpoints/006000/pretrained_model",
]

for pattern in stage_patterns:
    cands: list[Path] = []
    for p in artifacts.glob(pattern):
        if (p / "config.json").is_file():
            cands.append(p)
    if not cands:
        continue
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(cands[0])
    raise SystemExit(0)

print("")
PY
  )"
  if [[ -n "${auto_ckpt}" ]]; then
    export SMOLVLA_FINAL_EVAL_CHECKPOINT="${auto_ckpt}"
    echo "[stage09] auto-selected SMOLVLA_FINAL_EVAL_CHECKPOINT=${SMOLVLA_FINAL_EVAL_CHECKPOINT}"
  fi
fi

bash "${PIPE}" phase13_post_train_eval
bash "${PIPE}" phase12_reporting
