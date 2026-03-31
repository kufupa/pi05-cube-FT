#!/usr/bin/env bash
# VLAW loop with Qwen3-VL reward (see configs/droid_single_task_vlaw.yaml).
# Writes wm_predictor.pt, wm_rollout_preview.gif, results_vlaw.json under a new directory.
#
# Preconditions:
#   - Do NOT set VLAW_MOCK_REWARD (this script unsets it).
#   - Do NOT pass --smoke (not supported here; use full run_vlaw_loop.py for smoke).
#   - GPU strongly recommended: without CUDA the code falls back to mock RM for filtering.
#
# Optional env:
#   RUN=/path/or/name     Output root (default: output/vlaw_qwen3_<timestamp>)
#   BASE_REAL=/path/to/results_base_real.json  Passed as --base-real-path
#   UV_PYTHON=...         If using module-loaded Python with uv (e.g. EBROOTPYTHON/bin/python3)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d data/droid_sample/droid_100 ]]; then
  echo "FATAL: missing data/droid_sample/droid_100 (DROID sample episodes)."
  exit 1
fi

if type module &>/dev/null; then
  module load libffi/3.4.4-GCCcore-13.2.0 2>/dev/null || true
fi

unset VLAW_MOCK_REWARD || true

if [[ -n "${RUN:-}" ]]; then
  OUT="$RUN"
else
  OUT="output/vlaw_qwen3_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT"

BASE_ARGS=()
if [[ -n "${BASE_REAL:-}" ]]; then
  BASE_ARGS+=(--base-real-path "$BASE_REAL")
fi

echo "[run_vlaw_qwen3_isolated] output_dir=$OUT"
echo "[run_vlaw_qwen3_isolated] results=$OUT/results_vlaw.json"
echo "[run_vlaw_qwen3_isolated] VLAW_MOCK_REWARD unset; cuda will determine real vs mock RM."

set -o pipefail
uv run python src/training/run_vlaw_loop.py \
  --config configs/droid_single_task_vlaw.yaml \
  --output-dir "$OUT" \
  --results-path "$OUT/results_vlaw.json" \
  "${BASE_ARGS[@]}" \
  2>&1 | tee "$OUT/run.log"

echo "[run_vlaw_qwen3_isolated] done. RUN=$OUT" | tee -a "$OUT/run.log"
