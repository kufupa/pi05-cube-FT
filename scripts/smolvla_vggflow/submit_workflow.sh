#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="0"
if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  submit_workflow.sh [--dry-run] [--help]

Options:
  --dry-run   Validate launcher invocation and write workflow.json without submitting jobs.
  --help      Show this message.

GPU policy:
  Slurm-only stages in this pipeline are submitted via launcher with GPU partition fallback.
  Do not run phase06/07/09/10/12/13 stages directly on login nodes.

Serial DAG: 11 Slurm scripts (includes CPU stage01b Meta-World install). Optional: run the launcher with
  --branch-parallel --submit
for fan-out/fan-in dependencies (not compatible with SMOLVLA_STAGE11_ENABLED=1).

After submit: scripts/smolvla_vggflow/watch_workflow.sh runs/workflow_<id>.json — relative paths are resolved against the repo root when missing from the current working directory.
EOF
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="1"
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
export RUN_ID
WORKFLOW_JSON="${SMOLVLA_WORKFLOW_JSON:-${REPO_ROOT}/runs/workflow_${RUN_ID}.json}"

mkdir -p "${REPO_ROOT}/runs"

if [[ "${DRY_RUN}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/smolvla_workflow_launcher.py" --write-json "${WORKFLOW_JSON}"
else
  echo "[submit_workflow] partition fallback: ${SMOLVLA_PARTITION_LIST:-a100,a40,a30,t4,a16}"
  python3 "${SCRIPT_DIR}/smolvla_workflow_launcher.py" \
    --submit \
    --write-json "${WORKFLOW_JSON}"
fi

echo "workflow run id: ${RUN_ID}"
echo "workflow file: ${WORKFLOW_JSON}"

