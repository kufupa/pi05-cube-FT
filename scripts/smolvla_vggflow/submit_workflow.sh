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
EOF
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="1"
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
WORKFLOW_JSON="${SMOLVLA_WORKFLOW_JSON:-${REPO_ROOT}/runs/workflow_${RUN_ID}.json}"

mkdir -p "${REPO_ROOT}/runs"

if [[ "${DRY_RUN}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/smolvla_workflow_launcher.py" --write-json "${WORKFLOW_JSON}"
else
  python3 "${SCRIPT_DIR}/smolvla_workflow_launcher.py" \
    --submit \
    --write-json "${WORKFLOW_JSON}"
fi

echo "workflow run id: ${RUN_ID}"
echo "workflow file: ${WORKFLOW_JSON}"

