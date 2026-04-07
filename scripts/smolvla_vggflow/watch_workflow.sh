#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <workflow.json> [watch_workflow.py options before job expansion...]"
  echo "Example: $0 runs/workflow_x.json --poll 90 --auto-resubmit --max-retries 2"
  exit 2
fi
WORKFLOW_JSON="$1"
shift

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# If the path is relative and missing from cwd, resolve against repo root (common: runs/workflow_*.json).
if [[ "${WORKFLOW_JSON}" != /* ]] && [[ ! -f "${WORKFLOW_JSON}" ]] && [[ -f "${REPO_ROOT}/${WORKFLOW_JSON}" ]]; then
  WORKFLOW_JSON="${REPO_ROOT}/${WORKFLOW_JSON}"
fi

export WORKFLOW_JSON
mapfile -t JOB_IDS < <(
  python3 - <<'PY'
import json
import os
import sys

path = os.environ["WORKFLOW_JSON"]
data = json.loads(open(path, encoding="utf-8").read())
rid = data.get("run_id") or data.get("smolvla_run_id")
if rid:
    print(f"[watch_workflow] workflow run_id from JSON: {rid}", file=sys.stderr)
for item in data.get("stages", []):
    jid = item.get("job_id")
    if jid and str(jid) != "<unsubmitted>":
        print(str(jid))
PY
)

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
  echo "No job ids in workflow (or dry-run placeholders)." >&2
  exit 1
fi

echo "watching ${#JOB_IDS[@]} jobs"
# Pass "$@" before --job-ids so optional flags (e.g. --poll) are not swallowed by nargs="+".
exec python3 "${SCRIPT_DIR}/watch_workflow.py" "$@" --job-ids "${JOB_IDS[@]}"
