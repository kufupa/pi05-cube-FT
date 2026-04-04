#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <workflow.json>"
  exit 2
fi
WORKFLOW_JSON="$1"

python3 - <<PY
import json
import subprocess
from pathlib import Path
import sys
import os

path = Path(os.environ["WORKFLOW_JSON"])
data = json.loads(path.read_text(encoding="utf-8"))
job_ids = [str(item["job_id"]) for item in data.get("stages", []) if item.get("job_id")]
if not job_ids:
    raise SystemExit("No job ids in workflow")
print(f"watching {len(job_ids)} jobs")
subprocess.call(
    [
        "python3",
        (path.parent.parent / "scripts" / "smolvla_vggflow" / "watch_workflow.py").resolve().as_posix(),
        "--job-ids",
        *job_ids,
    ]
)
PY

