#!/usr/bin/env bash
# Verify artifacts from scripts/run_vlaw_qwen3_isolated.sh (or any compatible VLAW run dir).
# Usage: ./scripts/verify_vlaw_run.sh output/vlaw_qwen3_YYYYMMDD_HHMMSS
set -euo pipefail
RUN="${1:?usage: $0 <run_directory>}"
ok=0
for f in "$RUN/results_vlaw.json" "$RUN/wm_predictor.pt" "$RUN/wm_rollout_preview.gif"; do
  if [[ -f "$f" ]]; then
    echo "OK  $f"
  else
    echo "MISSING  $f"
    ok=1
  fi
done
if [[ -f "$RUN/results_vlaw.json" ]]; then
  python3 - <<PY
import json
from pathlib import Path
p = Path("$RUN") / "results_vlaw.json"
d = json.loads(p.read_text(encoding="utf-8"))
prov = d.get("provenance", {})
smoke = prov.get("smoke", "<missing>")
print(f"provenance.smoke = {smoke!r}")
base = d.get("metrics_meta", {}).get("Base", {})
print(f"metrics_meta.Base.rm_backend = {base.get('rm_backend', '<missing>')!r}")
PY
fi
exit "$ok"
