#!/usr/bin/env python3
"""Verify expected VLAW artifacts and schema consistency."""

import argparse
import json
from pathlib import Path


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/artifact_verification.json")
    args = parser.parse_args()

    required_files = [
        Path("scripts/pbs/run_openpi_real_eval.pbs"),
        Path("scripts/pbs/run_vlaw_loop.pbs"),
        Path("scripts/pbs/run_smoke_gpu.pbs"),
        Path("scripts/smoke/preflight_login.py"),
    ]
    optional_results = [Path("results_base.json"), Path("results_base_real.json"), Path("results_vlaw.json")]

    report = {"status": "pass", "missing_required": [], "findings": [], "results_schema": {}}

    for f in required_files:
        if not f.exists():
            report["missing_required"].append(str(f))
    if report["missing_required"]:
        report["status"] = "fail"

    for p in optional_results:
        if not p.exists():
            report["findings"].append(f"missing_optional_result:{p}")
            continue
        obj = _load_json(p)
        key = p.name
        report["results_schema"][key] = sorted(list(obj.keys()))

        if "success_rate" in obj and "episodes" not in obj:
            report["status"] = "fail"
            report["findings"].append(f"schema_mismatch:{p}:missing_episodes")
        if p.name == "results_vlaw.json" and "metrics" not in obj:
            report["status"] = "fail"
            report["findings"].append("schema_mismatch:results_vlaw.json:missing_metrics")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out)}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
