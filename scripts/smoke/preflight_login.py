#!/usr/bin/env python3
"""Fast login-node preflight for VLAW/OpenPI environments.

This script is intentionally lightweight:
- no dataset scans
- no model downloads
- no GPU requirement
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


def _run(cmd):
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "elapsed_s": round(time.time() - start, 3),
    }


def _check_import(module_name):
    try:
        importlib.import_module(module_name)
        return {"ok": True, "error": ""}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/preflight_login.json")
    parser.add_argument(
        "--strict-imports",
        default="openpi,torch,pynput,evdev",
        help="Comma-separated modules that must import successfully.",
    )
    parser.add_argument(
        "--resolver-probe",
        action="store_true",
        help="Run a cheap uv resolver probe on external/openpi.",
    )
    args = parser.parse_args()

    required = [m.strip() for m in args.strict_imports.split(",") if m.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    python_include = sysconfig.get_paths().get("include", "")
    python_h = str(Path(python_include) / "Python.h") if python_include else ""

    report = {
        "status": "pass",
        "host": os.uname().nodename,
        "python": {
            "executable": sys.executable,
            "version": sys.version.splitlines()[0],
            "include_dir": python_include,
            "python_h_exists": bool(python_h and Path(python_h).exists()),
            "python_h_path": python_h,
        },
        "tooling": {
            "uv_path": shutil.which("uv") or "",
            "bash_path": shutil.which("bash") or "",
        },
        "commands": [],
        "imports": {},
        "failures": [],
    }

    # Basic command probes
    for cmd in (["python", "--version"], ["uv", "--version"]):
        result = _run(cmd)
        report["commands"].append(result)
        if result["returncode"] != 0:
            report["failures"].append(f"command_failed:{result['cmd']}")

    # Import probes (transitive deps that previously failed are included)
    for module_name in required:
        imp = _check_import(module_name)
        report["imports"][module_name] = imp
        if not imp["ok"]:
            report["failures"].append(f"import_failed:{module_name}")

    # Optional low-cost dependency resolution sanity check
    if args.resolver_probe:
        probe = _run(["uv", "pip", "install", "--dry-run", "-e", "external/openpi"])
        report["commands"].append(probe)
        if probe["returncode"] != 0:
            report["failures"].append("resolver_probe_failed")

    if report["failures"]:
        report["status"] = "fail"

    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(out_path)}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
