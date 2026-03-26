#!/usr/bin/env python3
"""End-to-end smoke: import check + run_vlaw_loop --smoke (mock RM on CPU)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run gated smoke pipeline")
    args = parser.parse_args()
    if not args.smoke:
        print("Pass --smoke to run the pipeline smoke test.", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["VLAW_MOCK_REWARD"] = "1"

    r1 = subprocess.run(
        [sys.executable, str(root / "scripts" / "smoke_env_check.py")],
        cwd=root,
        env=env,
    )
    if r1.returncode != 0:
        return r1.returncode

    r2 = subprocess.run(
        [
            sys.executable,
            str(root / "src" / "training" / "run_vlaw_loop.py"),
            "--config",
            str(root / "configs" / "droid_single_task_vlaw.yaml"),
            "--smoke",
        ],
        cwd=root,
        env=env,
    )
    return r2.returncode


if __name__ == "__main__":
    raise SystemExit(main())
