#!/usr/bin/env python3
"""Evaluate a trained checkpoint on cube-single UR5e rollout script."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--joint-scale", type=float, default=0.05)
    parser.add_argument("--require-openpi", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    eval_dir = run_root / "eval_20ep"
    eval_dir.mkdir(parents=True, exist_ok=True)

    rollouts = project_root / "cube_dataset" / "run_pi05_base_ur5e_rollouts.py"
    cmd = [
        sys.executable,
        str(rollouts),
        "--checkpoint",
        args.checkpoint,
        "--out-dir",
        str(eval_dir),
        "--n",
        str(args.episodes),
        "--start-index",
        str(args.start_index),
        "--fps",
        str(args.fps),
        "--max-steps",
        str(args.max_steps),
        "--joint-scale",
        str(args.joint_scale),
    ]
    if args.require_openpi:
        cmd.append("--require-openpi")

    subprocess.check_call(cmd, cwd=str(project_root))

    results_path = eval_dir / "results.jsonl"
    rows = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    success_count = sum(1 for r in rows if bool(r.get("success", False)))
    summary = {
        "status": "pass" if len(rows) >= args.episodes else "fail",
        "checkpoint": args.checkpoint,
        "episodes_requested": args.episodes,
        "episodes_found": len(rows),
        "success_count": success_count,
        "success_rate": (float(success_count) / len(rows)) if rows else 0.0,
        "eval_dir": str(eval_dir),
        "results_path": str(results_path),
    }
    (run_root / "eval_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {(run_root / 'eval_report.json')}")
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

