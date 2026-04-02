#!/usr/bin/env python3
"""Package Slurm bundle only after local confidence gate passes."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def _count_metric_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if "step=" in ln)


def _max_checkpoint_step(ckpt_root: Path) -> int:
    if not ckpt_root.exists():
        return -1
    max_step = -1
    for p in ckpt_root.glob("**/*"):
        if p.is_dir() and p.name.isdigit():
            max_step = max(max_step, int(p.name))
    return max_step


def _write_slurm_templates(portable_dir: Path) -> None:
    slurm_dir = portable_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    main = """#!/bin/bash
#SBATCH --job-name=pi05_cube_main4g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pi05_cube_main4g_%j.log
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
REPO_ID="${REPO_ID:?set REPO_ID}"
EXP_NAME="${EXP_NAME:-main4g}"
python "$PROJECT_ROOT/cube_dataset/finetune_pi05_cube_single_v2/train_pi05_lora_cube.py" \
  --run-root "$RUN_ROOT" --repo-id "$REPO_ID" --exp-name "$EXP_NAME" \
  --batch-size 64 --num-train-steps 10000 --log-interval 100 --save-interval 2000 --keep-period 2000
"""
    alt = """#!/bin/bash
#SBATCH --job-name=pi05_cube_alt4g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pi05_cube_alt4g_%j.log
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
REPO_ID="${REPO_ID:?set REPO_ID}"
EXP_NAME="${EXP_NAME:-alt4g}"
python "$PROJECT_ROOT/cube_dataset/finetune_pi05_cube_single_v2/train_pi05_lora_cube.py" \
  --run-root "$RUN_ROOT" --repo-id "$REPO_ID" --exp-name "$EXP_NAME" \
  --batch-size 64 --num-train-steps 10000 --log-interval 100 --save-interval 2000 --keep-period 2000
"""
    eval20 = """#!/bin/bash
#SBATCH --job-name=pi05_cube_eval20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pi05_cube_eval20_%j.log
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
python "$PROJECT_ROOT/cube_dataset/finetune_pi05_cube_single_v2/eval_pi05_lora_cube.py" \
  --project-root "$PROJECT_ROOT" --run-root "$RUN_ROOT" --checkpoint "$CHECKPOINT" --episodes 20 --require-openpi
"""
    (slurm_dir / "run_train_pi05_cube_main4g.slurm").write_text(main, encoding="utf-8")
    (slurm_dir / "run_train_pi05_cube_alt4g.slurm").write_text(alt, encoding="utf-8")
    (slurm_dir / "run_eval_pi05_cube_20ep.slurm").write_text(eval20, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--local-run-root", type=Path, required=True, help=".../local_build/<run_tag>")
    parser.add_argument("--bundle-out-root", type=Path, required=True, help=".../slurm_bundle/<run_tag>")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--require-min-step", type=int, default=2000)
    parser.add_argument("--require-min-log-lines", type=int, default=2)
    args = parser.parse_args()

    local_run_root = args.local_run_root.resolve()
    bundle_out_root = args.bundle_out_root.resolve()
    bundle_out_root.mkdir(parents=True, exist_ok=True)

    metrics_log = local_run_root / "metrics_step.log"
    line_count = _count_metric_lines(metrics_log)
    max_ckpt_step = _max_checkpoint_step(args.checkpoint_root.resolve())

    gate = {
        "metrics_log": str(metrics_log),
        "metrics_log_lines": line_count,
        "min_required_log_lines": args.require_min_log_lines,
        "max_checkpoint_step": max_ckpt_step,
        "min_required_checkpoint_step": args.require_min_step,
    }
    gate_ok = line_count >= args.require_min_log_lines and max_ckpt_step >= args.require_min_step
    gate["status"] = "pass" if gate_ok else "fail"
    (bundle_out_root / "local_gate_report.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    if not gate_ok:
        raise SystemExit("Local confidence gate not satisfied; refusing Slurm bundle creation.")

    portable_dir = bundle_out_root / "portable"
    portable_dir.mkdir(parents=True, exist_ok=True)
    _write_slurm_templates(portable_dir)

    # Package workflow scripts.
    workflow_dir = args.project_root.resolve() / "cube_dataset" / "finetune_pi05_cube_single_v2"
    tar_path = bundle_out_root / "pi05_cube_single_v2_slurm_bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(workflow_dir), arcname="cube_dataset/finetune_pi05_cube_single_v2")
        tar.add(str(portable_dir), arcname="portable")
    (bundle_out_root / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "tarball": str(tar_path),
                "workflow_dir": str(workflow_dir),
                "portable_dir": str(portable_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {tar_path}")


if __name__ == "__main__":
    main()

