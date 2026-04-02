#!/usr/bin/env python3
"""Resume LoRA training from latest checkpoint with integrity checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cube_dataset.finetune_pi05_cube_single_v2.pipeline_common import (
    DEFAULT_CONFIG_NAME,
    build_train_config,
)


def _latest_step_dir(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    step_dirs = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not step_dirs:
        return None
    return max(step_dirs, key=lambda p: int(p.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-train-steps", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--keep-period", type=int, default=2000)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    cfg = build_train_config(
        repo_id=args.repo_id,
        run_root=run_root,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        fsdp_devices=args.fsdp_devices,
        resume=True,
        overwrite=False,
        wandb_enabled=False,
    )
    ckpt_dir = cfg.checkpoint_dir
    latest = _latest_step_dir(ckpt_dir)
    if latest is None:
        raise RuntimeError(f"No numeric checkpoint step directories found in {ckpt_dir}")

    info = {
        "status": "ready",
        "config_name": DEFAULT_CONFIG_NAME,
        "exp_name": args.exp_name,
        "checkpoint_dir": str(ckpt_dir),
        "latest_step_dir": str(latest),
    }
    out = run_root / "resume_check_report.json"
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

    if args.dry_run:
        print("Dry-run only. Not launching resume training.")
        return

    train_py = Path(__file__).resolve().parent / "train_pi05_lora_cube.py"
    cmd = [
        sys.executable,
        str(train_py),
        "--run-root",
        str(run_root),
        "--repo-id",
        args.repo_id,
        "--exp-name",
        args.exp_name,
        "--batch-size",
        str(args.batch_size),
        "--num-train-steps",
        str(args.num_train_steps),
        "--log-interval",
        str(args.log_interval),
        "--save-interval",
        str(args.save_interval),
        "--keep-period",
        str(args.keep_period),
        "--fsdp-devices",
        str(args.fsdp_devices),
        "--resume",
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

