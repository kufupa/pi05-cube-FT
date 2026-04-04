#!/usr/bin/env python3
"""Launch local pi0.5 LoRA training with explicit config."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

from cube_dataset.finetune_pi05_cube_single_v2.pipeline_common import (
    build_train_config,
    training_summary_dict,
)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0, help="If >0, overrides num_train_steps for smoke run.")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    # Keep local LeRobot datasets under run_root on /vol/bitbucket.
    os.environ.setdefault("LEROBOT_HOME", str(run_root / "lerobot_home"))

    steps = int(args.smoke_steps) if int(args.smoke_steps) > 0 else int(args.num_train_steps)
    config = build_train_config(
        repo_id=args.repo_id,
        run_root=run_root,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
        resume=args.resume,
        overwrite=args.overwrite,
        wandb_enabled=args.wandb_enabled,
    )

    summary = training_summary_dict(config, repo_id=args.repo_id, run_root=run_root)
    summary["smoke_steps"] = int(args.smoke_steps)
    (run_root / "train_launch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    train_py = Path(__file__).resolve().parents[2] / "external" / "openpi" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train_script", train_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load OpenPI train script from {train_py}")
    openpi_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(openpi_train)

    t0 = time.time()
    openpi_train.main(config)
    t1 = time.time()

    done = {
        "status": "pass",
        "elapsed_sec": t1 - t0,
        "checkpoint_dir": str(config.checkpoint_dir),
        "exp_name": config.exp_name,
        "resume": bool(config.resume),
    }
    (run_root / "train_report.json").write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {(run_root / 'train_report.json')}")


if __name__ == "__main__":
    main()

