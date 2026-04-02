#!/usr/bin/env python3
"""Build and serialize explicit pi0.5 LoRA training config summary."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--decay-steps", type=int, default=1000000)
    parser.add_argument("--decay-lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    config = build_train_config(
        repo_id=args.repo_id,
        run_root=run_root,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        fsdp_devices=args.fsdp_devices,
        peak_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        decay_lr=args.decay_lr,
        seed=args.seed,
        resume=args.resume,
        overwrite=args.overwrite,
        wandb_enabled=args.wandb_enabled,
    )
    summary = training_summary_dict(config, repo_id=args.repo_id, run_root=run_root)
    summary.update(
        {
            "peak_lr": args.peak_lr,
            "warmup_steps": args.warmup_steps,
            "decay_steps": args.decay_steps,
            "decay_lr": args.decay_lr,
            "seed": args.seed,
        }
    )
    out = run_root / "train_config_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

