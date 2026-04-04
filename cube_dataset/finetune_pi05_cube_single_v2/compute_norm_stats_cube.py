#!/usr/bin/env python3
"""Compute norm stats for local cube-single LoRA config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tqdm

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--exp-name", type=str, default="norm_stats_only")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=50000)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    # Ensure local LeRobot dataset paths resolve under /vol/bitbucket before OpenPI data loader imports.
    os.environ.setdefault("LEROBOT_HOME", str(run_root / "lerobot_home"))

    import openpi.shared.normalize as normalize
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms

    from cube_dataset.finetune_pi05_cube_single_v2.pipeline_common import build_train_config

    class RemoveStrings(_transforms.DataTransformFn):
        def __call__(self, x: dict) -> dict:
            return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

    config = build_train_config(
        repo_id=args.repo_id,
        run_root=run_root,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=100,
        wandb_enabled=False,
        overwrite=False,
        resume=False,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    if args.max_frames < len(dataset):
        num_batches = args.max_frames // args.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // args.batch_size
        shuffle = False
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        # Keep norm-stat pass single-process for portability on clusters.
        num_workers=0,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing cube norm stats"):
        for k in ("state", "actions"):
            stats[k].update(np.asarray(batch[k]))
    norm_stats = {k: stats[k].get_statistics() for k in stats}

    out_dir = config.assets_dirs / data_config.repo_id
    normalize.save(out_dir, norm_stats)

    report = {
        "status": "pass",
        "run_root": str(run_root),
        "repo_id": args.repo_id,
        "assets_output_dir": str(out_dir),
        "batch_size": args.batch_size,
        "num_batches": num_batches,
        "max_frames": args.max_frames,
    }
    (run_root / "norm_stats_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote norm stats to {out_dir}")
    print(f"Wrote {(run_root / 'norm_stats_report.json')}")


if __name__ == "__main__":
    main()

