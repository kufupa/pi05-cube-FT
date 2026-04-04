#!/usr/bin/env python3
"""Fast compatibility precheck for OpenPI pi05_droid_finetune datasets."""

from __future__ import annotations

import argparse
import sys

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a LeRobot dataset for pi05_droid_finetune.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. org/my_droid_dataset")
    args = parser.parse_args()

    try:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            episodes=[0],
            download_videos=False,
            # Minimal timestamp request so this remains a lightweight precheck.
            delta_timestamps={"action": [0.0]},
        )
        sample = dataset[0]
    except Exception as exc:  # pragma: no cover - this script is executed on-cluster
        print(f"[precheck] FAIL repo_id={args.repo_id}")
        print(f"[precheck] {type(exc).__name__}: {exc}")
        print(
            "[precheck] Hint: convert your DROID data with "
            "external/openpi/examples/droid/convert_droid_data_to_lerobot.py "
            "using this repo's pinned dependencies."
        )
        return 2

    required = ("observation.state", "observation.images.top", "observation.images.wrist", "action")
    missing = [k for k in required if k not in sample]
    if missing:
        print(f"[precheck] FAIL repo_id={args.repo_id}")
        print(f"[precheck] Missing required keys: {missing}")
        return 3

    print(f"[precheck] PASS repo_id={args.repo_id}")
    print(f"[precheck] len={len(dataset)} sample_keys={sorted(sample.keys())[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
