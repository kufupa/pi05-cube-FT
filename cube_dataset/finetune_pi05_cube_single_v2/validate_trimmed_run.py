#!/usr/bin/env python3
"""Validate trimmed cube run integrity and decoded frame counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio


def _count_frames(path: Path) -> int:
    reader = imageio.get_reader(path)
    n = 0
    try:
        for _ in reader:
            n += 1
    finally:
        reader.close()
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=300)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    videos_dir = run_dir / "videos"
    metadata_dir = run_dir / "metadata"
    manifest_dir = run_dir / "manifest"
    ready_md = run_dir / "RUN_READY_FOR_TRAINING.md"

    failures: list[dict] = []
    frame_counts: dict[str, int] = {}

    if not videos_dir.exists():
        failures.append({"error": "missing_videos_dir", "path": str(videos_dir)})
    if not metadata_dir.exists():
        failures.append({"error": "missing_metadata_dir", "path": str(metadata_dir)})
    if not manifest_dir.exists():
        failures.append({"error": "missing_manifest_dir", "path": str(manifest_dir)})
    if not ready_md.exists():
        failures.append({"error": "missing_run_ready_file", "path": str(ready_md)})

    mp4s = sorted(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
    if len(mp4s) != args.expected_videos:
        failures.append(
            {
                "error": "video_count_mismatch",
                "expected": args.expected_videos,
                "actual": len(mp4s),
            }
        )

    for p in mp4s:
        try:
            n = _count_frames(p)
            if n <= 0:
                failures.append({"error": "empty_video", "file": p.name})
            else:
                frame_counts[p.name] = int(n)
        except Exception as exc:  # pragma: no cover
            failures.append({"error": "decode_failed", "file": p.name, "message": str(exc)})

    counts = list(frame_counts.values())
    report = {
        "status": "fail" if failures else "pass",
        "run_dir": str(run_dir),
        "expected_videos": args.expected_videos,
        "actual_videos": len(mp4s),
        "decoded_frame_count_min": min(counts) if counts else None,
        "decoded_frame_count_max": max(counts) if counts else None,
        "decoded_frame_count_mean": (sum(counts) / len(counts)) if counts else None,
        "failures": failures,
    }
    (run_root / "validate_trimmed_run_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_root / "decoded_frame_counts.json").write_text(
        json.dumps(frame_counts, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {run_root / 'validate_trimmed_run_report.json'}")
    if failures:
        print("Validation failed. See validate_trimmed_run_report.json", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

