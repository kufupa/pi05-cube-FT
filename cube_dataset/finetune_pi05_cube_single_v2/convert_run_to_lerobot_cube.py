#!/usr/bin/env python3
"""Convert trimmed cube run artifacts into a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _read_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    frames: list[np.ndarray] = []
    try:
        for fr in reader:
            arr = np.asarray(fr)
            if arr.ndim != 3:
                continue
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            frames.append(np.ascontiguousarray(arr, dtype=np.uint8))
    finally:
        reader.close()
    return frames


def _gripper_open01_from_qpos(qpos_row: np.ndarray) -> float:
    return float(np.clip(float(qpos_row[6]) / 0.8, 0.0, 1.0))


def _model_action_from_qpos_next(qpos_next: np.ndarray) -> np.ndarray:
    action = np.zeros((7,), dtype=np.float32)
    action[:6] = np.asarray(qpos_next[:6], dtype=np.float32)
    # Model convention in this pipeline path: 0=open, 1=closed.
    action[6] = 1.0 - _gripper_open01_from_qpos(qpos_next)
    # Align joint 0 with model distribution.
    action[0] += np.pi
    return action


def _ensure_lerobot_imports() -> None:
    try:
        import lerobot.common.datasets.lerobot_dataset  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "lerobot is not available in this environment. "
            "Run this script with external/openpi uv environment."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=None, help="LeRobot repo id, e.g. local/pi05_cube_single_v2")
    parser.add_argument("--prompt", type=str, default="pick up the red cube")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=100)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    _ensure_lerobot_imports()
    lerobot_home = run_root / "lerobot_home"
    lerobot_home.mkdir(parents=True, exist_ok=True)
    os.environ["LEROBOT_HOME"] = str(lerobot_home)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    q = np.load(args.source_npz.resolve())
    qpos = np.asarray(q["qpos"], dtype=np.float32)
    metadata_dir = run_dir / "metadata"
    videos_dir = run_dir / "videos"

    episode_meta_files = sorted(metadata_dir.glob("episode_*.json"))[: args.max_episodes]
    if not episode_meta_files:
        raise RuntimeError(f"No episode metadata files found in {metadata_dir}")

    # Probe one image to set feature shape.
    probe_meta = json.loads(episode_meta_files[0].read_text(encoding="utf-8"))
    probe_scene = _read_frames(videos_dir / probe_meta["files"]["scene"])
    probe_wrist = _read_frames(videos_dir / probe_meta["files"]["wrist"])
    if not probe_scene or not probe_wrist:
        raise RuntimeError("Probe videos are empty")
    h, w, _ = probe_scene[0].shape
    hw, ww, _ = probe_wrist[0].shape

    repo_id = args.repo_id or f"local/pi05_cube_single_v2_{run_root.name}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=int(args.fps),
        features={
            "base_rgb": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
            "wrist_rgb": {"dtype": "image", "shape": (hw, ww, 3), "names": ["height", "width", "channel"]},
            "joints": {"dtype": "float32", "shape": (6,), "names": ["joints"]},
            "gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    accepted = 0
    dropped = 0
    episode_reports: list[dict] = []

    for meta_path in episode_meta_files:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        scene_path = videos_dir / m["files"]["scene"]
        wrist_path = videos_dir / m["files"]["wrist"]
        scene_frames = _read_frames(scene_path)
        wrist_frames = _read_frames(wrist_path)
        n_frames = min(len(scene_frames), len(wrist_frames))
        if n_frames < 2:
            dropped += 1
            episode_reports.append(
                {
                    "episode": meta_path.name,
                    "status": "dropped",
                    "reason": "too_few_frames",
                    "scene_frames": len(scene_frames),
                    "wrist_frames": len(wrist_frames),
                }
            )
            continue

        start_idx = int(m["state_index_start"])
        end_idx = int(m["state_index_end_inclusive"])
        max_available = max(0, end_idx - start_idx + 1)
        use_frames = min(n_frames, max_available)
        if use_frames < 2:
            dropped += 1
            episode_reports.append(
                {
                    "episode": meta_path.name,
                    "status": "dropped",
                    "reason": "state_range_too_short",
                    "use_frames": use_frames,
                }
            )
            continue

        for t in range(use_frames):
            idx_curr = start_idx + t
            idx_next = min(idx_curr + 1, end_idx)
            qpos_curr = qpos[idx_curr]
            qpos_next = qpos[idx_next]

            joints = np.asarray(qpos_curr[:6], dtype=np.float32)
            gripper_open = _gripper_open01_from_qpos(qpos_curr)
            action = _model_action_from_qpos_next(qpos_next)

            dataset.add_frame(
                {
                    "base_rgb": scene_frames[t],
                    "wrist_rgb": wrist_frames[t],
                    "joints": joints,
                    "gripper": np.asarray([gripper_open], dtype=np.float32),
                    "actions": action,
                    "task": args.prompt,
                }
            )

        dataset.save_episode()
        accepted += 1
        episode_reports.append(
            {
                "episode": meta_path.name,
                "status": "accepted",
                "frames_used": use_frames,
                "state_start": start_idx,
                "state_end": end_idx,
            }
        )

    report = {
        "status": "pass" if accepted > 0 else "fail",
        "run_dir": str(run_dir),
        "run_root": str(run_root),
        "repo_id": repo_id,
        "lerobot_home": str(lerobot_home),
        "accepted_episodes": accepted,
        "dropped_episodes": dropped,
        "requested_episodes": len(episode_meta_files),
        "episode_reports": episode_reports,
    }
    (run_root / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_root / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "source_npz": str(args.source_npz.resolve()),
                "repo_id": repo_id,
                "prompt": args.prompt,
                "fps": args.fps,
                "max_episodes": args.max_episodes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {run_root / 'conversion_report.json'}")
    if accepted <= 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

