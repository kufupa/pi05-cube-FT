"""Minimal LeRobot dataset API compatible with OpenPI training code.

This local implementation supports the subset used by:
- OpenPI `openpi.training.data_loader`
- cube conversion script in this repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

HF_LEROBOT_HOME = Path(os.environ.get("LEROBOT_HOME", "~/.cache/lerobot")).expanduser().resolve()


def _resolve_repo_path(repo_id: str) -> Path:
    p = Path(repo_id)
    if p.is_absolute():
        return p
    return (HF_LEROBOT_HOME / repo_id).resolve()


def _meta_path(repo_path: Path) -> Path:
    return repo_path / "meta.json"


def _episodes_dir(repo_path: Path) -> Path:
    return repo_path / "episodes"


class LeRobotDatasetMetadata:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.repo_path = _resolve_repo_path(repo_id)
        mp = _meta_path(self.repo_path)
        if not mp.exists():
            raise FileNotFoundError(f"LeRobot meta not found: {mp}")
        meta = json.loads(mp.read_text(encoding="utf-8"))
        self.fps = int(meta["fps"])
        self.tasks = {int(k): str(v) for k, v in dict(meta.get("tasks", {})).items()}
        self.features = dict(meta.get("features", {}))


class LeRobotDataset:
    """Small local LeRobot-compatible dataset."""

    def __init__(self, repo_id: str, delta_timestamps: dict[str, list[float]] | None = None):
        self.repo_id = repo_id
        self.repo_path = _resolve_repo_path(repo_id)
        self.delta_timestamps = delta_timestamps or {}
        self._mode = "read"
        self._cache: dict[Path, dict[str, np.ndarray]] = {}

        meta = LeRobotDatasetMetadata(repo_id)
        self._fps = meta.fps
        self._tasks = dict(meta.tasks)
        self._features = dict(meta.features)
        self._offsets = {
            k: [int(round(float(dt) * self._fps)) for dt in dts] for k, dts in self.delta_timestamps.items()
        }

        self._episode_files = sorted(_episodes_dir(self.repo_path).glob("episode_*.npz"))
        self._index: list[tuple[Path, int]] = []
        for ep in self._episode_files:
            with np.load(ep, allow_pickle=False) as z:
                keys = list(z.keys())
                if not keys:
                    continue
                n = int(z[keys[0]].shape[0])
            self._index.extend((ep, i) for i in range(n))

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        robot_type: str,
        fps: int,
        features: dict[str, Any],
        image_writer_threads: int = 1,  # kept for API compatibility
        image_writer_processes: int = 1,  # kept for API compatibility
    ) -> "LeRobotDataset":
        del image_writer_threads, image_writer_processes
        repo_path = _resolve_repo_path(repo_id)
        _episodes_dir(repo_path).mkdir(parents=True, exist_ok=True)
        mp = _meta_path(repo_path)
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            tasks = meta.get("tasks", {})
        else:
            tasks = {}
        meta = {
            "repo_id": repo_id,
            "robot_type": robot_type,
            "fps": int(fps),
            "features": features,
            "tasks": tasks,
        }
        mp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        ds = cls.__new__(cls)
        ds.repo_id = repo_id
        ds.repo_path = repo_path
        ds.delta_timestamps = {}
        ds._mode = "write"
        ds._fps = int(fps)
        ds._tasks = {int(k): str(v) for k, v in dict(tasks).items()}
        ds._features = dict(features)
        ds._task_to_idx = {v: int(k) for k, v in ds._tasks.items()}
        ds._episode_idx = len(list(_episodes_dir(repo_path).glob("episode_*.npz")))
        ds._write_frames: list[dict[str, Any]] = []
        ds._cache = {}
        ds._episode_files = []
        ds._index = []
        ds._offsets = {}
        return ds

    def add_frame(self, frame: dict[str, Any]) -> None:
        if self._mode != "write":
            raise RuntimeError("add_frame is only valid in write mode")
        self._write_frames.append(dict(frame))

    def save_episode(self) -> None:
        if self._mode != "write":
            raise RuntimeError("save_episode is only valid in write mode")
        if not self._write_frames:
            return

        frames = self._write_frames
        self._write_frames = []
        n = len(frames)
        task_indices = np.zeros((n,), dtype=np.int32)
        arrays: dict[str, np.ndarray] = {}
        for key in self._features:
            vals = []
            for i, fr in enumerate(frames):
                vals.append(np.asarray(fr[key]))
                if "task" in fr:
                    task = str(fr["task"])
                    if task not in self._task_to_idx:
                        next_idx = 0 if not self._task_to_idx else (max(self._task_to_idx.values()) + 1)
                        self._task_to_idx[task] = int(next_idx)
                        self._tasks[int(next_idx)] = task
                    task_indices[i] = self._task_to_idx[task]
            arrays[key] = np.stack(vals, axis=0)

        arrays["task_index"] = task_indices
        ep_path = _episodes_dir(self.repo_path) / f"episode_{self._episode_idx:06d}.npz"
        np.savez_compressed(ep_path, **arrays)
        self._episode_idx += 1

        # Persist task updates.
        mp = _meta_path(self.repo_path)
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["tasks"] = {str(k): v for k, v in sorted(self._tasks.items(), key=lambda kv: kv[0])}
        mp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def _load_episode(self, ep_path: Path) -> dict[str, np.ndarray]:
        if ep_path not in self._cache:
            with np.load(ep_path, allow_pickle=False) as z:
                self._cache[ep_path] = {k: np.asarray(v) for k, v in z.items()}
        return self._cache[ep_path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        ep_path, t = self._index[int(index)]
        episode = self._load_episode(ep_path)
        out: dict[str, Any] = {}
        for k, arr in episode.items():
            if k == "task_index":
                out[k] = np.asarray(arr[t], dtype=np.int32)
                continue
            if k in self._offsets:
                offs = self._offsets[k]
                seq = [arr[min(max(0, t + off), arr.shape[0] - 1)] for off in offs]
                out[k] = np.stack(seq, axis=0)
            else:
                out[k] = arr[t]
        return out

    def __len__(self) -> int:
        return len(self._index)

