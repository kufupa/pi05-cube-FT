#!/usr/bin/env python3
"""Regenerate cube-single dual-camera videos for pi0.5 fine-tuning.

This script implements steps 1-3 of the data regeneration plan:
1) Dependency/preflight gate.
2) Full 100-episode generation (scene + wrist + side-by-side), capped at 200 steps.
3) QA checks and run freeze marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_npz() -> Path:
    return _project_root() / "cube_dataset" / "datasets" / "cube-single-play-v0.npz"


def _default_run_root() -> Path:
    return _project_root() / "cube_dataset" / "finetune_runs" / "pi05_cube_single_v1" / "data_gen"


def _ensure_pythonpath() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _assert_headless_contract() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        print(
            "FATAL: no DISPLAY and no MUJOCO_GL.\n"
            "Use either:\n"
            "  export MUJOCO_GL=egl   # or osmesa\n"
            'or run under xvfb-run -a -s "-screen 0 1024x768x24" ...',
            file=sys.stderr,
        )
        raise SystemExit(3)


def _import_checks() -> None:
    import gymnasium  # noqa: F401
    import imageio.v2  # noqa: F401
    import mujoco  # noqa: F401
    import ogbench  # noqa: F401


def _to_rgb_u8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _hstack_same_height(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    hl, wl, _ = left.shape
    hr, wr, _ = right.shape
    if hl == hr:
        return np.concatenate([left, right], axis=1)
    h = max(hl, hr)
    out_l = np.zeros((h, wl, 3), dtype=np.uint8)
    out_r = np.zeros((h, wr, 3), dtype=np.uint8)
    out_l[:hl, :wl] = left
    out_r[:hr, :wr] = right
    return np.concatenate([out_l, out_r], axis=1)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    stack = [np.asarray(f, dtype=np.uint8) for f in frames]
    if not stack:
        raise ValueError("No frames to write")
    with imageio.get_writer(
        path,
        fps=fps,
        codec="mpeg4",
        macro_block_size=1,
        ffmpeg_log_level="error",
        ffmpeg_params=["-q:v", "5"],
    ) as writer:
        for frame in stack:
            writer.append_data(frame)


def _episode_ranges(terminals: np.ndarray) -> list[tuple[int, int]]:
    ends = np.nonzero(terminals.astype(bool))[0]
    if len(ends) == 0:
        return []
    starts = np.concatenate([[0], ends[:-1] + 1])
    return [(int(s), int(e) + 1) for s, e in zip(starts, ends)]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _make_env():
    import gymnasium as gym
    from cube_dataset.pi05_joint_space import register_joint_target_envs

    register_joint_target_envs()
    env = gym.make(
        "cube-single-joint-target-v0",
        ob_type="states",
        render_mode="rgb_array",
        joint_scale=0.05,
    )
    return env


def _render_scene_wrist(env, raw, render_size: int) -> tuple[np.ndarray, np.ndarray]:
    wrist_camera = f"ur5e/{raw.WRIST_CAM_NAME}" if hasattr(raw, "WRIST_CAM_NAME") else "ur5e/wrist_cam"
    try:
        scene = _to_rgb_u8(raw.render(width=render_size, height=render_size))
    except Exception:
        scene = _to_rgb_u8(env.render())
    try:
        wrist = _to_rgb_u8(raw.render(camera=wrist_camera, width=render_size, height=render_size))
    except Exception:
        wrist = _to_rgb_u8(raw.render_wrist())
    return scene, wrist


def run_preflight(run_root: Path, fps: int, render_size: int) -> None:
    _assert_headless_contract()
    _import_checks()
    _ensure_pythonpath()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pre_dir = run_root / f"preflight_{ts}"
    pre_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env()
    try:
        env.reset(seed=0, options={"render_goal": True, "task_id": 1})
        raw = env.unwrapped
        if not hasattr(raw, "render_wrist"):
            raise RuntimeError("env.unwrapped has no render_wrist()")

        scene, wrist = _render_scene_wrist(env, raw, render_size)
        side = _hstack_same_height(scene, wrist)

        _write_mp4(pre_dir / "scene_probe.mp4", [scene, scene], fps=fps)
        _write_mp4(pre_dir / "wrist_probe.mp4", [wrist, wrist], fps=fps)
        _write_mp4(pre_dir / "side_by_side_probe.mp4", [side, side], fps=fps)

        probe_meta = {
            "status": "ok",
            "scene_shape": list(scene.shape),
            "wrist_shape": list(wrist.shape),
            "side_shape": list(side.shape),
            "fps": fps,
            "render_size": render_size,
            "timestamp": ts,
        }
        (pre_dir / "probe_meta.json").write_text(json.dumps(probe_meta, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()

    print(f"preflight: PASS ({pre_dir})")


def _select_episode_indices(total_episodes: int, n_episodes: int) -> list[int]:
    return list(range(min(total_episodes, n_episodes)))


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def generate_dataset(
    *,
    npz_path: Path,
    run_root: Path,
    run_id: str | None,
    n_episodes: int,
    max_steps: int,
    fps: int,
    render_size: int,
    resume: bool,
) -> Path:
    _assert_headless_contract()
    _import_checks()
    _ensure_pythonpath()

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset NPZ: {npz_path}")

    rid = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_root / rid
    videos_dir = run_dir / "videos"
    metadata_dir = run_dir / "metadata"
    manifest_dir = run_dir / "manifest"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    ds = np.load(npz_path)
    required = ("qpos", "qvel", "terminals")
    for k in required:
        if k not in ds:
            raise KeyError(f"Dataset is missing required key: {k}")

    ranges = _episode_ranges(ds["terminals"])
    if not ranges:
        raise RuntimeError("No terminal-marked episodes found in dataset")
    picked = _select_episode_indices(len(ranges), n_episodes)

    selection = {
        "dataset_npz": str(npz_path.resolve()),
        "dataset_npz_sha256": _sha256_file(npz_path),
        "selection_mode": "first_n_episode_order",
        "requested_n_episodes": int(n_episodes),
        "selected_n_episodes": int(len(picked)),
        "max_steps": int(max_steps),
        "fps": int(fps),
        "render_size": int(render_size),
        "episode_indices": picked,
        "episode_ranges": [
            {"episode_index": int(i), "start": int(ranges[i][0]), "end_exclusive": int(ranges[i][1])}
            for i in picked
        ],
    }
    _save_json(manifest_dir / "selection_manifest.json", selection)

    env = _make_env()
    try:
        raw = env.unwrapped
        if not hasattr(raw, "render_wrist"):
            raise RuntimeError("env.unwrapped has no render_wrist()")

        for rank, epi in enumerate(picked):
            s, e = ranges[epi]
            ep_len = e - s
            effective_steps = min(max_steps, max(0, ep_len - 1))
            scene_name = f"scene_ep_{rank:04d}.mp4"
            wrist_name = f"wrist_ep_{rank:04d}.mp4"
            side_name = f"side_by_side_ep_{rank:04d}.mp4"
            scene_path = videos_dir / scene_name
            wrist_path = videos_dir / wrist_name
            side_path = videos_dir / side_name
            meta_path = metadata_dir / f"episode_{rank:04d}.json"

            if resume and scene_path.exists() and wrist_path.exists() and side_path.exists() and meta_path.exists():
                try:
                    prev = json.loads(meta_path.read_text(encoding="utf-8"))
                    prev_frames = int(prev.get("n_frames", 0))
                    if prev_frames >= 2:
                        print(
                            f"[{rank + 1:03d}/{len(picked):03d}] epi={epi} reuse existing artifacts",
                            flush=True,
                        )
                        continue
                except Exception:
                    pass

            env.reset(seed=0)
            scene_frames: list[np.ndarray] = []
            wrist_frames: list[np.ndarray] = []
            side_frames: list[np.ndarray] = []

            q0 = np.asarray(ds["qpos"][s], dtype=np.float64)
            v0 = np.asarray(ds["qvel"][s], dtype=np.float64)
            raw.set_state(q0, v0)

            first_scene, first_wrist = _render_scene_wrist(env, raw, render_size)
            scene_frames.append(first_scene)
            wrist_frames.append(first_wrist)
            side_frames.append(_hstack_same_height(first_scene, first_wrist))

            # Add up to 200 next states -> total frames ~= 201 including initial.
            for idx in range(s + 1, s + effective_steps + 1):
                raw.set_state(
                    np.asarray(ds["qpos"][idx], dtype=np.float64),
                    np.asarray(ds["qvel"][idx], dtype=np.float64),
                )
                sc, wr = _render_scene_wrist(env, raw, render_size)
                scene_frames.append(sc)
                wrist_frames.append(wr)
                side_frames.append(_hstack_same_height(sc, wr))

            _write_mp4(scene_path, scene_frames, fps=fps)
            _write_mp4(wrist_path, wrist_frames, fps=fps)
            _write_mp4(side_path, side_frames, fps=fps)

            ep_meta = {
                "rank": int(rank),
                "episode_index": int(epi),
                "range_start": int(s),
                "range_end_exclusive": int(e),
                "episode_length_transitions": int(ep_len),
                "max_steps_requested": int(max_steps),
                "effective_steps_used": int(effective_steps),
                "state_index_start": int(s),
                "state_index_end_inclusive": int(s + effective_steps),
                "n_frames": int(len(scene_frames)),
                "fps": int(fps),
                "render_size": int(render_size),
                "files": {
                    "scene": scene_name,
                    "wrist": wrist_name,
                    "side_by_side": side_name,
                },
            }
            _save_json(meta_path, ep_meta)

            print(
                f"[{rank + 1:03d}/{len(picked):03d}] epi={epi} frames={len(scene_frames)} "
                f"-> {scene_name}, {wrist_name}, {side_name}",
                flush=True,
            )
    finally:
        env.close()
        ds.close()

    return run_dir


def qa_and_freeze(run_dir: Path, expected_episodes: int) -> None:
    videos_dir = run_dir / "videos"
    metadata_dir = run_dir / "metadata"
    manifest_dir = run_dir / "manifest"

    scene_files = sorted(videos_dir.glob("scene_ep_*.mp4"))
    wrist_files = sorted(videos_dir.glob("wrist_ep_*.mp4"))
    side_files = sorted(videos_dir.glob("side_by_side_ep_*.mp4"))
    meta_files = sorted(metadata_dir.glob("episode_*.json"))

    if len(scene_files) != expected_episodes:
        raise RuntimeError(f"scene videos mismatch: expected {expected_episodes}, got {len(scene_files)}")
    if len(wrist_files) != expected_episodes:
        raise RuntimeError(f"wrist videos mismatch: expected {expected_episodes}, got {len(wrist_files)}")
    if len(side_files) != expected_episodes:
        raise RuntimeError(f"side videos mismatch: expected {expected_episodes}, got {len(side_files)}")
    if len(meta_files) != expected_episodes:
        raise RuntimeError(f"metadata mismatch: expected {expected_episodes}, got {len(meta_files)}")

    # File size sanity + metadata frame sanity.
    frame_counts: list[int] = []
    for p in scene_files + wrist_files + side_files:
        if p.stat().st_size <= 0:
            raise RuntimeError(f"empty video file: {p}")
    for mf in meta_files:
        m = json.loads(mf.read_text(encoding="utf-8"))
        n_frames = int(m["n_frames"])
        frame_counts.append(n_frames)
        if n_frames < 2:
            raise RuntimeError(f"too few frames in {mf}: {n_frames}")
        for key in ("scene", "wrist", "side_by_side"):
            if not (videos_dir / m["files"][key]).exists():
                raise RuntimeError(f"missing referenced video {m['files'][key]} in {mf}")

    # Spot-check inventory (deterministic 10 sample ids).
    sample_ids = list(np.linspace(0, expected_episodes - 1, num=min(10, expected_episodes), dtype=int))
    inventory = {
        "expected_episodes": expected_episodes,
        "scene_count": len(scene_files),
        "wrist_count": len(wrist_files),
        "side_by_side_count": len(side_files),
        "metadata_count": len(meta_files),
        "frame_count_min": int(min(frame_counts)),
        "frame_count_max": int(max(frame_counts)),
        "sample_episode_ids_for_visual_spotcheck": [int(i) for i in sample_ids],
    }
    _save_json(manifest_dir / "qa_inventory.json", inventory)

    selection_manifest = manifest_dir / "selection_manifest.json"
    qa_inventory = manifest_dir / "qa_inventory.json"

    ready = f"""# RUN READY FOR TRAINING

Run directory: `{run_dir}`

## Summary
- Episodes: {expected_episodes}
- Videos per episode: scene + wrist + side_by_side
- Metadata per episode: yes
- Frame count range: {min(frame_counts)}..{max(frame_counts)}

## Hashes
- selection_manifest.json: `{_sha256_file(selection_manifest)}`
- qa_inventory.json: `{_sha256_file(qa_inventory)}`

## Spot-check suggestion
Open these episode ids first for visual review:
{", ".join(str(i) for i in sample_ids)}
"""
    (run_dir / "RUN_READY_FOR_TRAINING.md").write_text(ready, encoding="utf-8")

    print(f"qa: PASS ({run_dir})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=_default_npz())
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--render-size", type=int, default=160)
    parser.add_argument("--no-resume", action="store_true", help="Regenerate episodes even if outputs already exist")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--qa-only", type=Path, default=None, help="Run QA/freeze only on an existing run dir")
    args = parser.parse_args()

    _ensure_pythonpath()

    if args.preflight:
        run_preflight(args.run_root, fps=args.fps, render_size=args.render_size)
        return

    if args.qa_only is not None:
        qa_and_freeze(args.qa_only.resolve(), expected_episodes=args.n_episodes)
        return

    run_dir = generate_dataset(
        npz_path=args.npz.resolve(),
        run_root=args.run_root.resolve(),
        run_id=args.run_id,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        render_size=args.render_size,
        resume=(not args.no_resume),
    )
    qa_and_freeze(run_dir, expected_episodes=args.n_episodes)
    print(f"Done. run_dir={run_dir}")


if __name__ == "__main__":
    main()

