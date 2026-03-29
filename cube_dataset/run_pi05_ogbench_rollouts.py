#!/usr/bin/env python3
"""Closed-loop π0.5-DROID (OpenPI) rollouts on OGBench cube-single-v0 from vlm_start_goal metas.

Environment: use stable-worldmodel's venv (ogbench, mujoco, gymnasium, torch, openpi, imageio).

  cd /path/to/project && export PYTHONPATH=/path/to/project && export MUJOCO_GL=glfw \\
    xvfb-run -a -s "-screen 0 1024x768x24" \\
    stable-worldmodel/.venv/bin/python cube_dataset/run_pi05_ogbench_rollouts.py --smoke-test

One-time venv fixes (if OpenPI fails to load): from ``stable-worldmodel/`` run
``uv pip install --python .venv/bin/python transformers==4.53.2 torchvision==0.22.1 tfrecord``,
then copy OpenPI's patched HF modules into ``site-packages/transformers/`` (see OpenPI README).

PBS uses ``MUJOCO_GL=glfw`` + xvfb; JAX uses the requested GPU.

Default checkpoint: gs://openpi-assets/checkpoints/pi05_droid (override with --checkpoint or PI05_OGBENCH_CHECKPOINT).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid"
BRIDGE_VARIANT = "scaled_cartesian"
INSTRUCTION = (
    "Move the solid red cube on the table to overlap the faded red target cube."
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cube_dataset_root() -> Path:
    return Path(__file__).resolve().parent


def assert_display_or_mujoco_gl() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        print(
            "FATAL: no DISPLAY and no MUJOCO_GL. Use e.g.\n"
            '  xvfb-run -a -s "-screen 0 1024x768x24" .../python cube_dataset/run_pi05_ogbench_rollouts.py',
            file=sys.stderr,
        )
        sys.exit(3)


def build_state_14(env) -> "torch.Tensor":
    """Pack UR5 qpos (6) + gripper opening [0,1] into 14-D tensor for OpenPI (gripper at index -1)."""
    import torch

    raw = env.unwrapped
    qpos = raw._data.qpos
    arm = np.asarray(qpos[raw._arm_joint_ids], dtype=np.float32).reshape(6)
    g_raw = float(qpos[raw._gripper_opening_joint_id])
    g_open = float(np.clip(g_raw / 0.8, 0.0, 1.0))
    s = np.zeros(14, dtype=np.float32)
    s[:6] = arm
    s[6] = 0.0
    s[-1] = g_open
    return torch.from_numpy(s).view(1, 14)


def render_chw01(env) -> "torch.Tensor":
    """RGB from env.render() as float CHW [0,1], shape (1,3,H,W)."""
    import torch

    frame = env.render()
    u8 = np.asarray(frame)
    if u8.dtype != np.uint8:
        u8 = np.clip(u8, 0, 255).astype(np.uint8)
    if u8.shape[-1] == 4:
        u8 = u8[..., :3]
    f = u8.astype(np.float32) / 255.0
    chw = np.transpose(f, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def map_pi05_to_ogbench_scaled_cartesian(
    a8: "torch.Tensor", raw_env
) -> np.ndarray:
    """Map policy [1,8] to OGBench normalized 5-D EE action in [-1, 1]."""
    a = a8.detach().cpu().float().reshape(-1).numpy()
    d_trans = a[0:3].astype(np.float64) * 0.03
    d_yaw = float(a[3]) * 0.5
    d_grip = float(a[4])
    physical = np.array(
        [d_trans[0], d_trans[1], d_trans[2], d_yaw, d_grip], dtype=np.float64
    )
    action_range = np.array([0.05, 0.05, 0.05, 0.3, 1.0], dtype=np.float64)
    n = np.clip(physical / action_range, -1.0, 1.0).astype(np.float32)
    return n


def write_mp4(path: Path, frames_hwc_u8: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    stack = [np.asarray(f, dtype=np.uint8) for f in frames_hwc_u8]
    if not stack:
        raise ValueError("no frames for MP4")
    try:
        imageio.mimwrite(path, stack, fps=fps, codec="libx264")
    except Exception:
        imageio.mimwrite(path, stack, fps=fps, plugin="ffmpeg")


def load_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_info_as_numpy(task_info: dict[str, Any]) -> dict[str, Any]:
    """JSON meta stores lists; OGBench expects ndarray for init_xyzs / goal_xyzs indexing."""
    out = dict(task_info)
    if "init_xyzs" in out:
        out["init_xyzs"] = np.asarray(out["init_xyzs"], dtype=np.float64)
    if "goal_xyzs" in out:
        out["goal_xyzs"] = np.asarray(out["goal_xyzs"], dtype=np.float64)
    return out


def reset_options_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    opts: dict[str, Any] = dict(meta.get("reset_options") or {})
    if meta.get("task_info") is not None:
        opts["task_info"] = _task_info_as_numpy(meta["task_info"])
    elif meta.get("task_id") is not None:
        opts["task_id"] = int(meta["task_id"])
    return opts


def rollout_one(
    env,
    policy,
    *,
    meta: dict[str, Any],
    max_steps: int,
    fps: int,
    video_path: Path,
) -> dict[str, Any]:
    seed = int(meta["reset_seed"])
    opts = reset_options_from_meta(meta)
    env.reset(seed=seed, options=opts)

    frames: list[np.ndarray] = []
    first = np.asarray(env.render(), dtype=np.uint8)
    if first.shape[-1] == 4:
        first = first[..., :3]
    frames.append(first)

    obs_t = None
    terminated = False
    truncated = False
    info_last: dict[str, Any] = {}
    steps = 0

    for t in range(max_steps):
        rgb = render_chw01(env)
        st = build_state_14(env)
        observation = {
            "obs": rgb,
            "state": st,
            "instruction": INSTRUCTION,
            "timestep": t,
        }
        a8 = policy.act(observation)
        a5 = map_pi05_to_ogbench_scaled_cartesian(a8, env.unwrapped)
        _obs, _reward, terminated, truncated, info_last = env.step(a5)
        steps += 1
        fr = np.asarray(env.render(), dtype=np.uint8)
        if fr.shape[-1] == 4:
            fr = fr[..., :3]
        frames.append(fr)
        if terminated or truncated:
            break

    write_mp4(video_path, frames, fps=fps)
    success = bool(info_last.get("success", False))
    return {
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": success,
        "video_path": str(video_path),
        "reset_seed": seed,
        "bridge_variant": BRIDGE_VARIANT,
    }


def smoke_test(checkpoint: str) -> None:
    print("=== pi05_ogbench smoke ===")
    assert_display_or_mujoco_gl()

    import gymnasium as gym
    import imageio.v2 as imageio
    import mujoco  # noqa: F401
    import ogbench  # noqa: F401
    import torch

    from src.vla.pi05_droid import Pi05DroidPolicy

    env = gym.make("cube-single-v0", ob_type="states", render_mode="rgb_array")
    try:
        meta0 = {
            "reset_seed": 0,
            "reset_options": {"render_goal": True},
            "task_id": 1,
        }
        env.reset(seed=0, options=reset_options_from_meta(meta0))
        rgb = render_chw01(env)
        assert rgb.shape[1] == 3, rgb.shape
        st = build_state_14(env)
        assert st.shape == (1, 14), st.shape

        policy = Pi05DroidPolicy(checkpoint_path=checkpoint)
        observation = {
            "obs": rgb,
            "state": st,
            "instruction": INSTRUCTION,
            "timestep": 0,
        }
        a8 = policy.act(observation)
        assert a8.shape[-1] == 8, a8.shape
        a5 = map_pi05_to_ogbench_scaled_cartesian(a8, env.unwrapped)
        assert a5.shape == (5,), a5.shape
        env.step(a5)

        td = Path(tempfile.mkdtemp(prefix="pi05_smoke_mp4_"))
        p = td / "probe.mp4"
        f1 = np.asarray(env.render(), dtype=np.uint8)[..., :3]
        f2 = f1.copy()
        write_mp4(p, [f1, f2], fps=2)
        if not p.exists() or p.stat().st_size < 10:
            print("FATAL: MP4 probe write failed", file=sys.stderr)
            sys.exit(5)
        print(f"  openpi_loaded={policy.uses_openpi()}  mp4_bytes={p.stat().st_size}")
    finally:
        env.close()
    print("smoke test: PASS")


def main() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PI05_OGBENCH_CHECKPOINT", DEFAULT_CHECKPOINT),
        help=f"OpenPI checkpoint (default: {DEFAULT_CHECKPOINT} or env PI05_OGBENCH_CHECKPOINT)",
    )
    parser.add_argument(
        "--vlm-root",
        type=Path,
        default=None,
        help="vlm_start_goal root (default: cube_dataset/goal_images/vlm_start_goal)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for MP4 + results.jsonl (default: cube_dataset/pi05_rollouts)",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of episodes (samples)")
    parser.add_argument("--start-index", type=int, default=0, help="First sample index (e.g. resume)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode (cube-single-v0 default horizon)",
    )
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args.checkpoint)
        return

    assert_display_or_mujoco_gl()

    import gymnasium as gym
    import ogbench  # noqa: F401

    from src.vla.pi05_droid import Pi05DroidPolicy

    vlm = args.vlm_root or (
        _cube_dataset_root() / "goal_images" / "vlm_start_goal"
    )
    env_id = "cube-single-v0"
    fam = vlm / env_id.replace("/", "_")

    out_dir = args.out_dir or (_cube_dataset_root() / "pi05_rollouts")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    policy = Pi05DroidPolicy(checkpoint_path=args.checkpoint)
    uses_openpi = policy.uses_openpi()

    env = gym.make(
        env_id,
        ob_type="states",
        render_mode="rgb_array",
    )
    try:
        end = args.start_index + args.n
        for idx in range(args.start_index, end):
            sample_dir = fam / f"sample_{idx:04d}"
            meta_path = sample_dir / "meta.json"
            if not meta_path.is_file():
                print(f"FATAL: missing {meta_path}", file=sys.stderr)
                sys.exit(4)
            meta = load_meta(meta_path)
            video_path = out_dir / f"sample_{idx:04d}.mp4"
            print(f"=== rollout sample_{idx:04d} -> {video_path.name} ===")
            row = rollout_one(
                env,
                policy,
                meta=meta,
                max_steps=args.max_steps,
                fps=args.fps,
                video_path=video_path,
            )
            row.update(
                {
                    "sample_index": idx,
                    "sample_dir": str(sample_dir),
                    "checkpoint": args.checkpoint,
                    "uses_openpi": uses_openpi,
                    "env_id": env_id,
                }
            )
            if policy._openpi_failed_reason and not uses_openpi:
                row["openpi_failed_reason"] = policy._openpi_failed_reason
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  success={row['success']} steps={row['steps']}")
    finally:
        env.close()
    print(f"Done. Wrote under {out_dir}")


if __name__ == "__main__":
    main()
