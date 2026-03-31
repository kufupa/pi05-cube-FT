#!/usr/bin/env python3
"""Closed-loop pi0.5-base (UR5e transforms) rollouts on OGBench cube joint-target env."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base"
INSTRUCTION = "pick up the solid red cube"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cube_dataset_root() -> Path:
    return Path(__file__).resolve().parent


def assert_display_or_mujoco_gl() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        print(
            "FATAL: no DISPLAY and no MUJOCO_GL. Use e.g.\n"
            '  xvfb-run -a -s "-screen 0 1024x768x24" .../python cube_dataset/run_pi05_base_ur5e_rollouts.py',
            file=sys.stderr,
        )
        sys.exit(3)


def get_joints6_and_gripper_open01(env) -> tuple[np.ndarray, float]:
    raw = env.unwrapped
    qpos = raw._data.qpos
    joints = np.asarray(qpos[raw._arm_joint_ids], dtype=np.float32).reshape(6)
    g_raw = float(qpos[raw._gripper_opening_joint_id])
    g_open = float(np.clip(g_raw / 0.8, 0.0, 1.0))
    return joints, g_open


def build_state_7(env) -> "torch.Tensor":
    import torch

    j, g = get_joints6_and_gripper_open01(env)
    s = np.zeros(7, dtype=np.float32)
    s[:6] = j
    s[6] = g
    return torch.from_numpy(s).view(1, 7)


def _frame_to_chw01(frame) -> "torch.Tensor":
    import torch

    u8 = np.asarray(frame)
    if u8.dtype != np.uint8:
        u8 = np.clip(u8, 0, 255).astype(np.uint8)
    if u8.shape[-1] == 4:
        u8 = u8[..., :3]
    f = u8.astype(np.float32) / 255.0
    chw = np.transpose(f, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def render_chw01(env) -> "torch.Tensor":
    return _frame_to_chw01(env.render())


def render_wrist_chw01(env) -> "torch.Tensor":
    raw = env.unwrapped
    if hasattr(raw, "render_wrist"):
        return _frame_to_chw01(raw.render_wrist())
    return _frame_to_chw01(np.zeros((200, 200, 3), dtype=np.uint8))


def map_pi05_ur5e_to_ogbench_joint7(
    a7: "torch.Tensor",
    current_qpos6: np.ndarray,
    *,
    joint_scale: float,
) -> np.ndarray:
    from src.envs.droid.observation_openpi_ur5e import JOINT_0_OFFSET

    a = a7.detach().cpu().float().reshape(-1).numpy()
    if a.size < 7:
        raise ValueError(f"expected at least 7 action dims, got {a.size}")

    a[0] -= JOINT_0_OFFSET

    out = np.zeros(7, dtype=np.float32)
    arm_delta = (a[:6].astype(np.float64) - current_qpos6.astype(np.float64)) / float(joint_scale)
    out[:6] = np.clip(arm_delta, -1.0, 1.0).astype(np.float32)
    # OpenPI gripper is 0=open, 1=closed; OGBench expects 0=closed, 1=open.
    out[6] = float(np.clip(1.0 - float(a[6]), 0.0, 1.0))
    return out


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
    raw = env.unwrapped
    joint_scale = float(getattr(raw, "_joint_scale", 0.05))

    frames: list[np.ndarray] = []
    first = np.asarray(env.render(), dtype=np.uint8)
    if first.shape[-1] == 4:
        first = first[..., :3]
    frames.append(first)

    terminated = False
    truncated = False
    info_last: dict[str, Any] = {}
    steps = 0

    for t in range(max_steps):
        rgb = render_chw01(env)
        wrist_rgb = render_wrist_chw01(env)
        joints6, g_open = get_joints6_and_gripper_open01(env)
        observation = {
            "obs": rgb,
            "wrist_obs": wrist_rgb,
            "state": build_state_7(env),
            "joints_6": joints6,
            "gripper_open_01": g_open,
            "instruction": INSTRUCTION,
            "timestep": t,
        }
        a7 = policy.act(observation)
        a_env = map_pi05_ur5e_to_ogbench_joint7(a7, joints6, joint_scale=joint_scale)
        _obs, _reward, terminated, truncated, info_last = env.step(a_env)
        steps += 1
        fr = np.asarray(env.render(), dtype=np.uint8)
        if fr.shape[-1] == 4:
            fr = fr[..., :3]
        frames.append(fr)
        if terminated or truncated:
            break

    write_mp4(video_path, frames, fps=fps)
    return {
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(info_last.get("success", False)),
        "video_path": str(video_path),
        "reset_seed": seed,
        "control": "joint",
        "policy": "pi05_base_ur5e",
        "joint_scale": joint_scale,
    }


def smoke_test(checkpoint: str, *, require_openpi: bool = False, joint_scale: float = 0.05) -> None:
    print("=== pi05_base_ur5e smoke (joint) ===")
    assert_display_or_mujoco_gl()

    import gymnasium as gym
    import imageio  # noqa: F401
    import jax  # noqa: F401
    import mujoco  # noqa: F401
    import ogbench  # noqa: F401

    from cube_dataset.pi05_joint_space import register_joint_target_envs
    from src.vla.pi05_ur5e import Pi05UR5ePolicy

    meta0 = {"reset_seed": 0, "reset_options": {"render_goal": True}, "task_id": 1}

    register_joint_target_envs()
    env = gym.make(
        "cube-single-joint-target-v0",
        ob_type="states",
        render_mode="rgb_array",
        joint_scale=joint_scale,
    )
    try:
        ob, _ = env.reset(seed=0, options=reset_options_from_meta(meta0))
        rgb = render_chw01(env)
        assert rgb.shape[1] == 3, rgb.shape
        st = build_state_7(env)
        assert st.shape == (1, 7), st.shape

        raw = env.unwrapped
        qarm = np.asarray(raw._data.qpos[raw._arm_joint_ids], dtype=np.float64)
        if not np.allclose(ob[:6].astype(np.float64), qarm, rtol=0, atol=1e-5):
            print("FATAL: state obs[:6] != arm qpos", file=sys.stderr)
            sys.exit(6)

        policy = Pi05UR5ePolicy(checkpoint_path=checkpoint)
        if require_openpi and not policy.uses_openpi():
            reason = policy._openpi_failed_reason or "unknown"
            print(f"FATAL: --require-openpi but OpenPI did not load: {reason}", file=sys.stderr)
            sys.exit(8)

        wrist_rgb = render_wrist_chw01(env)
        joints6, g_open = get_joints6_and_gripper_open01(env)
        observation = {
            "obs": rgb,
            "wrist_obs": wrist_rgb,
            "state": st,
            "joints_6": joints6,
            "gripper_open_01": g_open,
            "instruction": INSTRUCTION,
            "timestep": 0,
        }
        a7 = policy.act(observation)
        assert a7.shape[-1] == 7, a7.shape
        a_env = map_pi05_ur5e_to_ogbench_joint7(a7, joints6, joint_scale=joint_scale)
        assert a_env.shape == (7,), a_env.shape
        env.step(a_env)

        td = Path(tempfile.mkdtemp(prefix="pi05_base_ur5e_smoke_"))
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
        help="Output directory for MP4 + results.jsonl",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of episodes")
    parser.add_argument("--start-index", type=int, default=0, help="First sample index")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument(
        "--joint-scale",
        type=float,
        default=0.05,
        help="CubeEnvJointTargetDelta joint_scale",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="cube-single-joint-target-v0",
        help="Gym env id",
    )
    parser.add_argument(
        "--require-openpi",
        action="store_true",
        help="Exit non-zero if OpenPI checkpoint did not load",
    )
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(
            args.checkpoint,
            require_openpi=args.require_openpi,
            joint_scale=args.joint_scale,
        )
        return

    assert_display_or_mujoco_gl()

    import gymnasium as gym
    import ogbench  # noqa: F401

    from cube_dataset.pi05_joint_space import register_joint_target_envs
    from src.vla.pi05_ur5e import Pi05UR5ePolicy

    register_joint_target_envs()
    vlm = args.vlm_root or (_cube_dataset_root() / "goal_images" / "vlm_start_goal")
    fam = vlm / "cube-single-v0".replace("/", "_")

    out_dir = args.out_dir or (_cube_dataset_root() / "pi05_base_ur5e_rollouts")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    policy = Pi05UR5ePolicy(checkpoint_path=args.checkpoint)
    uses_openpi = policy.uses_openpi()
    if args.require_openpi and not uses_openpi:
        reason = policy._openpi_failed_reason or "unknown"
        print(f"FATAL: --require-openpi but OpenPI did not load: {reason}", file=sys.stderr)
        sys.exit(8)

    env = gym.make(
        args.env_id,
        ob_type="states",
        render_mode="rgb_array",
        joint_scale=args.joint_scale,
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
                    "env_id": args.env_id,
                    "joint_scale_arg": float(args.joint_scale),
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
