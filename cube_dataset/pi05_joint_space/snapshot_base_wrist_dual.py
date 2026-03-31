#!/usr/bin/env python3
"""Save one side-by-side PNG: default scene camera (left) + wrist camera (right).

No policy, no GPU. One env reset and two MuJoCo renders.

Headless usage
--------------
- If you have no DISPLAY, set e.g. ``MUJOCO_GL=osmesa`` or ``MUJOCO_GL=egl`` before
  running, **or** use a virtual framebuffer::

    xvfb-run -a -s "-screen 0 1024x768x24" \\
      python cube_dataset/pi05_joint_space/snapshot_base_wrist_dual.py

- On clusters where glfw works with xvfb, ``MUJOCO_GL=glfw`` with xvfb-run also works.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assert_display_or_mujoco_gl() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        here = Path(__file__).resolve().relative_to(Path.cwd()) if Path.cwd() in Path(__file__).resolve().parents else Path(__file__)
        print(
            "FATAL: no DISPLAY and no MUJOCO_GL. Examples:\n"
            "  export MUJOCO_GL=osmesa    # or egl\n"
            f'  xvfb-run -a -s "-screen 0 1024x768x24" python {here}\n',
            file=sys.stderr,
        )
        sys.exit(3)


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


def _to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    u8 = np.asarray(arr, dtype=np.uint8)
    if u8.ndim != 3 or u8.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC image, got shape {u8.shape}")
    if u8.shape[-1] == 4:
        u8 = u8[..., :3]
    return np.ascontiguousarray(u8)


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


def _save_png(path: Path, rgb_u8: np.ndarray) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, rgb_u8)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(rgb_u8)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore[assignment]
    w2 = rgb_u8.shape[1] // 2
    for label, x0 in (("scene", 6), ("wrist", w2 + 6)):
        draw.text((x0, 4), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(path, format="PNG")


def main() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    default_out = Path(__file__).resolve().parent / "dual_pov_snapshot.png"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=default_out, help="Output PNG path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=None, help="Ignored if --meta is set")
    parser.add_argument("--joint-scale", type=float, default=0.05)
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="vlm_start_goal meta.json (uses reset_seed / reset_options / task_id)",
    )
    args = parser.parse_args()

    assert_display_or_mujoco_gl()

    import gymnasium as gym
    import ogbench  # noqa: F401

    from cube_dataset.pi05_joint_space import register_joint_target_envs

    register_joint_target_envs()

    if args.meta is not None:
        meta = load_meta(args.meta)
        seed = int(meta.get("reset_seed", args.seed))
        opts = reset_options_from_meta(meta)
    else:
        seed = args.seed
        opts: dict[str, Any] = {"render_goal": True}
        if args.task_id is not None:
            opts["task_id"] = int(args.task_id)

    env = gym.make(
        "cube-single-joint-target-v0",
        ob_type="states",
        render_mode="rgb_array",
        joint_scale=float(args.joint_scale),
    )
    try:
        try:
            env.reset(seed=seed, options=opts)
        except Exception as exc:
            print(
                f"FATAL: env.reset failed ({exc}).\n"
                "Check MUJOCO_GL / DISPLAY; try xvfb-run or MUJOCO_GL=osmesa|egl.\n",
                file=sys.stderr,
            )
            raise SystemExit(4) from exc

        raw = env.unwrapped
        try:
            base = _to_rgb_u8(env.render())
            if not hasattr(raw, "render_wrist"):
                print("FATAL: env has no render_wrist (wrong env class?)", file=sys.stderr)
                raise SystemExit(6)
            wrist = _to_rgb_u8(raw.render_wrist())
        except Exception as exc:
            print(
                f"FATAL: render failed ({exc}).\n"
                "If this is a GL/headless issue, set MUJOCO_GL or use xvfb-run (see script docstring).\n",
                file=sys.stderr,
            )
            raise SystemExit(5) from exc

        dual = _hstack_same_height(base, wrist)
        _save_png(args.out, dual)
        print(f"Wrote {args.out.resolve()}  shape={dual.shape}  seed={seed}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
