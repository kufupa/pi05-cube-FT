#!/usr/bin/env python3
"""Sample OGBench cube envs: save start RGB, gold goal RGB (info['goal_rendered']), side-by-side PNGs + meta.json.

Run with stable-worldmodel venv (ogbench, mujoco, gymnasium, imageio), e.g.:
  cd /path/to/project && xvfb-run -a -s "-screen 0 1024x768x24" \\
    stable-worldmodel/.venv/bin/python cube_dataset/sample_start_goal_images.py

GPU / PBS: export MUJOCO_GL=egl (and request a GPU) instead of xvfb.

Smoke test (must pass before batch jobs):
  .../python cube_dataset/sample_start_goal_images.py --smoke-test
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# Defaults (override via argparse)
# -----------------------------------------------------------------------------
N_SAMPLES = 100
BASE_SEED = 0
ENV_IDS = ("cube-single-v0",)
OUT_ROOT: Path | None = None  # default: cube_dataset/goal_images/vlm_start_goal
SKIP_EXISTING = True
SCHEMA_VERSION = 1


def _cube_dataset_root() -> Path:
    return Path(__file__).resolve().parent


def default_out_root() -> Path:
    return _cube_dataset_root() / "goal_images" / "vlm_start_goal"


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return str(obj)


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("ogbench", "gymnasium", "mujoco", "numpy", "imageio"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


def rgb_u8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.shape[-1] == 4:
        x = x[..., :3]
    return x


def assert_display_or_mujoco_gl() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        print(
            "FATAL: no DISPLAY and no MUJOCO_GL. Use e.g.\n"
            "  xvfb-run -a -s \"-screen 0 1024x768x24\" .../python cube_dataset/sample_start_goal_images.py\n"
            "or on a GPU node: export MUJOCO_GL=egl",
            file=sys.stderr,
        )
        sys.exit(3)


def smoke_test() -> None:
    """Imports, GL, one reset(render_goal), PNG write."""
    print("=== smoke test ===")
    try:
        import gymnasium as gym  # noqa: F401
        import imageio.v2 as imageio  # noqa: F401
        import mujoco  # noqa: F401
        import ogbench  # noqa: F401
    except ImportError as e:
        print(f"FATAL: import failed: {e}", file=sys.stderr)
        sys.exit(2)

    assert_display_or_mujoco_gl()

    import gymnasium as gym
    import imageio.v2 as imageio

    env = gym.make("cube-single-v0", ob_type="states", render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=0, options={"render_goal": True})
        gr = info.get("goal_rendered")
        if gr is None:
            print("FATAL: goal_rendered is None after reset(render_goal=True)", file=sys.stderr)
            sys.exit(4)
        start = env.render()
        start_u8 = rgb_u8(start)
        goal_u8 = rgb_u8(gr)
        if start_u8.shape[:2] != goal_u8.shape[:2]:
            print(
                f"WARN: shape mismatch start {start_u8.shape} vs goal {goal_u8.shape} (continuing smoke)",
                file=sys.stderr,
            )
        td = Path(tempfile.mkdtemp(prefix="vlm_smoke_"))
        p = td / "probe.png"
        imageio.imwrite(p, start_u8)
        if not p.exists() or p.stat().st_size < 10:
            print("FATAL: PNG write probe failed", file=sys.stderr)
            sys.exit(5)
        print(f"  imports: ok  goal_rendered: {goal_u8.shape}  probe: {p.stat().st_size} B")
    finally:
        env.close()
    print("smoke test: PASS")


def replay_python_snippet(env_id: str, reset_seed: int) -> str:
    return (
        "import ogbench  # noqa: F401 — registers OGBench envs with Gymnasium\n"
        "import gymnasium as gym\n"
        f'env = gym.make("{env_id}", ob_type="states", render_mode="rgb_array")\n'
        f"obs, info = env.reset(seed={reset_seed}, options={{\"render_goal\": True}})\n"
        "start = env.render()\n"
        "goal = info[\"goal_rendered\"]\n"
        "# start and goal are uint8 RGB arrays (H, W, 3)\n"
    )


def sample_one(
    env,
    *,
    env_id: str,
    env_kwargs: dict[str, Any],
    reset_seed: int,
    sample_index: int,
    out_dir: Path,
    skip_existing: bool,
) -> dict[str, Any] | None:
    import imageio.v2 as imageio

    sample_dir = out_dir / f"sample_{sample_index:04d}"
    start_p = sample_dir / "start.png"
    if skip_existing and start_p.exists() and (sample_dir / "goal.png").exists():
        print(f"  [skip] {sample_dir.name}")
        return None

    sample_dir.mkdir(parents=True, exist_ok=True)
    obs, info = env.reset(seed=reset_seed, options={"render_goal": True})
    raw = env.unwrapped
    task_id = getattr(raw, "cur_task_id", None)
    task_info = getattr(raw, "cur_task_info", None)
    if task_info is not None:
        task_info_json = to_jsonable(task_info)
    else:
        task_info_json = None

    goal_img = info.get("goal_rendered")
    if goal_img is None:
        raise RuntimeError(f"goal_rendered missing for seed={reset_seed}")

    start_u8 = rgb_u8(env.render())
    goal_u8 = rgb_u8(goal_img)

    h = min(start_u8.shape[0], goal_u8.shape[0])
    w0 = start_u8.shape[1]
    w1 = goal_u8.shape[1]
    # align heights by crop (top-left) if mismatch
    s_crop = start_u8[:h, :w0]
    g_crop = goal_u8[:h, :w1]
    side = np.concatenate([s_crop, g_crop], axis=1)

    imageio.imwrite(sample_dir / "start.png", s_crop)
    imageio.imwrite(sample_dir / "goal.png", g_crop)
    imageio.imwrite(sample_dir / "side_by_side.png", side)

    reset_options = {"render_goal": True}
    meta = {
        "schema_version": SCHEMA_VERSION,
        "env_id": env_id,
        "env_kwargs": env_kwargs,
        "reset_seed": int(reset_seed),
        "reset_options": reset_options,
        "task_id": int(task_id) if task_id is not None else None,
        "task_info": task_info_json,
        "sample_index": int(sample_index),
        "packages": package_versions(),
        "replay_python": replay_python_snippet(env_id, reset_seed),
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # out_dir's parent is vlm_start_goal root passed as fam_root parent... actually out_dir = fam_root = out_root/env_id
    vlm_root = out_dir.parent
    manifest_row = {
        "rel_path": str(sample_dir.relative_to(vlm_root)),
        **{k: meta[k] for k in ("env_id", "reset_seed", "sample_index", "task_id")},
    }
    return manifest_row


def run_batch(
    *,
    n_samples: int,
    base_seed: int,
    env_ids: tuple[str, ...],
    out_root: Path,
    skip_existing: bool,
) -> None:
    import gymnasium as gym

    import ogbench  # noqa: F401 — register cube-*-v0 with Gymnasium

    assert_display_or_mujoco_gl()

    manifest_path = out_root / "manifest.jsonl"
    if not skip_existing:
        manifest_path.write_text("", encoding="utf-8")

    env_kwargs = {"ob_type": "states", "render_mode": "rgb_array"}
    env_cache: dict[str, Any] = {}
    rows_written = 0
    try:
        for i in range(n_samples):
            env_id = env_ids[i % len(env_ids)]
            if env_id not in env_cache:
                env_cache[env_id] = gym.make(env_id, **env_kwargs)
            env = env_cache[env_id]
            fam_root = out_root / env_id.replace("/", "_")
            fam_root.mkdir(parents=True, exist_ok=True)
            row = sample_one(
                env,
                env_id=env_id,
                env_kwargs=env_kwargs,
                reset_seed=base_seed + i,
                sample_index=i,
                out_dir=fam_root,
                skip_existing=skip_existing,
            )
            if row is not None:
                with open(manifest_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(row) + "\n")
                rows_written += 1
                print(f"  [ok] {env_id} sample_{i:04d}")
    finally:
        for e in env_cache.values():
            e.close()

    print(f"Done. Wrote {rows_written} new samples under {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run dependency/render gate and exit")
    parser.add_argument("--root", type=Path, default=None, help="Output base (default: cube_dataset/goal_images/vlm_start_goal parent)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--env-ids",
        type=str,
        default=",".join(ENV_IDS),
        help="Comma-separated gym ids (round-robin)",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Overwrite existing sample folders")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    out = args.root if args.root is not None else default_out_root()
    out.mkdir(parents=True, exist_ok=True)
    env_ids = tuple(x.strip() for x in args.env_ids.split(",") if x.strip())
    if not env_ids:
        print("FATAL: empty --env-ids", file=sys.stderr)
        sys.exit(2)

    run_batch(
        n_samples=args.n_samples,
        base_seed=args.base_seed,
        env_ids=env_ids,
        out_root=out,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
