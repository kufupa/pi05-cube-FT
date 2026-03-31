#!/usr/bin/env python3
"""Add VLA triptych outputs next to existing vlm_start_goal samples (NEW files only).

Never writes to: start.png, goal.png, side_by_side.png, meta.json.

Per sample_dir with meta.json, writes:
  vla_start_clean.png       — reset(render_goal=False), env.render()
  vla_start_with_markers.png — reset(render_goal=True), env.render()
  vla_goal_only.png         — info[\"goal_rendered\"] from second reset
  vla_triptych.png          — horizontal concat of the three above
  vlm_triptych.json         — sidecar metadata

Appends one line per written sample to manifest_triptych.jsonl (truncated only with --force).

Run:
  cd project && xvfb-run -a -s \"-screen 0 1024x768x24\" \\
    stable-worldmodel/.venv/bin/python cube_dataset/extend_vlm_triptych.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

TRIPTYCH_SCHEMA = 1
FILES = {
    "vla_start_clean": "vla_start_clean.png",
    "vla_start_with_markers": "vla_start_with_markers.png",
    "vla_goal_only": "vla_goal_only.png",
    "vla_triptych": "vla_triptych.png",
    "sidecar": "vlm_triptych.json",
}


def default_root() -> Path:
    return Path(__file__).resolve().parent / "goal_images" / "vlm_start_goal"


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
            "FATAL: no DISPLAY and no MUJOCO_GL. Use xvfb-run or MUJOCO_GL=egl.",
            file=sys.stderr,
        )
        sys.exit(3)


def crop_to_height(*frames: np.ndarray) -> list[np.ndarray]:
    h = min(f.shape[0] for f in frames)
    return [f[:h, : f.shape[1]] for f in frames]


def triptych_concat(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ca, cb, cc = crop_to_height(a, b, c)
    return np.concatenate([ca, cb, cc], axis=1)


def replay_python_triptych(env_id: str, reset_seed: int, task_id: int) -> str:
    return (
        "import ogbench  # noqa: F401\n"
        "import gymnasium as gym\n"
        f'env = gym.make("{env_id}", ob_type="states", render_mode="rgb_array")\n'
        f"env.reset(seed={reset_seed}, options={{\"task_id\": {task_id}, \"render_goal\": False}})\n"
        "clean = env.render()\n"
        f"obs, info = env.reset(seed={reset_seed}, options={{\"task_id\": {task_id}, \"render_goal\": True}})\n"
        "with_markers = env.render()\n"
        "goal = info[\"goal_rendered\"]\n"
    )


def vlm_goal_root(sample_dir: Path) -> Path:
    """vlm_start_goal/ containing env_id subdirs."""
    return sample_dir.parent.parent


def write_triptych_sample(sample_dir: Path, env: Any, manifest_triptych: Path) -> str:
    import imageio.v2 as imageio

    meta_path = sample_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    env_id = meta.get("env_id")
    reset_seed = int(meta["reset_seed"])
    task_id = meta.get("task_id")
    if task_id is None:
        raise ValueError("missing task_id")
    task_id = int(task_id)

    _, _ = env.reset(seed=reset_seed, options={"task_id": task_id, "render_goal": False})
    clean = rgb_u8(env.render())

    _, info = env.reset(seed=reset_seed, options={"task_id": task_id, "render_goal": True})
    with_m = rgb_u8(env.render())
    goal_raw = info.get("goal_rendered")
    if goal_raw is None:
        raise RuntimeError("goal_rendered None")
    goal_only = rgb_u8(goal_raw)

    trip = triptych_concat(clean, with_m, goal_only)

    imageio.imwrite(sample_dir / FILES["vla_start_clean"], clean)
    imageio.imwrite(sample_dir / FILES["vla_start_with_markers"], with_m)
    imageio.imwrite(sample_dir / FILES["vla_goal_only"], goal_only)
    imageio.imwrite(sample_dir / FILES["vla_triptych"], trip)

    root = vlm_goal_root(sample_dir)
    rel_sample = str(sample_dir.relative_to(root))
    sidecar = {
        "schema_version": TRIPTYCH_SCHEMA,
        "env_id": env_id,
        "reset_seed": reset_seed,
        "task_id": task_id,
        "files": FILES.copy(),
        "notes": (
            "vla_start_clean: reset(render_goal=False). "
            "vla_start_with_markers / vla_goal_only: second reset(render_goal=True)."
        ),
        "replay_python": replay_python_triptych(str(env_id), reset_seed, task_id),
        "sample_rel_path": rel_sample,
    }
    (sample_dir / FILES["sidecar"]).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    with open(manifest_triptych, "a", encoding="utf-8") as mf:
        mf.write(
            json.dumps({"rel_path": rel_sample, "env_id": env_id, "task_id": task_id}) + "\n"
        )

    return f"[ok] {rel_sample}"


def discover_samples(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("**/sample_*/meta.json"))


def smoke_test(root: Path | None) -> None:
    print("=== extend_vlm_triptych smoke ===")
    import gymnasium as gym

    import ogbench  # noqa: F401

    try:
        import imageio.v2 as imageio  # noqa: F401
        import mujoco  # noqa: F401
    except ImportError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)

    assert_display_or_mujoco_gl()

    r = root or default_root()
    samples = discover_samples(r)
    if not samples:
        print("FATAL: no sample_*/meta.json under", r, file=sys.stderr)
        sys.exit(4)

    td_base = Path(tempfile.mkdtemp(prefix="triptych_smoke_"))
    # Mirror real layout so vlm_goal_root(sample_dir) resolves to .../vlm_start_goal
    vlm = td_base / "vlm_start_goal" / "cube-single-v0" / "sample_0000"
    vlm.mkdir(parents=True)
    shutil.copy2(samples[0] / "meta.json", vlm / "meta.json")
    meta0 = json.loads((vlm / "meta.json").read_text(encoding="utf-8"))
    try:
        env = gym.make(
            meta0["env_id"],
            **(meta0.get("env_kwargs") or {"ob_type": "states", "render_mode": "rgb_array"}),
        )
        try:
            mf = td_base / "vlm_start_goal" / "_smoke_manifest.jsonl"
            msg = write_triptych_sample(vlm, env, mf)
            print(f"  {msg}")
        finally:
            env.close()
        for name in (
            FILES["vla_start_clean"],
            FILES["vla_start_with_markers"],
            FILES["vla_goal_only"],
            FILES["vla_triptych"],
            FILES["sidecar"],
        ):
            p = vlm / name
            if not p.is_file() or p.stat().st_size < 50:
                print(f"FATAL: bad output {p}", file=sys.stderr)
                sys.exit(5)
    finally:
        shutil.rmtree(td_base, ignore_errors=True)

    print("smoke test: PASS")


def run_batch(root: Path, *, skip_existing: bool, force: bool) -> None:
    import gymnasium as gym

    import ogbench  # noqa: F401

    assert_display_or_mujoco_gl()

    samples = discover_samples(root)
    if not samples:
        print("No samples found under", root)
        return

    manifest_triptych = root / "manifest_triptych.jsonl"
    if force:
        manifest_triptych.write_text("", encoding="utf-8")

    env_cache: dict[str, Any] = {}
    ok = 0
    skipped = 0
    errs = 0
    try:
        for sample_dir in samples:
            marker = sample_dir / FILES["vla_triptych"]
            if skip_existing and marker.exists() and not force:
                skipped += 1
                print(f"  [skip] {sample_dir.name}")
                continue

            meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
            env_id = meta["env_id"]
            env_kwargs = meta.get("env_kwargs") or {"ob_type": "states", "render_mode": "rgb_array"}

            if env_id not in env_cache:
                env_cache[env_id] = gym.make(env_id, **env_kwargs)
            env = env_cache[env_id]

            try:
                msg = write_triptych_sample(sample_dir, env, manifest_triptych)
                print(f"  {msg}")
                ok += 1
            except Exception as ex:
                print(f"  [err] {sample_dir}: {ex}", file=sys.stderr)
                errs += 1
    finally:
        for e in env_cache.values():
            e.close()

    print(f"Done. ok={ok} skipped={skipped} errors={errs}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="vlm_start_goal root")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=True,
        help="Re-process even when vla_triptych.png exists",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite new triptych outputs and truncate manifest_triptych.jsonl",
    )
    args = parser.parse_args()

    root = args.root or default_root()
    skip = args.skip_existing

    if args.smoke_test:
        smoke_test(root)
        return

    run_batch(root, skip_existing=skip, force=args.force)


if __name__ == "__main__":
    main()
