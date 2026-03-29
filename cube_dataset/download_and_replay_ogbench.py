#!/usr/bin/env python3
"""Download OGBench cube play datasets, pick top-33 episodes by return, save HDF5, replay to MP4/NPZ.

Run with stable-worldmodel's venv so `ogbench` and MuJoCo resolve correctly, e.g.:
  cd /path/to/project/stable-worldmodel && .venv/bin/python ../cube_dataset/download_and_replay_ogbench.py

Preflight (imports, env reset+render, tiny MP4):
  .../python ../cube_dataset/download_and_replay_ogbench.py --preflight

Headless / cluster: set MUJOCO_GL before running (often `export MUJOCO_GL=egl` on GPU nodes).
If render fails on a login node, preflight may still PASS with “render skipped”.
"""
from __future__ import annotations

import argparse
import os
import shutil
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# Optional SSL: some clusters MITM HTTPS; ogbench uses plain urllib.
# -----------------------------------------------------------------------------


_urllib_ssl_patch_done = False


def ensure_urllib_ssl_patch() -> None:
    """Patch urllib for ogbench downloads inside joblib worker processes."""
    global _urllib_ssl_patch_done
    if _urllib_ssl_patch_done:
        return
    _orig = urllib.request.urlopen

    def _patched(url, *args, **kwargs):
        if args or kwargs:
            return _orig(url, *args, **kwargs)
        return _urlopen(url)

    urllib.request.urlopen = _patched
    _urllib_ssl_patch_done = True


def _urlopen(url: str):
    """HTTPS fetch; on MITM/self-signed chains urllib often raises URLError, not SSLCertVerificationError."""
    ctx = ssl.create_default_context()

    def _open(c):
        return urllib.request.urlopen(url, context=c)

    try:
        return _open(ctx)
    except (ssl.SSLCertVerificationError, urllib.error.URLError) as e:
        if os.environ.get("OGBENCH_INSECURE_SSL", "1") == "0":
            raise
        reason = getattr(e, "reason", e)
        msg = str(e)
        if not isinstance(e, ssl.SSLCertVerificationError) and "CERTIFICATE_VERIFY_FAILED" not in msg:
            if not isinstance(reason, ssl.SSLCertVerificationError):
                raise
        import warnings

        warnings.warn(
            "SSL verification failed; retrying with unverified context. "
            "Prefer fixing SSL_CERT_FILE; set OGBENCH_INSECURE_SSL=0 to disable insecure retry.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _open(ssl._create_unverified_context())


def _npz_readable(path: Path) -> bool:
    try:
        with np.load(path) as z:
            _ = z.files
        return True
    except Exception:
        return False


def download_file(url: str, dest: Path, desc: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.suffix == ".npz" and _npz_readable(dest):
        print(f"  [skip] ok {dest.name} ({human_bytes(dest.stat().st_size)})")
        return
    if dest.exists():
        dest.unlink()
        print(f"  [redo] replaced corrupt/incomplete {dest.name}")
    print(f"  [get] {url}")
    tmp = dest.parent / f".{dest.name}.part.{os.getpid()}"
    ok = False
    try:
        resp = _urlopen(url)
        total = getattr(resp, "length", None)
        try:
            from tqdm import tqdm

            with open(tmp, "wb") as f, tqdm(
                desc=desc or dest.name,
                total=int(total) if total else None,
                unit="B",
                unit_scale=True,
            ) as pbar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
                f.flush()
                os.fsync(f.fileno())
        except ImportError:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
                f.flush()
                os.fsync(f.fileno())
        tmp.replace(dest)
        ok = True
    finally:
        if not ok and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# -----------------------------------------------------------------------------
# Dataset loading (ogbench + optional rewards)
# -----------------------------------------------------------------------------


def load_dataset_with_rewards(
    dataset_path: str | Path,
    *,
    ob_dtype=np.float32,
    action_dtype=np.float32,
    compact_dataset: bool = False,
    add_info: bool = True,
) -> dict[str, Any]:
    """Mirror ogbench.utils.load_dataset but keep `rewards` when present in the npz."""
    from ogbench.utils import load_dataset

    path = str(dataset_path)
    file = np.load(path)
    raw_rewards = file["rewards"].astype(np.float32) if "rewards" in file.files else None
    file.close()

    ds = load_dataset(
        path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=add_info,
    )
    if raw_rewards is not None:
        z = np.load(path)
        t = z["terminals"].astype(np.float32)
        ob_mask = (1.0 - t).astype(bool)
        ds["rewards"] = raw_rewards[ob_mask].astype(np.float32)
        z.close()
    return ds


def episode_ranges_from_dataset(ds: dict[str, Any]) -> list[tuple[int, int]]:
    """Non-compact OGBench layout: one terminal=1 per episode end; lengths match actions."""
    t = ds["terminals"] > 0.5
    ends = np.nonzero(t)[0]
    if len(ends) == 0:
        return []
    starts = np.concatenate([[0], ends[:-1] + 1])
    return [(int(s), int(e) + 1) for s, e in zip(starts, ends)]


def merge_shard_npz(shard_paths: list[Path], out_path: Path, *, force: bool = False) -> None:
    if not force and out_path.exists() and _npz_readable(out_path):
        print(f"  [skip merge] ok {out_path.name} ({human_bytes(out_path.stat().st_size)})")
        return
    keys_ref: list[str] | None = None
    chunks: dict[str, list[np.ndarray]] = {}
    for p in shard_paths:
        with np.load(p) as z:
            keys = sorted(z.files)
            if keys_ref is None:
                keys_ref = keys
            elif keys != keys_ref:
                raise ValueError(f"Key mismatch in {p}: {keys} vs {keys_ref}")
            for k in keys_ref:
                chunks.setdefault(k, []).append(np.asarray(z[k]))
    merged = {k: np.concatenate(chunks[k], axis=0) for k in keys_ref or []}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)
    print(f"  [merge] wrote {out_path} ({human_bytes(out_path.stat().st_size)})")


# -----------------------------------------------------------------------------
# Top-33 + HDF5 + replay
# -----------------------------------------------------------------------------


def select_top_episodes(
    ds: dict[str, Any],
    *,
    top_k: int = 33,
    reward_frac: float = 0.8,
) -> tuple[list[tuple[int, int]], np.ndarray, str]:
    ranges = episode_ranges_from_dataset(ds)
    if not ranges:
        return [], np.array([], dtype=np.float32), "no_episodes"

    rewards_present = "rewards" in ds
    totals: list[float] = []
    for s, e in ranges:
        if rewards_present:
            totals.append(float(np.sum(ds["rewards"][s:e])))
        else:
            # Sparse success signal: count terminal successes (usually 1 at end).
            totals.append(float(np.sum(ds["terminals"][s:e])))

    totals_arr = np.asarray(totals, dtype=np.float64)
    max_r = float(np.max(totals_arr)) if len(totals_arr) else 0.0
    mode = "rewards" if rewards_present else "terminals_sum_fallback"

    if max_r <= 0:
        qualified = list(range(len(ranges)))
    else:
        qualified = [i for i, tr in enumerate(totals_arr) if tr > reward_frac * max_r]

    if not qualified:
        qualified = list(range(len(ranges)))

    order = sorted(qualified, key=lambda i: totals_arr[i], reverse=True)
    picked = order[: min(top_k, len(order))]
    picked_ranges = [ranges[i] for i in picked]
    picked_totals = totals_arr[np.array(picked, dtype=int)]
    return picked_ranges, picked_totals.astype(np.float32), mode


def save_top_h5(
    path: Path,
    ds: dict[str, Any],
    ranges: list[tuple[int, int]],
    totals: np.ndarray,
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    obs_chunks = []
    act_chunks = []
    rew_chunks = []
    term_chunks = []
    goal_rows = []

    next_obs = ds.get("next_observations")

    for (s, e), tr in zip(ranges, totals):
        obs_chunks.append(ds["observations"][s:e].copy())
        act_chunks.append(ds["actions"][s:e].copy())
        if "rewards" in ds:
            rew_chunks.append(ds["rewards"][s:e].copy())
        else:
            rew_chunks.append(np.zeros(e - s, dtype=np.float32))
        term_chunks.append(ds["terminals"][s:e].copy())
        if next_obs is not None:
            g = next_obs[e - 1].astype(np.float32).reshape(1, -1)
        else:
            g = ds["observations"][e - 1].astype(np.float32).reshape(1, -1)
        goal_rows.append(g)

    states = np.concatenate(obs_chunks, axis=0)
    actions = np.concatenate(act_chunks, axis=0)
    rewards = np.concatenate(rew_chunks, axis=0)
    terminals = np.concatenate(term_chunks, axis=0)
    lengths = np.array([e - s for s, e in ranges], dtype=np.int64)
    boundaries = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    goals = np.concatenate(goal_rows, axis=0)

    with h5py.File(path, "w") as f:
        f.create_dataset("states", data=states, compression="gzip", compression_opts=4)
        f.create_dataset("actions", data=actions, compression="gzip", compression_opts=4)
        f.create_dataset("rewards", data=rewards, compression="gzip", compression_opts=4)
        f.create_dataset("terminals", data=terminals, compression="gzip", compression_opts=4)
        f.create_dataset("episode_boundaries", data=boundaries)
        f.create_dataset("total_rewards", data=totals.astype(np.float32))
        f.create_dataset("goals", data=goals, compression="gzip", compression_opts=4)
        if "qpos" in ds:
            qp = np.concatenate([ds["qpos"][s:e] for s, e in ranges], axis=0)
            qv = np.concatenate([ds["qvel"][s:e] for s, e in ranges], axis=0)
            f.create_dataset("qpos", data=qp, compression="gzip", compression_opts=4)
            f.create_dataset("qvel", data=qv, compression="gzip", compression_opts=4)

    print(f"  [h5] {path} ({human_bytes(path.stat().st_size)})")


def replay_episodes(
    env,
    ds: dict[str, Any],
    ranges: list[tuple[int, int]],
    out_video_dir: Path,
    *,
    fps: int = 10,
    skip_existing: bool = True,
) -> None:
    import imageio.v2 as imageio

    out_video_dir.mkdir(parents=True, exist_ok=True)
    has_q = "qpos" in ds and "qvel" in ds
    raw = env.unwrapped

    for i, (s, e) in enumerate(ranges):
        vid = out_video_dir / f"ep_{i}.mp4"
        npz_path = out_video_dir / f"ep_{i}_frames.npz"
        if (
            skip_existing
            and vid.exists()
            and npz_path.exists()
            and vid.stat().st_size > 4096
            and npz_path.stat().st_size > 1000
        ):
            print(f"    [skip] {vid.name} (+ npz) already present")
            continue

        actions_ep = ds["actions"][s:e]
        frames: list[np.ndarray] = []

        env.reset(seed=0)
        if has_q:
            q0 = np.asarray(ds["qpos"][s], dtype=np.float64)
            v0 = np.asarray(ds["qvel"][s], dtype=np.float64)
            raw.set_state(q0, v0)
        else:
            print(f"    [warn] ep {i}: no qpos/qvel; reset-only replay (may drift)")

        frames.append(env.render())

        for a in actions_ep:
            env.step(np.asarray(a, dtype=np.float32))
            frames.append(env.render())

        stack = np.stack(frames, axis=0).astype(np.uint8)
        try:
            imageio.mimwrite(vid, stack, fps=fps, codec="libx264")
        except Exception as ex:
            print(f"    [warn] libx264 failed ({ex}); trying ffmpeg plugin")
            imageio.mimwrite(vid, stack, fps=fps, plugin="ffmpeg")
        np.savez_compressed(npz_path, frames=stack)
        print(f"    [vid] {vid.name} frames={stack.shape[0]} {stack.shape[1:]} -> {human_bytes(vid.stat().st_size)}")


# -----------------------------------------------------------------------------
# Per-family pipeline
# -----------------------------------------------------------------------------

STANDARD_DATASETS = ["cube-single-play-v0", "cube-double-play-v0"]
TRIPLE_TRAIN_SHARDS = [f"cube-triple-play-v0-{i:03d}.npz" for i in range(5)]
TRIPLE_VAL_SHARD = "cube-triple-play-v0-000-val.npz"
TRIPLE_SUBDIR = "cube-triple-play-100m-v0"
DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"


def process_standard(
    dataset_name: str,
    family: str,
    root: Path,
    *,
    skip_existing_videos: bool = True,
) -> None:
    ensure_urllib_ssl_patch()
    from ogbench.utils import make_env_and_datasets

    ddir = root / "datasets"
    ddir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {dataset_name} ({family}) ===")
    train_path = ddir / f"{dataset_name}.npz"
    val_path = ddir / f"{dataset_name}-val.npz"
    download_file(f"{DATASET_URL}/{train_path.name}", train_path, desc=train_path.name)
    download_file(f"{DATASET_URL}/{val_path.name}", val_path, desc=val_path.name)

    env, _, val_ds = make_env_and_datasets(
        dataset_name,
        dataset_dir=str(ddir),
        dataset_path=str(train_path),
        compact_dataset=False,
        add_info=True,
        ob_type="states",
        render_mode="rgb_array",
    )
    train_ds = load_dataset_with_rewards(
        train_path,
        ob_dtype=np.float32,
        action_dtype=np.float32,
        compact_dataset=False,
        add_info=True,
    )

    n_t = len(train_ds["observations"])
    print(f"  train transitions: {n_t:,}  val: {len(val_ds['observations']):,}")

    ranges, totals, mode = select_top_episodes(train_ds)
    print(f"  selection mode: {mode}  episodes>=filter: {len(ranges)}  top totals[0:3]={totals[:3]}")

    h5_path = root / "top_episodes" / f"{family}-top33.h5"
    save_top_h5(h5_path, train_ds, ranges, totals)

    vdir = root / "videos_frames" / family
    replay_episodes(env, train_ds, ranges, vdir, fps=10, skip_existing=skip_existing_videos)
    env.close()
    print(f"  done {family}")


def process_triple_100m_partial(
    root: Path,
    *,
    skip_existing_videos: bool = True,
) -> None:
    ensure_urllib_ssl_patch()
    import gymnasium as gym

    family = "cube-triple-100m"
    ddir = root / "datasets"
    ddir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== cube-triple-play-100m-v0 (partial 5 shards) ({family}) ===")

    base_url = f"{DATASET_URL}/{TRIPLE_SUBDIR}"
    shard_paths: list[Path] = []
    for name in TRIPLE_TRAIN_SHARDS:
        url = f"{base_url}/{name}"
        dest = ddir / name
        download_file(url, dest, desc=name)
        if not dest.exists():
            raise FileNotFoundError(f"Missing shard after download: {dest}")
        shard_paths.append(dest)

    val_name = TRIPLE_VAL_SHARD
    val_url = f"{base_url}/{val_name}"
    val_path = ddir / val_name
    download_file(val_url, val_path, desc=val_name)

    merged = ddir / "merged_cube_triple_play_v0_shards_0-4.npz"
    merge_shard_npz(shard_paths, merged)

    env = gym.make("cube-triple-v0", ob_type="states", render_mode="rgb_array")
    train_ds = load_dataset_with_rewards(
        merged,
        ob_dtype=np.float32,
        action_dtype=np.float32,
        compact_dataset=False,
        add_info=True,
    )
    _ = load_dataset_with_rewards(
        val_path,
        ob_dtype=np.float32,
        action_dtype=np.float32,
        compact_dataset=False,
        add_info=True,
    )

    n_t = len(train_ds["observations"])
    print(f"  train transitions (merged): {n_t:,}")

    ranges, totals, mode = select_top_episodes(train_ds)
    print(f"  selection mode: {mode}  picked: {len(ranges)}  top totals[0:3]={totals[:3]}")

    h5_path = root / "top_episodes" / f"{family}-top33.h5"
    save_top_h5(h5_path, train_ds, ranges, totals)

    vdir = root / "videos_frames" / family
    replay_episodes(env, train_ds, ranges, vdir, fps=10, skip_existing=skip_existing_videos)
    env.close()
    print(f"  done {family}")


def run_preflight() -> None:
    print("=== preflight ===")
    import gymnasium as gym
    import h5py
    import imageio.v2 as imageio
    import joblib
    import mujoco
    import ogbench  # noqa: F401

    for pkg in ("numpy", "h5py", "gymnasium", "mujoco", "joblib", "tqdm", "imageio"):
        __import__(pkg)
    print("  imports: ok")

    render_ok = True
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        render_ok = False
        print("  render: SKIP (no DISPLAY and MUJOCO_GL unset — avoids MuJoCo abort on some login nodes)")
    else:
        for eid in ("cube-single-v0", "cube-double-v0", "cube-triple-v0"):
            try:
                env = gym.make(eid, ob_type="states", render_mode="rgb_array")
                env.reset(seed=0)
                frame = env.render()
                assert frame is not None and frame.ndim == 3 and frame.shape[-1] in (3, 4)
                env.close()
                print(f"  env {eid}: reset+render {frame.shape}")
            except Exception as ex:
                render_ok = False
                print(f"  env {eid}: render SKIP ({type(ex).__name__}: {ex})")
    if not render_ok:
        print(
            "  hint: on PBS compute, export MUJOCO_GL=egl (GPU) or use OSMesa where installed; "
            "then re-run --preflight or the full script."
        )

    td = Path(tempfile.mkdtemp(prefix="ogbench_pf_"))
    try:
        vid = td / "oneframe.mp4"
        imageio.mimwrite(vid, [np.zeros((64, 64, 3), dtype=np.uint8)], fps=10, codec="libx264")
        assert vid.exists() and vid.stat().st_size > 0
        print(f"  imageio mp4: ok ({vid.stat().st_size} bytes)")
    except Exception as ex:
        print(f"  imageio mp4: FAILED ({ex})")
        raise
    finally:
        shutil.rmtree(td, ignore_errors=True)

    print("preflight: PASS" + ("" if render_ok else " (render skipped — see hint)"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="cube_dataset root (default: directory containing this script)",
    )
    parser.add_argument("--preflight", action="store_true", help="Run dependency smoke tests and exit")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process datasets sequentially instead of joblib parallel",
    )
    parser.add_argument(
        "--families",
        type=str,
        default="all",
        help="Comma-separated subset: cube-single,cube-double,cube-triple-100m (default: all)",
    )
    parser.add_argument(
        "--no-skip-videos",
        action="store_true",
        help="Re-encode all episodes even if mp4/npz already exist",
    )
    args = parser.parse_args()

    root = args.root or Path(__file__).resolve().parent
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "top_episodes").mkdir(parents=True, exist_ok=True)
    (root / "videos_frames").mkdir(parents=True, exist_ok=True)

    if args.preflight:
        run_preflight()
        return

    ensure_urllib_ssl_patch()

    # Xvfb sets DISPLAY; GPU nodes often set MUJOCO_GL=egl.
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        print(
            "FATAL: headless replay needs DISPLAY (e.g. xvfb-run) or MUJOCO_GL=egl on a GPU node. "
            "Or use --preflight on a login node to check imports only.",
            file=sys.stderr,
        )
        sys.exit(3)

    try:
        import ogbench  # noqa: F401
        import gymnasium as gym  # noqa: F401
        import h5py  # noqa: F401
        import imageio  # noqa: F401
        import joblib
        import mujoco  # noqa: F401
    except ImportError as e:
        print("FATAL: missing dependency:", e, file=sys.stderr)
        print("Use stable-worldmodel .venv and/or: pip install ogbench gymnasium mujoco h5py imageio joblib tqdm", file=sys.stderr)
        sys.exit(2)

    want: set[str] | None = None
    if args.families.strip().lower() != "all":
        want = {x.strip() for x in args.families.split(",") if x.strip()}
        valid = {"cube-single", "cube-double", "cube-triple-100m"}
        bad = want - valid
        if bad:
            print(f"FATAL: unknown --families entries {bad}; allowed {valid}", file=sys.stderr)
            sys.exit(2)

    jobs = []
    for name in STANDARD_DATASETS:
        fam = name.replace("-play-v0", "")
        if want is not None and fam not in want:
            continue
        jobs.append(("standard", name, fam))

    run_triple = want is None or "cube-triple-100m" in want

    sk_vid = not args.no_skip_videos

    def _run_standard(name: str, fam: str) -> None:
        process_standard(name, fam, root, skip_existing_videos=sk_vid)

    def _run_triple() -> None:
        process_triple_100m_partial(root, skip_existing_videos=sk_vid)

    print(f"Root: {root.resolve()}")
    if want is not None:
        print(f"Families filter: {sorted(want)}")
    print(f"Datasets dir size (before): {human_bytes(dir_size(root / 'datasets'))}")

    if args.sequential:
        for kind, name, fam in jobs:
            if kind == "standard":
                _run_standard(name, fam)
        if run_triple:
            _run_triple()
    else:
        from joblib import Parallel, delayed

        if jobs:
            Parallel(n_jobs=max(1, len(jobs)), backend="loky")(
                delayed(_run_standard)(name, fam) for kind, name, fam in jobs
            )
        if run_triple:
            _run_triple()

    print(f"\nDatasets dir size (after): {human_bytes(dir_size(root / 'datasets'))}")
    print(f"All of cube_dataset: {human_bytes(dir_size(root))}")
    print("Done.")


if __name__ == "__main__":
    main()
