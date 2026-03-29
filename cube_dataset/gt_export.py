#!/usr/bin/env python3
"""OGBench play → ranked top-k, trim, HDF5, MP4s under gt_export/run_* only (never videos_frames/).

Run with stable-worldmodel .venv:
  project/stable-worldmodel/.venv/bin/python project/cube_dataset/gt_export.py
  ... --smoke   # pre-PBS checks only

Headless: set `MUJOCO_GL=egl` on GPU nodes, or use a virtual DISPLAY, e.g.
`xvfb-run -a -s "-screen 0 1024x768x24" ...`, or pass `--headless-xvfb` to re-exec under xvfb-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Project root on path for `cube_dataset.*` imports (works from any cwd).
_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# Import helpers only — do not run download_and_replay_ogbench.main()
from cube_dataset.download_and_replay_ogbench import (
    episode_ranges_from_dataset,
    human_bytes,
    load_dataset_with_rewards,
)

# -----------------------------------------------------------------------------
# Defaults (plan: not CLI-first)
# -----------------------------------------------------------------------------

TOP_K = 33
EPSILON = 0.04
K_SUSTAIN = 3
N_PRE = 80
N_POST = 60
L_MIN = 100
FPS = 10
REWARD_FRAC = 0.8

QPOS_OBJ_START = 14
QPOS_CUBE_BLOCK = 7


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cube_dataset_root() -> Path:
    return Path(__file__).resolve().parent


def _default_npz_path() -> Path:
    return _cube_dataset_root() / "datasets" / "cube-single-play-v0.npz"


def _family_from_dataset_name(dataset_stem: str) -> str:
    # cube-single-play-v0 -> cube-single
    return dataset_stem.replace("-play-v0", "")


def _gym_env_id_for_family(family: str) -> str:
    if family == "cube-single":
        return "cube-single-v0"
    if family == "cube-double":
        return "cube-double-v0"
    raise ValueError(f"Unsupported family for env id: {family!r} (use cube-single or cube-double)")


def _git_commit() -> str | None:
    try:
        root = _project_root()
        return (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return None


def _episode_distance_trace(
    ds: dict[str, Any],
    env,
    s: int,
    e: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-transition mean distance (m) from each cube xyz to its mocap target."""
    raw = env.unwrapped
    q0 = np.asarray(ds["qpos"][s], dtype=np.float64)
    v0 = np.asarray(ds["qvel"][s], dtype=np.float64)
    raw.set_state(q0, v0)
    n_cubes: int = int(raw._num_cubes)
    mocap = np.asarray(raw._data.mocap_pos.copy(), dtype=np.float64)
    targets = mocap.reshape(-1, 3)[:n_cubes]
    if targets.shape[0] < n_cubes:
        raise RuntimeError(f"mocap_pos rows {targets.shape[0]} < num_cubes {n_cubes}")

    qp = ds["qpos"][s:e]
    T = qp.shape[0]
    dists = np.zeros(T, dtype=np.float64)
    for t in range(T):
        dd = []
        for j in range(n_cubes):
            sl = slice(QPOS_OBJ_START + j * QPOS_CUBE_BLOCK, QPOS_OBJ_START + j * QPOS_CUBE_BLOCK + 3)
            cube_xyz = qp[t, sl]
            dd.append(float(np.linalg.norm(cube_xyz - targets[j])))
        dists[t] = float(np.mean(dd))
    target_flat = targets.reshape(-1)
    return dists, target_flat


def _score_all_episodes(
    ds: dict[str, Any],
    env,
    ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns per-episode (d_min, f_succ, d_mean, reward_sum_or_nan)."""
    n = len(ranges)
    d_mins = np.zeros(n, dtype=np.float64)
    f_succs = np.zeros(n, dtype=np.float64)
    d_means = np.zeros(n, dtype=np.float64)
    r_sums = np.full(n, np.nan, dtype=np.float64)
    has_rewards = "rewards" in ds

    for i, (s, e) in enumerate(ranges):
        dists, _ = _episode_distance_trace(ds, env, s, e)
        d_mins[i] = float(np.min(dists))
        d_means[i] = float(np.mean(dists))
        f_succs[i] = float(np.mean((dists <= EPSILON).astype(np.float64)))
        if has_rewards:
            r_sums[i] = float(np.sum(ds["rewards"][s:e]))

    return d_mins, f_succs, d_means, r_sums


def _rank_episode_indices(
    n_eps: int,
    rewards_present: bool,
    r_sums: np.ndarray,
    d_mins: np.ndarray,
    f_succs: np.ndarray,
    d_means: np.ndarray,
    *,
    reward_frac: float,
) -> list[int]:
    if rewards_present:
        max_r = float(np.max(r_sums)) if n_eps else 0.0
        if max_r <= 0:
            qualified = list(range(n_eps))
        else:
            qualified = [i for i in range(n_eps) if r_sums[i] > reward_frac * max_r]
        if not qualified:
            qualified = list(range(n_eps))
        order = sorted(qualified, key=lambda i: r_sums[i], reverse=True)
        return order

    # Distance tuple: ascending d_min, descending f_succ, ascending d_mean, ascending index
    keys = [(d_mins[i], -f_succs[i], d_means[i], i) for i in range(n_eps)]
    order_idx = sorted(range(n_eps), key=lambda j: keys[j])
    return order_idx


def _find_sustained_success(dists: np.ndarray) -> int | None:
    T = len(dists)
    for t in range(T - K_SUSTAIN + 1):
        if bool(np.all(dists[t : t + K_SUSTAIN] < EPSILON)):
            return t
    return None


def _build_trim_window(t_star: int, T: int) -> tuple[int, int] | None:
    t_anchor = t_star
    t_start = max(0, t_anchor - N_PRE)
    t_end = min(T - 1, t_anchor + N_POST)
    span = t_end - t_start + 1
    while span < L_MIN:
        expanded = False
        if t_start > 0:
            t_start -= 1
            expanded = True
        span = t_end - t_start + 1
        if span >= L_MIN:
            break
        if t_end < T - 1:
            t_end += 1
            expanded = True
        span = t_end - t_start + 1
        if span >= L_MIN:
            break
        if not expanded:
            break
    if t_end - t_start + 1 < L_MIN:
        return None
    return t_start, t_end


@dataclass
class TrimResult:
    eligible: bool
    t_start: int | None
    t_end: int | None
    t_star: int | None
    reason: str


def _trim_episode(dists: np.ndarray) -> TrimResult:
    T = len(dists)
    t_star = _find_sustained_success(dists)
    if t_star is None:
        return TrimResult(False, None, None, None, "no_sustained_success")
    w = _build_trim_window(t_star, T)
    if w is None:
        return TrimResult(False, None, None, t_star, "below_L_min")
    t_start, t_end = w
    return TrimResult(True, t_start, t_end, t_star, "")


def _replay_segment_to_mp4(
    env,
    ds: dict[str, Any],
    s: int,
    e: int,
    t_start: int,
    t_end: int,
    out_path: Path,
    *,
    fps: int = FPS,
) -> None:
    import imageio.v2 as imageio

    raw = env.unwrapped
    actions_ep = ds["actions"][s:e]
    assert 0 <= t_start <= t_end < len(actions_ep)

    env.reset(seed=0)
    q0 = np.asarray(ds["qpos"][s], dtype=np.float64)
    v0 = np.asarray(ds["qvel"][s], dtype=np.float64)
    raw.set_state(q0, v0)

    for j in range(t_start):
        env.step(np.asarray(actions_ep[j], dtype=np.float32))

    frames: list[np.ndarray] = [env.render()]
    for j in range(t_start, t_end + 1):
        env.step(np.asarray(actions_ep[j], dtype=np.float32))
        frames.append(env.render())

    stack = np.stack(frames, axis=0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimwrite(out_path, stack, fps=fps, codec="libx264")
    except Exception as ex:
        print(f"    [warn] libx264 failed ({ex}); ffmpeg plugin")
        imageio.mimwrite(out_path, stack, fps=fps, plugin="ffmpeg")


def _replay_full_episode_to_mp4(
    env,
    ds: dict[str, Any],
    s: int,
    e: int,
    out_path: Path,
    *,
    fps: int = FPS,
) -> None:
    import imageio.v2 as imageio

    raw = env.unwrapped
    actions_ep = ds["actions"][s:e]

    env.reset(seed=0)
    q0 = np.asarray(ds["qpos"][s], dtype=np.float64)
    v0 = np.asarray(ds["qvel"][s], dtype=np.float64)
    raw.set_state(q0, v0)

    frames: list[np.ndarray] = [env.render()]
    for a in actions_ep:
        env.step(np.asarray(a, dtype=np.float32))
        frames.append(env.render())

    stack = np.stack(frames, axis=0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimwrite(out_path, stack, fps=fps, codec="libx264")
    except Exception as ex:
        print(f"    [warn] libx264 failed ({ex}); ffmpeg plugin")
        imageio.mimwrite(out_path, stack, fps=fps, plugin="ffmpeg")


def save_gt_h5(
    path: Path,
    ds: dict[str, Any],
    ranges: list[tuple[int, int]],
    totals: np.ndarray,
    npz_episode_indices: np.ndarray,
    d_mins: np.ndarray,
    f_succs: np.ndarray,
    d_means: np.ndarray,
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
        f.create_dataset("npz_episode_index", data=npz_episode_indices.astype(np.int64))
        f.create_dataset("episode_d_min", data=d_mins.astype(np.float32))
        f.create_dataset("episode_f_succ", data=f_succs.astype(np.float32))
        f.create_dataset("episode_d_mean", data=d_means.astype(np.float32))
        if "qpos" in ds:
            qp = np.concatenate([ds["qpos"][s:e] for s, e in ranges], axis=0)
            qv = np.concatenate([ds["qvel"][s:e] for s, e in ranges], axis=0)
            f.create_dataset("qpos", data=qp, compression="gzip", compression_opts=4)
            f.create_dataset("qvel", data=qv, compression="gzip", compression_opts=4)

    print(f"  [h5] {path} ({human_bytes(path.stat().st_size)})")


def run_smoke() -> None:
    import ogbench  # noqa: F401
    import gymnasium as gym

    npz = _default_npz_path()
    if not npz.exists():
        print(f"SMOKE: skip distance check (no {npz})")
    else:
        ds = load_dataset_with_rewards(npz, compact_dataset=False, add_info=True)
        ranges = episode_ranges_from_dataset(ds)
        eid = _gym_env_id_for_family("cube-single")
        env = gym.make(eid, ob_type="states", render_mode="rgb_array")
        s, e = ranges[0]
        dists, _ = _episode_distance_trace(ds, env, s, e)
        print(f"SMOKE: ep0 dist min={dists.min():.4f} mean={dists.mean():.4f}")
        env.reset(seed=0)
        q0 = np.asarray(ds["qpos"][s], dtype=np.float64)
        v0 = np.asarray(ds["qvel"][s], dtype=np.float64)
        env.unwrapped.set_state(q0, v0)
        fr = env.render()
        assert fr is not None and fr.ndim == 3
        print(f"SMOKE: render ok shape={fr.shape}")
        env.close()
    print("SMOKE: PASS")


def main_export(*, top_k: int = TOP_K, npz_path: Path | None = None) -> Path:
    import ogbench  # noqa: F401
    import gymnasium as gym

    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        print(
            "FATAL: headless export needs DISPLAY (e.g. xvfb-run -a -s \"-screen 0 1024x768x24\" "
            "stable-worldmodel/.venv/bin/python cube_dataset/gt_export.py …) "
            "or MUJOCO_GL=egl on a GPU node. Same contract as download_and_replay_ogbench.py full replay. "
            "Shortcut: add --headless-xvfb to this script to re-exec under xvfb-run.",
            file=sys.stderr,
        )
        sys.exit(3)

    npz_path = Path(npz_path).resolve() if npz_path is not None else _default_npz_path()
    if not npz_path.exists():
        print(f"FATAL: missing {npz_path}", file=sys.stderr)
        sys.exit(2)

    dataset_stem = npz_path.stem
    family = _family_from_dataset_name(dataset_stem)

    pbs_id = os.environ.get("PBS_JOBID", "")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{pbs_id}" if pbs_id else ""
    run_dir = _cube_dataset_root() / "gt_export" / f"run_{stamp}{suffix}"
    fam_dir = run_dir / family
    fam_dir.mkdir(parents=True, exist_ok=True)
    (fam_dir / "full_episodes").mkdir(parents=True, exist_ok=True)
    (fam_dir / "clips").mkdir(parents=True, exist_ok=True)

    sw_py = sys.executable

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "pbs_jobid": pbs_id or None,
        "npz_path": str(npz_path.resolve()),
        "family": family,
        "top_k": top_k,
        "epsilon": EPSILON,
        "K_sustain": K_SUSTAIN,
        "N_pre": N_PRE,
        "N_post": N_POST,
        "L_min": L_MIN,
        "fps": FPS,
        "reward_frac": REWARD_FRAC,
        "export_python": sw_py,
        "vlm_mock": False,
        "git_commit": _git_commit(),
    }

    run_meta_path = run_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    ds = load_dataset_with_rewards(npz_path, compact_dataset=False, add_info=True)
    ranges = episode_ranges_from_dataset(ds)
    n_eps = len(ranges)
    print(f"  episodes in NPZ: {n_eps}")

    try:
        eid = _gym_env_id_for_family(family)
    except ValueError as ex:
        print(f"FATAL: {ex}", file=sys.stderr)
        sys.exit(2)

    env = gym.make(eid, ob_type="states", render_mode="rgb_array")

    d_mins_all, f_succs_all, d_means_all, r_sums_all = _score_all_episodes(ds, env, ranges)
    rewards_present = "rewards" in ds

    order = _rank_episode_indices(
        n_eps,
        rewards_present,
        r_sums_all,
        d_mins_all,
        f_succs_all,
        d_means_all,
        reward_frac=REWARD_FRAC,
    )
    picked_episode_idx = order[: min(top_k, len(order))]
    picked_ranges = [ranges[i] for i in picked_episode_idx]

    if rewards_present:
        totals = r_sums_all[np.array(picked_episode_idx, dtype=int)].astype(np.float32)
        ranking_mode = "rewards_sum"
    else:
        totals = (-d_mins_all[np.array(picked_episode_idx, dtype=int)]).astype(np.float32)
        ranking_mode = "distance_lexicographic"

    meta["ranking_mode"] = ranking_mode
    meta["rewards_present_in_npz"] = rewards_present
    run_meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    d_pick = d_mins_all[np.array(picked_episode_idx, dtype=int)]
    f_pick = f_succs_all[np.array(picked_episode_idx, dtype=int)]
    dm_pick = d_means_all[np.array(picked_episode_idx, dtype=int)]
    npz_idx_arr = np.array(picked_episode_idx, dtype=np.int64)

    h5_name = f"repick_top{len(picked_ranges)}.h5"
    h5_path = fam_dir / h5_name
    save_gt_h5(h5_path, ds, picked_ranges, totals, npz_idx_arr, d_pick, f_pick, dm_pick)

    rejects_path = fam_dir / "rejects.jsonl"
    manifest_path = fam_dir / "manifest.jsonl"
    rejects_f = rejects_path.open("w", encoding="utf-8")
    manifest_f = manifest_path.open("w", encoding="utf-8")

    for rank, ep_i in enumerate(picked_episode_idx):
        s, e = ranges[ep_i]
        dists, _ = _episode_distance_trace(ds, env, s, e)
        tr = _trim_episode(dists)
        full_vid = fam_dir / "full_episodes" / f"ep_{rank}.mp4"
        _replay_full_episode_to_mp4(env, ds, s, e, full_vid)
        print(f"    [full] {full_vid.name} ep_npz={ep_i}")

        if not tr.eligible:
            rejects_f.write(
                json.dumps(
                    {
                        "npz_episode_index": ep_i,
                        "rank": rank,
                        "reason": tr.reason,
                        "d_min": float(d_mins_all[ep_i]),
                        "f_succ": float(f_succs_all[ep_i]),
                        "d_mean": float(d_means_all[ep_i]),
                    }
                )
                + "\n"
            )
            continue

        clip_id = f"ep{ep_i}_r{rank}"
        clip_path = fam_dir / "clips" / f"{clip_id}.mp4"
        assert tr.t_start is not None and tr.t_end is not None and tr.t_star is not None
        _replay_segment_to_mp4(env, ds, s, e, tr.t_start, tr.t_end, clip_path)

        T = e - s
        q0 = ds["qpos"][s + tr.t_start]
        init_cube = q0[QPOS_OBJ_START : QPOS_OBJ_START + 3].astype(float).tolist()
        env.unwrapped.set_state(
            np.asarray(ds["qpos"][s], dtype=np.float64),
            np.asarray(ds["qvel"][s], dtype=np.float64),
        )
        tgt = np.asarray(env.unwrapped._data.mocap_pos.copy(), dtype=np.float64).reshape(-1, 3)[0]

        row = {
            "clip_id": clip_id,
            "video_path": str(clip_path.resolve()),
            "npz_episode_index": ep_i,
            "rank": rank,
            "t_start": tr.t_start,
            "t_end": tr.t_end,
            "t_star": tr.t_star,
            "epsilon": EPSILON,
            "K": K_SUSTAIN,
            "N_pre": N_PRE,
            "N_post": N_POST,
            "target_xyz": [float(tgt[0]), float(tgt[1]), float(tgt[2])],
            "init_cube_xyz": init_cube,
            "d_min": float(d_mins_all[ep_i]),
            "f_succ": float(f_succs_all[ep_i]),
            "d_mean": float(d_means_all[ep_i]),
            "eligible": True,
        }
        manifest_f.write(json.dumps(row) + "\n")
        print(f"    [clip] {clip_id}.mp4")

    rejects_f.close()
    manifest_f.close()

    env.close()

    # Pointer for PBS / follow-up scripts
    last_run = _cube_dataset_root() / "gt_export" / "_last_run_dir.txt"
    last_run.parent.mkdir(parents=True, exist_ok=True)
    last_run.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")

    print(f"GT_EXPORT_RUN_DIR={run_dir.resolve()}")
    return run_dir


def _xvfb_reexec_early() -> None:
    """If --headless-xvfb is present, re-exec this process under xvfb-run (sets DISPLAY)."""
    if "--headless-xvfb" not in sys.argv:
        return
    if "-h" in sys.argv or "--help" in sys.argv:
        return
    # Strip flag; nested xvfb is unnecessary if DISPLAY already set (e.g. user wrapped xvfb-run).
    if os.environ.get("DISPLAY"):
        while "--headless-xvfb" in sys.argv:
            sys.argv.remove("--headless-xvfb")
        return
    if os.environ.get("GT_EXPORT_INSIDE_XVFB"):
        while "--headless-xvfb" in sys.argv:
            sys.argv.remove("--headless-xvfb")
        return
    new_argv = [a for a in sys.argv if a != "--headless-xvfb"]
    os.environ["GT_EXPORT_INSIDE_XVFB"] = "1"
    # Xvfb provides DISPLAY; GLFW+X is reliable here. EGL/OSMesa often break on shared login nodes.
    os.environ["MUJOCO_GL"] = "glfw"
    cmd = ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", sys.executable] + new_argv
    os.execvp("xvfb-run", cmd)


def main() -> None:
    _xvfb_reexec_early()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless-xvfb",
        action="store_true",
        help="Re-exec under xvfb-run (virtual DISPLAY) when EGL/OSMesa is unavailable",
    )
    parser.add_argument("--smoke", action="store_true", help="Import/render sanity only")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        metavar="K",
        help=f"Top episodes to replay (default {TOP_K})",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to *-play-v0.npz (default: cube_dataset/datasets/cube-single-play-v0.npz)",
    )
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
        return

    t0 = time.time()
    run_dir = main_export(top_k=args.top_k, npz_path=args.npz)
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
