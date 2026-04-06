#!/usr/bin/env python3
"""LEGACY: random-action MetaWorld export. Not used by phase07.

phase07 calls `jepa_cem_paired_pushv3_export.py` (CEM + paired latents). Kept for reference only.

Schema: list of episode dicts with images, state, actions, action_chunk, language, done, success, meta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_rgb_list(arr: Any) -> list[list[list[float]]]:
    if arr is None:
        return []
    x = np.asarray(arr)
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x.astype(np.float32) / 255.0
    return x.tolist()


def _flatten_obs_state(obs: Any) -> list[float]:
    if isinstance(obs, dict):
        for k in ("observation.state", "state", "agent_pos", "observation"):
            if k in obs:
                v = obs[k]
                return np.asarray(v, dtype=np.float32).reshape(-1).tolist()
        # fall back: first array-like value
        for v in obs.values():
            if hasattr(v, "shape"):
                return np.asarray(v, dtype=np.float32).reshape(-1).tolist()
    return np.asarray(obs, dtype=np.float32).reshape(-1).tolist()


def _find_image(obs: Any) -> Any:
    if not isinstance(obs, dict):
        return None
    for k in ("image", "top", "pixels/top", "observation.image", "rgb"):
        if k in obs:
            return obs[k]
    for v in obs.values():
        if hasattr(v, "shape") and len(getattr(v, "shape", ())) == 3:
            return v
    return None


def rollout_one(env: Any, max_steps: int, rng: np.random.Generator) -> dict[str, Any]:
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple):
        obs = out[0]
        info = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
    else:
        obs, info = out, {}

    images: list[Any] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []

    for _ in range(max_steps):
        img = _find_image(obs)
        if img is not None:
            images.append(_to_rgb_list(img))
        states.append(_flatten_obs_state(obs))

        action = env.action_space.sample()
        a = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
        actions.append(a)

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        if done:
            break

    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    pair_key = f"mw_{seed}"
    return {
        "images": images,
        "state": states,
        "actions": actions,
        "action_chunk": actions,
        "language": "push the puck to the goal",
        "done": True,
        "success": success,
        "meta": {
            "schema_version": "jepa_rollout_export_v1",
            "pair_key": pair_key,
            "export_mode": "metaworld_random_policy",
            "task": "push-v3",
            "steps": len(actions),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push-v3")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="Directory; writes trajectories.pt + export_manifest.json")
    args = ap.parse_args()

    import metaworld  # noqa: PLC0415 — after argparse for fast --help

    ml1 = metaworld.ML1(args.task)
    env_cls = ml1.train_classes[args.task]
    env = env_cls()
    try:
        if hasattr(env, "render_mode"):
            env.render_mode = "rgb_array"
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    episodes: list[dict[str, Any]] = []
    for _ in range(args.episodes):
        episodes.append(rollout_one(env, args.max_steps, rng))

    try:
        env.close()
    except Exception:
        pass

    args.out.mkdir(parents=True, exist_ok=True)
    traj_path = args.out / "trajectories.pt"
    torch.save(episodes, traj_path)

    manifest = {
        "schema_version": "jepa_export_manifest_v1",
        "task": args.task,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "trajectory_file": str(traj_path.name),
        "bridge_hint": "Point SMOLVLA_JEPA_SOURCE at this directory for phase08.",
    }
    (args.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[jepa_export] wrote {len(episodes)} episodes -> {traj_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
