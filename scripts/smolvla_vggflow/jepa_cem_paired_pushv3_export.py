#!/usr/bin/env python3
"""Paired push-v3 rollouts: executed trajectories + JEPA-WM CEM-planned latents.

Writes trajectories.pt (list of episode dicts) and export_manifest.json for bridge_builder.
See docs/CEM_PAIRED_PUSHV3_SCHEMA.md for the contract.

- **Executed actions:** When the world model loads, uses the first action of a short CEM plan
  (MPC-style). Otherwise falls back to a non-random heuristic push controller.
- **Latent / CEM arm:** Records per-step CEM metadata and a latent summary from WM unroll
  when available; if WM is unavailable, writes explicit wm_skipped metadata (still
  export_mode cem_paired_push_v3 for pipeline wiring).
- **CUDA:** WM+CEM runs whenever the hub model loads; ``device`` follows ``--device`` /
  availability (CPU ok for **smoke** / dev; Slurm phase07 should still use a GPU job).
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = "cem_paired_push_v3_v0"
EXPORT_MODE = "cem_paired_push_v3"


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


def heuristic_push_action(obs: Any, env: Any) -> np.ndarray:
    """Non-random push-v3 prior: move in the direction object -> goal in proprio slice."""
    adim = int(np.prod(env.action_space.shape))
    if isinstance(obs, dict):
        flat = _flatten_obs_state(obs)
        o = np.asarray(flat, dtype=np.float32).reshape(-1)
    else:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
    if o.size < 12:
        return np.zeros(adim, dtype=np.float32)
    # Typical MT1 Sawyer layouts: gripper(4) object(3) ... goal(3) at end
    obj = o[4:7] if o.size > 10 else o[:3]
    goal = o[-3:] if o.size >= 15 else obj
    delta = (goal - obj)[:adim]
    n = float(np.linalg.norm(delta) + 1e-6)
    vec = (delta / n) * 2.0
    return np.clip(vec, -1.0, 1.0).astype(np.float32)


def _render_to_wm_visual(env: Any, device: torch.device) -> torch.Tensor | None:
    """Return visual tensor (1,1,3,256,256) float, or None."""
    try:
        frame = env.render()
    except Exception:
        return None
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    t = torch.from_numpy(arr).float()
    if t.max() > 1.5:
        t = t / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    t = torch.nn.functional.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return t.unsqueeze(0).to(device)  # 1,1,3,256,256


def _build_proprio(flat_state: list[float], proprio_dim: int, device: torch.device) -> torch.Tensor:
    v = np.asarray(flat_state, dtype=np.float32).reshape(-1)
    if v.size >= proprio_dim:
        p = v[:proprio_dim].copy()
    else:
        p = np.zeros(proprio_dim, dtype=np.float32)
        p[: v.size] = v
    t = torch.from_numpy(p).float().view(1, 1, -1).to(device)
    return t


def _resolve_ckpt(ckpt_hint: str) -> str:
    if not ckpt_hint:
        return "jepa_wm_metaworld.pth.tar"
    maybe = Path(ckpt_hint)
    if maybe.is_file():
        return str(maybe.resolve())
    hf_home = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_home.exists():
        matches = sorted(hf_home.rglob(ckpt_hint))
        if matches:
            return str(matches[0].resolve())
    return ckpt_hint


def _try_load_wm(repo: Path, ckpt: str, device: torch.device) -> tuple[Any, Any] | None:
    try:
        import os

        if not os.environ.get("JEPAWM_LOGS"):
            os.environ["JEPAWM_LOGS"] = str((Path.home() / ".cache" / "jepa_wm").resolve())
        os.environ["JEPAWM_CKPT"] = _resolve_ckpt(ckpt)
        if repo.is_dir():
            model, preprocessor = torch.hub.load(
                str(repo), "jepa_wm_metaworld", source="local", pretrained=True, device=str(device)
            )
        else:
            model, preprocessor = torch.hub.load(
                str(repo), "jepa_wm_metaworld", source="github", pretrained=True, device=str(device)
            )
        model.eval()
        return model, preprocessor
    except Exception:
        return None


def _infer_action_dim(model: Any, preprocessor: Any) -> int:
    for dim in (
        int(getattr(getattr(preprocessor, "action_mean", None), "numel", lambda: 0)() or 0),
        int(getattr(getattr(model, "model", None), "action_dim", 0) or 0),
    ):
        if dim > 0:
            return dim
    return 4


def _score_unroll(z_pred: Any) -> float:
    """Higher is better (CEM maximizes)."""
    try:
        if isinstance(z_pred, dict):
            lat = z_pred.get("latent")
            if lat is None:
                lat = next(iter(z_pred.values()))
        else:
            lat = z_pred
        if not torch.is_tensor(lat):
            return 0.0
        return float(-lat.pow(2).mean().item())
    except Exception:
        return 0.0


def cem_first_action(
    model: Any,
    z: torch.Tensor,
    action_dim: int,
    horizon: int,
    pop_size: int,
    cem_iters: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return first action of best CEM sequence and debug metadata."""
    best_seq = torch.zeros(horizon, action_dim, device=device, dtype=torch.float32)
    best_score = -1e18
    best_z_pred: Any = None
    for _ in range(cem_iters):
        for _p in range(pop_size):
            noise = torch.randn(horizon, action_dim, device=device, dtype=torch.float32) * 0.35
            seq = torch.clamp(best_seq + noise, -1.0, 1.0)
            try:
                act_suffix = seq.unsqueeze(1)
                z_pred = model.unroll(z, act_suffix=act_suffix, debug=False)
                sc = _score_unroll(z_pred)
                if sc > best_score:
                    best_score = sc
                    best_seq = seq.detach().clone()
                    best_z_pred = z_pred
            except Exception:
                continue
    meta = {
        "cem_iterations": int(cem_iters * pop_size),
        "cem_cost": float(-best_score),
        "cem_seed": int(rng.integers(0, 2**31 - 1)),
        "cem_horizon": horizon,
        "cem_population": pop_size,
    }
    a0 = best_seq[0].detach().cpu().numpy().reshape(-1)
    latent_summary: list[float] = []
    if best_z_pred is not None:
        try:
            if isinstance(best_z_pred, dict):
                lat = best_z_pred.get("latent", list(best_z_pred.values())[0])
            else:
                lat = best_z_pred
            if torch.is_tensor(lat):
                latent_summary = lat.detach().float().cpu().reshape(-1)[:256].tolist()
        except Exception:
            pass
    return a0, {"meta": meta, "latent_pred": latent_summary}


def rollout_episode(
    env: Any,
    max_steps: int,
    pair_key: str,
    wm_bundle: tuple[Any, Any, int, torch.device] | None,
    cem_horizon: int,
    cem_pop: int,
    cem_iters: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
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
    cem_steps: list[dict[str, Any]] = []

    model = preproc = proprio_dim = None
    device = torch.device("cpu")
    action_dim = int(np.prod(env.action_space.shape))
    if wm_bundle is not None:
        model, preproc, proprio_dim, device = wm_bundle
        action_dim = _infer_action_dim(model, preproc)

    policy_used = "heuristic"

    for step_idx in range(max_steps):
        img = _find_image(obs)
        if img is not None:
            images.append(_to_rgb_list(img))
        st = _flatten_obs_state(obs)
        states.append(st)

        step_record: dict[str, Any] = {
            "step_index": step_idx,
            "cem_iterations": 0,
            "cem_cost": 0.0,
            "cem_seed": seed,
            "latent_pred": [],
            "planner_metadata": {},
        }

        # WM+CEM whenever the world model loaded (CPU is fine for smoke tests; use GPU in Slurm).
        if model is not None and preproc is not None:
            try:
                vis = _render_to_wm_visual(env, device)
                if vis is None:
                    raise RuntimeError("no render")
                pr = _build_proprio(st, proprio_dim, device)
                obs_wm = {"visual": vis, "proprio": pr}
                with torch.no_grad():
                    z = model.encode(obs_wm)
                z = z.to(device)
                a_cem, cem_dbg = cem_first_action(
                    model, z, action_dim, cem_horizon, cem_pop, cem_iters, device, rng
                )
                a_exec = np.clip(a_cem.reshape(-1)[: env.action_space.shape[0]], -1.0, 1.0).astype(
                    np.float32
                )
                policy_used = "cem_mpc_wm"
                step_record.update(
                    {
                        "cem_iterations": cem_dbg["meta"]["cem_iterations"],
                        "cem_cost": cem_dbg["meta"]["cem_cost"],
                        "cem_seed": cem_dbg["meta"]["cem_seed"],
                        "latent_pred": cem_dbg["latent_pred"],
                        "planner_metadata": {
                            "horizon": cem_horizon,
                            "population": cem_pop,
                        },
                    }
                )
            except Exception as exc:
                step_record["planner_metadata"] = {"wm_step_error": str(exc)[:200]}
                a_exec = heuristic_push_action(obs, env)
                policy_used = "heuristic_fallback"
        else:
            if wm_bundle is None:
                step_record["planner_metadata"] = {"wm_skipped": True}
            a_exec = heuristic_push_action(obs, env)

        a_list = np.asarray(a_exec, dtype=np.float32).reshape(-1).tolist()
        actions.append(a_list)
        cem_steps.append(step_record)

        step_out = env.step(a_exec)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        if done:
            break

    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    return {
        "images": images,
        "state": states,
        "actions": actions,
        "action_chunk": actions,
        "language": "push the puck to the goal",
        "done": True,
        "success": success,
        "pair_key": pair_key,
        "cem_plan": {"per_step": cem_steps},
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "pair_key": pair_key,
            "export_mode": EXPORT_MODE,
            "task": "push-v3",
            "steps": len(actions),
            "policy": policy_used,
            "pairing": "executed_latent_aligned",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="push-v3")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jepa-repo", type=Path, default=None, help="Path to local jepa-wms repo (with hubconf)")
    ap.add_argument("--jepa-ckpt", type=str, default="")
    ap.add_argument("--cem-horizon", type=int, default=4)
    ap.add_argument("--cem-pop", type=int, default=8)
    ap.add_argument("--cem-iters", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev_name = args.device
    if dev_name == "auto":
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"

    import metaworld  # noqa: PLC0415

    ml1 = metaworld.ML1(args.task)
    env_cls = ml1.train_classes[args.task]
    env = env_cls()
    try:
        if hasattr(env, "render_mode"):
            env.render_mode = "rgb_array"
    except Exception:
        pass

    if dev_name == "cuda" and not torch.cuda.is_available():
        dev_name = "cpu"
    dev = torch.device(dev_name)
    wm_bundle: tuple[Any, Any, int, torch.device] | None = None
    repo = args.jepa_repo
    if repo is not None and repo.is_dir():
        loaded = _try_load_wm(repo, args.jepa_ckpt or "jepa_wm_metaworld.pth.tar", dev)
        if loaded is not None:
            model, preprocessor = loaded
            proprio_dim = int(getattr(preprocessor, "proprio_mean").numel())
            wm_bundle = (model, preprocessor, proprio_dim, dev)

    rng = np.random.default_rng(args.seed)
    episodes: list[dict[str, Any]] = []
    for ep in range(args.episodes):
        pk = str(uuid.uuid4())
        episodes.append(
            rollout_episode(
                env,
                args.max_steps,
                pk,
                wm_bundle,
                args.cem_horizon,
                args.cem_pop,
                args.cem_iters,
                rng,
            )
        )

    try:
        env.close()
    except Exception:
        pass

    args.out.mkdir(parents=True, exist_ok=True)
    traj_path = args.out / "trajectories.pt"
    torch.save(episodes, traj_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "export_mode": EXPORT_MODE,
        "trajectories_file": str(traj_path.name),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "task_id": args.task,
        "jepa_ckpt": args.jepa_ckpt or _resolve_ckpt("jepa_wm_metaworld.pth.tar"),
        "pairing": "executed_latent_aligned",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "cem_horizon": args.cem_horizon,
        "cem_pop": args.cem_pop,
        "cem_iters": args.cem_iters,
        "wm_loaded": wm_bundle is not None,
        "bridge_hint": "Point SMOLVLA_JEPA_SOURCE at this directory for phase08.",
    }
    (args.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[cem_paired_export] wrote {len(episodes)} episodes -> {traj_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
