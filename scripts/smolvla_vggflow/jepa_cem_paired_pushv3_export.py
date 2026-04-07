#!/usr/bin/env python3
"""Paired push-v3 rollouts: executed trajectories + JEPA-WM CEM-planned latents.

Writes trajectories.pt (list of episode dicts) and export_manifest.json for bridge_builder.
See docs/CEM_PAIRED_PUSHV3_SCHEMA.md for the contract.

- **Executed actions:** If ``--policy-checkpoint`` (or ``SMOLVLA_INIT_CHECKPOINT``) is set and
  LeRobot loads, uses ``SmolVLA`` + hub ``policy_preprocessor`` / ``policy_postprocessor``
  (same path as ``lerobot-eval``). Otherwise, when the world model loads, uses the first
  action of a short CEM plan; else a non-random heuristic push controller.
- **Latent / CEM arm:** Whenever the WM loads, still runs CEM on the latent unroll and
  records per-step metadata (independent of which controller produced ``a_exec``).
- **CUDA:** WM+CEM runs whenever the hub model loads; ``device`` follows ``--device`` /
  availability (CPU ok for **smoke** / dev; Slurm phase07 should still use a GPU job).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import site
import sys
import uuid
from dataclasses import dataclass
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


def _patch_external_datasets() -> None:
    """Avoid local ``lerobot.datasets`` shadowing HuggingFace ``datasets`` when importing policies."""
    for item in site.getsitepackages() + [site.getusersitepackages() or ""]:
        if not item:
            continue
        path = Path(item) / "datasets" / "__init__.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("datasets", str(path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.__file__ = str(path)
                sys.modules["datasets"] = module
                return


@dataclass
class SmolVLAExecBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device


def _try_load_smolvla_exec(checkpoint: str, device: torch.device) -> SmolVLAExecBundle | None:
    ckpt = (checkpoint or "").strip()
    if not ckpt:
        return None
    try:
        _patch_external_datasets()
        import inspect  # noqa: PLC0415

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415
        from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
        from lerobot.processor import PolicyProcessorPipeline  # noqa: PLC0415
        from lerobot.processor.converters import (  # noqa: PLC0415
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )

        policy_dev_raw = os.environ.get("SMOLVLA_JEPA_EXPORT_POLICY_DEVICE", "").strip().lower()
        if policy_dev_raw in ("", "default"):
            dev = device
        elif policy_dev_raw == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif policy_dev_raw == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        else:
            dev = torch.device(policy_dev_raw)
        dev_str = str(dev)
        load_vlm_raw = os.environ.get(
            "SMOLVLA_JEPA_EXPORT_POLICY_LOAD_VLM_WEIGHTS", "1"
        )
        load_vlm_weights = load_vlm_raw.strip().lower() not in ("0", "false", "no")
        print(
            f"[cem_paired_export] policy load_vlm raw='{load_vlm_raw}' parsed={load_vlm_weights}"
        )
        print(
            f"[cem_paired_export] policy device raw='{policy_dev_raw or '<default>'}' resolved='{dev_str}'"
        )
        model_kwargs: dict[str, Any] = {
            "device": dev_str,
            "n_action_steps": 1,
            "expert_width_multiplier": 0.5,
            "self_attn_every_n_layers": 0,
            "load_vlm_weights": load_vlm_weights,
            "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Instruct",
        }
        sig = inspect.signature(SmolVLAPolicy.from_pretrained)
        params = sig.parameters
        pretrained_keys = (
            "pretrained_name_or_path",
            "pretrained_model_name_or_path",
            "pretrained_path",
        )
        pretrained_key = next((k for k in pretrained_keys if k in params), None)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kwargs:
            supported_kwargs = dict(model_kwargs)
        else:
            supported_kwargs = {k: v for k, v in model_kwargs.items() if k in params}
        config_override_applied = False
        if "config" in params:
            try:
                policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=ckpt)
                for key, value in model_kwargs.items():
                    if hasattr(policy_cfg, key):
                        setattr(policy_cfg, key, value)
                supported_kwargs["config"] = policy_cfg
                config_override_applied = True
                print(
                    "[cem_paired_export] policy config override:"
                    f" load_vlm_weights={getattr(policy_cfg, 'load_vlm_weights', None)}"
                    f" device={getattr(policy_cfg, 'device', None)}"
                )
            except Exception as exc:
                print(f"[cem_paired_export] policy config override skipped: {exc}")
        print(f"[cem_paired_export] policy config override applied={config_override_applied}")
        overrides = {"device_processor": {"device": dev_str}}
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt,
            config_filename="policy_preprocessor.json",
            overrides=overrides,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt,
            config_filename="policy_postprocessor.json",
            overrides=overrides,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        if pretrained_key:
            supported_kwargs[pretrained_key] = ckpt
            policy = SmolVLAPolicy.from_pretrained(**supported_kwargs)
        else:
            policy = SmolVLAPolicy.from_pretrained(ckpt, **supported_kwargs)
        print(
            "[cem_paired_export] loaded policy config:"
            f" load_vlm_weights={getattr(getattr(policy, 'config', None), 'load_vlm_weights', None)}"
        )
        policy.eval()
        return SmolVLAExecBundle(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=dev,
        )
    except Exception:
        return None


def _smolvla_state_dims(policy: Any) -> tuple[int, int]:
    """Return (agent_state_dim, environment_state_dim) from policy input features with MetaWorld-ish defaults."""
    agent_dim, env_dim = 4, 39
    feats = getattr(getattr(policy, "config", None), "input_features", None) or {}
    for name, ft in feats.items():
        sh = getattr(ft, "shape", None) or ()
        dim0 = int(sh[0]) if sh else 0
        if not dim0:
            continue
        if "environment_state" in name:
            env_dim = dim0
        elif name.endswith(".state") and "environment" not in name:
            agent_dim = dim0
    return agent_dim, env_dim


def _vectors_for_smolvla(flat: np.ndarray, agent_dim: int, env_dim: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    env_vec = np.zeros(env_dim, dtype=np.float32)
    env_vec[: min(env_dim, flat.size)] = flat[: min(env_dim, flat.size)]
    agent_vec = np.zeros(agent_dim, dtype=np.float32)
    agent_vec[:] = env_vec[:agent_dim]
    return agent_vec, env_vec


def _policy_rgb_hwc(env: Any, obs: Any) -> np.ndarray:
    img = None
    if isinstance(obs, dict):
        raw_img = _find_image(obs)
        if raw_img is not None:
            img = np.asarray(raw_img)
    if img is None:
        frame = env.render()
        if frame is None:
            raise RuntimeError("no camera frame")
        img = np.asarray(frame)
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise RuntimeError("bad image shape")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        if float(np.max(img)) <= 1.5:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _smolvla_exec_action(
    bundle: SmolVLAExecBundle,
    obs: Any,
    env: Any,
    task_text: str,
) -> np.ndarray:
    from lerobot.utils.constants import (  # noqa: PLC0415
        OBS_ENV_STATE,
        OBS_IMAGE,
        OBS_STATE,
    )

    agent_dim, env_dim = _smolvla_state_dims(bundle.policy)
    if isinstance(obs, np.ndarray):
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    else:
        flat = np.asarray(_flatten_obs_state(obs), dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise RuntimeError("empty state vector")
    _agent_vec, env_vec = _vectors_for_smolvla(flat, agent_dim, env_dim)
    rgb = _policy_rgb_hwc(env, obs)
    timg = torch.from_numpy(rgb).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float() / 255.0
    timg = timg.to(bundle.device)
    st = torch.from_numpy(_agent_vec).unsqueeze(0).to(bundle.device)
    es = torch.from_numpy(env_vec).unsqueeze(0).to(bundle.device)
    batch = {
        OBS_IMAGE: timg,
        OBS_STATE: st,
        OBS_ENV_STATE: es,
        "task": task_text,
    }
    proc = bundle.preprocessor(batch)
    with torch.inference_mode():
        act = bundle.policy.select_action(proc)
    act = bundle.postprocessor(act)
    out = act.detach().float().cpu().numpy().reshape(-1)
    return out


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
    delta = goal - obj
    n = float(np.linalg.norm(delta) + 1e-6)
    planar = (delta / n) * 2.0
    vec = np.zeros(adim, dtype=np.float32)
    m = min(adim, int(planar.shape[0]))
    vec[:m] = np.clip(planar[:m], -1.0, 1.0)
    return vec


def _render_to_wm_visual(env: Any, obs: Any, device: torch.device) -> torch.Tensor | None:
    """Return visual tensor (1,1,3,256,256) float, or None.

    Prefer pixels already in ``obs`` (avoids an extra ``env.render()`` when available).
    """
    frame = None
    if isinstance(obs, dict):
        raw = _find_image(obs)
        if raw is not None:
            frame = np.asarray(raw)
    if frame is None:
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


def _infer_action_dims(model: Any, preprocessor: Any) -> list[int]:
    """Match ``jepa_smoke_check._infer_action_dims``: WM may accept 20-D actions while env is 4-D."""
    dims: list[int] = []

    def _add(value: object) -> None:
        try:
            dim = int(value)
        except Exception:
            return
        if dim > 0 and dim not in dims:
            dims.append(dim)

    _add(getattr(getattr(preprocessor, "action_mean", None), "numel", lambda: 0)() or 0)
    model_module = getattr(model, "model", None)
    if model_module is not None:
        _add(getattr(model_module, "action_dim", 0) or 0)
        _add(getattr(getattr(model_module, "action_encoder", None), "in_features", 0) or 0)
        predictor = getattr(model_module, "predictor", None)
        if predictor is not None:
            _add(getattr(getattr(predictor, "action_encoder", None), "in_features", 0) or 0)
    if not dims:
        dims = [4]
    return dims


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
    wm_bundle: tuple[Any, Any, int, int, torch.device] | None,
    smolvla_bundle: SmolVLAExecBundle | None,
    task_text: str,
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

    if smolvla_bundle is not None:
        smolvla_bundle.policy.reset()
        smolvla_bundle.preprocessor.reset()
        smolvla_bundle.postprocessor.reset()

    images: list[Any] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []
    cem_steps: list[dict[str, Any]] = []

    model = preproc = proprio_dim = None
    device = torch.device("cpu")
    env_action_dim = int(np.prod(env.action_space.shape))
    planner_action_dim = env_action_dim
    if wm_bundle is not None:
        model, preproc, proprio_dim, planner_action_dim, device = wm_bundle

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
            "planner_metadata": {
                "planner_action_dim": int(planner_action_dim),
                "env_action_dim": int(env_action_dim),
            },
        }

        a_cem: np.ndarray | None = None
        if model is not None and preproc is not None:
            try:
                vis = _render_to_wm_visual(env, obs, device)
                if vis is None:
                    raise RuntimeError("no render")
                pr = _build_proprio(st, proprio_dim, device)
                obs_wm = {"visual": vis, "proprio": pr}
                with torch.no_grad():
                    z = model.encode(obs_wm)
                z = z.to(device)
                a_cem, cem_dbg = cem_first_action(
                    model,
                    z,
                    planner_action_dim,
                    cem_horizon,
                    cem_pop,
                    cem_iters,
                    device,
                    rng,
                )
                meta = dict(step_record.get("planner_metadata") or {})
                meta.update(
                    {
                        "horizon": cem_horizon,
                        "population": cem_pop,
                    }
                )
                step_record.update(
                    {
                        "cem_iterations": cem_dbg["meta"]["cem_iterations"],
                        "cem_cost": cem_dbg["meta"]["cem_cost"],
                        "cem_seed": cem_dbg["meta"]["cem_seed"],
                        "latent_pred": cem_dbg["latent_pred"],
                        "planner_metadata": meta,
                    }
                )
            except Exception as exc:
                meta = dict(step_record.get("planner_metadata") or {})
                meta["wm_step_error"] = str(exc)[:200]
                step_record["planner_metadata"] = meta
        elif wm_bundle is None:
            step_record["planner_metadata"] = {"wm_skipped": True}

        a_exec: np.ndarray | None = None
        policy_used = "heuristic"
        if smolvla_bundle is not None:
            try:
                raw_act = _smolvla_exec_action(smolvla_bundle, obs, env, task_text)
                a_exec = np.clip(
                    raw_act.reshape(-1)[:env_action_dim],
                    -1.0,
                    1.0,
                ).astype(np.float32)
                policy_used = "smolvla"
            except Exception as exc:
                meta = dict(step_record.get("planner_metadata") or {})
                meta["policy_exec_error"] = str(exc)[:200]
                step_record["planner_metadata"] = meta

        if a_exec is None:
            if a_cem is not None:
                a_exec = np.clip(
                    a_cem.reshape(-1)[:env_action_dim], -1.0, 1.0
                ).astype(np.float32)
                policy_used = "cem_mpc_wm"
            else:
                a_exec = heuristic_push_action(obs, env)
                policy_used = "heuristic_fallback" if model is not None else "heuristic"

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
    ap.add_argument(
        "--policy-checkpoint",
        default=os.environ.get("SMOLVLA_INIT_CHECKPOINT", ""),
        help="SmolVLA HF id or local dir; empty disables. Default: $SMOLVLA_INIT_CHECKPOINT.",
    )
    args = ap.parse_args()

    dev_name = args.device
    if dev_name == "auto":
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_name == "cuda" and not torch.cuda.is_available():
        dev_name = "cpu"
    dev = torch.device(dev_name)

    policy_ckpt = (args.policy_checkpoint or "").strip()
    smolvla_bundle = _try_load_smolvla_exec(policy_ckpt, dev) if policy_ckpt else None
    task_text = "push the puck to the goal"

    wm_bundle: tuple[Any, Any, int, int, torch.device] | None = None
    repo = args.jepa_repo
    skip_wm = os.environ.get("SMOLVLA_JEPA_EXPORT_SKIP_WM", "").strip().lower() in ("1", "true", "yes")
    if skip_wm:
        print("[cem_paired_export] SMOLVLA_JEPA_EXPORT_SKIP_WM=1: skipping WM hub load (heuristic / env-only rollouts)")
    elif repo is not None and repo.is_dir():
        loaded = _try_load_wm(repo, args.jepa_ckpt or "jepa_wm_metaworld.pth.tar", dev)
        if loaded is not None:
            model, preprocessor = loaded
            proprio_dim = int(getattr(preprocessor, "proprio_mean").numel())
            action_dims = _infer_action_dims(model, preprocessor)
            planner_action_dim = max(action_dims)
            print(
                f"[cem_paired_export] wm action_dim candidates={action_dims}, "
                f"planner_action_dim={planner_action_dim} (env uses 4-D MT1 actions)"
            )
            wm_bundle = (model, preprocessor, proprio_dim, planner_action_dim, dev)

    import metaworld  # noqa: PLC0415

    ml1 = metaworld.ML1(args.task, seed=int(args.seed))
    env_cls = ml1.train_classes[args.task]
    env = env_cls()
    try:
        if hasattr(env, "render_mode"):
            env.render_mode = "rgb_array"
    except Exception:
        pass
    tasks = getattr(ml1, "train_tasks", None)
    if tasks:
        env.set_task(tasks[0])

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
                smolvla_bundle,
                task_text,
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
        "wm_planner_action_dim": (wm_bundle[3] if wm_bundle is not None else None),
        "wm_skipped_export": bool(skip_wm),
        "policy_checkpoint": policy_ckpt or None,
        "policy_loaded": smolvla_bundle is not None,
        "policy_load_vlm_weights": os.environ.get("SMOLVLA_JEPA_EXPORT_POLICY_LOAD_VLM_WEIGHTS", "1"),
        "bridge_hint": "Point SMOLVLA_JEPA_SOURCE at this directory for phase08.",
    }
    (args.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[cem_paired_export] wrote {len(episodes)} episodes -> {traj_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
