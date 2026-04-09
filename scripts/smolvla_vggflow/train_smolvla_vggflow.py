#!/usr/bin/env python3
"""Orchestrate staged SmolVLA finetuning jobs with optional JEPA+VGG branches."""

from __future__ import annotations

import argparse
import re
import os
import shlex
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, List

import torch
from torch import nn


def _compat_pythonpath(extra: str = "") -> str:
    compat_root = Path(__file__).resolve().parent / "compat"
    parts: List[str] = [str(compat_root)]
    if extra:
        parts.append(str(extra))
    existing = os.environ.get("PYTHONPATH", "").strip()
    if existing:
        parts.append(existing)
    return ":".join(parts)


_LEROBOT_TRAIN_FLAGS: set[str] | None = None


def _normalize_flag_name(name: str) -> str:
    clean = name.strip()
    if clean.startswith("--"):
        clean = clean[2:]
    for ch in (".",):
        clean = clean.replace(ch, "_")
    return clean.replace("-", "_")


def _train_flags_from_help(train_bin: str) -> set[str]:
    global _LEROBOT_TRAIN_FLAGS
    if _LEROBOT_TRAIN_FLAGS is not None:
        return _LEROBOT_TRAIN_FLAGS

    try:
        raw = subprocess.check_output(
            [train_bin, "--help"],
            text=True,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
        flags: set[str] = set()
        for token in re.findall(r"--[A-Za-z0-9][A-Za-z0-9._-]*", raw):
            flags.add(_normalize_flag_name(token))
        _LEROBOT_TRAIN_FLAGS = flags
        return flags
    except Exception:
        _LEROBOT_TRAIN_FLAGS = set()
        return _LEROBOT_TRAIN_FLAGS


def _append_flag_if_supported(cmd_parts: List[str], train_flags: set[str], flag: str, value: str | int | float | None) -> None:
    if value is None:
        return
    name = _normalize_flag_name(flag)
    for candidate in {name, name.replace("_", "-"), name.replace("_", ".")}:
        if _normalize_flag_name(candidate) in train_flags:
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{candidate}")
                return
            cmd_parts.append(f"--{candidate}")
            cmd_parts.append(str(value))
            return
    # Keep behavior resilient on older trainer builds where CLI options vary.
    if isinstance(value, bool):
        return
    print(f"WARN: unsupported lerobot option '{flag}', skipping from command")


def _extract_metrics_from_line(line: str) -> tuple[int | None, dict[str, float]]:
    step = None
    metrics: dict[str, float] = {}

    step_match = re.search(r"\b(?:step|global_step)\b[:\s=]?(\d+)", line, flags=re.IGNORECASE)
    if step_match:
        try:
            step = int(step_match.group(1))
        except Exception:
            step = None

    for token in re.finditer(
        r"\b(?P<name>loss|vgg_aux_match_loss|vgg_aux_value_loss|vgg_aux_total|aux_loss|base_loss|match_loss|value_loss)\b\s*[:=]\s*(?P<value>-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
        line,
    ):
        name = token.group("name").lower()
        if name in {"match_loss", "value_loss"}:
            name = f"vgg_aux_{name}"
        try:
            metrics[name] = float(token.group("value"))
        except Exception:
            continue
    return step, metrics


def _run(cmd: str, metrics_path: Path | None = None, log_interval: int = 100) -> None:
    print(f"[train-orch] running: {cmd}")
    start = time.time()
    if metrics_path is None:
        subprocess.check_call(cmd, shell=True)
        return

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    if proc.stdout is None:
        raise RuntimeError("training process failed to create stdout pipe")

    last_step: int | None = None
    prev_log_step: int | None = None
    prev_log_time: float | None = None
    sec_per_100_samples: list[float] = []
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path = metrics_path.parent / "timing_100step.jsonl"
    with metrics_path.open("w", encoding="utf-8") as mf, timing_path.open("w", encoding="utf-8") as tf:
        for line in proc.stdout:
            print(line, end="")
            step, metrics = _extract_metrics_from_line(line)
            if step is not None:
                last_step = step
                if log_interval > 0 and step % log_interval != 0:
                    continue
                now = time.time()
                row: dict[str, Any] = {
                    "step": step,
                    "elapsed_sec": now - start,
                    "optimizer_state": "unknown",
                }
                if metrics:
                    row.update(metrics)
                mf.write(json.dumps(row) + "\n")
                mf.flush()
                if log_interval > 0 and prev_log_step is not None and prev_log_time is not None:
                    ds = float(step - prev_log_step)
                    if ds > 0:
                        dt = now - prev_log_time
                        sec_per_100 = dt / ds * 100.0
                        sec_per_100_samples.append(sec_per_100)
                        tf.write(
                            json.dumps(
                                {
                                    "schema_version": "timing_window_v1",
                                    "window_end_step": step,
                                    "window_start_step": prev_log_step,
                                    "window_elapsed_sec": dt,
                                    "steps_in_window": int(ds),
                                    "sec_per_100_steps": sec_per_100,
                                }
                            )
                            + "\n"
                        )
                        tf.flush()
                prev_log_step = step
                prev_log_time = now

        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        if last_step is not None and log_interval <= 0:
            row = {
                "step": last_step,
                "elapsed_sec": time.time() - start,
                "optimizer_state": "unknown",
            }
            mf.write(json.dumps(row) + "\n")
        if sec_per_100_samples:
            sorted_s = sorted(sec_per_100_samples)
            n = len(sorted_s)

            def _pct(p: float) -> float:
                if n == 0:
                    return 0.0
                i = min(n - 1, max(0, int(p * (n - 1))))
                return float(sorted_s[i])

            rollup = {
                "schema_version": "timing_rollup_v1",
                "wall_time_sec": time.time() - start,
                "n_windows": n,
                "p50_sec_per_100_steps": _pct(0.50),
                "p90_sec_per_100_steps": _pct(0.90),
                "p95_sec_per_100_steps": _pct(0.95),
            }
            (metrics_path.parent / "timing_rollup.json").write_text(json.dumps(rollup, indent=2), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _infer_dataset_repo_id(dataset_root: Path) -> str:
    forced = os.environ.get("SMOLVLA_DATASET_REPO_ID", "").strip()
    if forced:
        return forced
    # LeRobot still validates `repo_id` against Hub refs even for local `--dataset.root` paths.
    # Use a stable public repo id unless the caller overrides via SMOLVLA_DATASET_REPO_ID.
    return "lerobot/pusht"


def _trace_value_head(
    action_dim: int,
    value_head_dim: int,
    trace_steps: int,
    seed: int = 0,
) -> dict[str, Any]:
    if action_dim <= 0 or trace_steps <= 0 or value_head_dim <= 0:
        return {
            "ok": False,
            "reason": "invalid_dimensions",
            "grad_norm": 0.0,
            "value": 0.0,
        }

    torch.manual_seed(seed)
    value_head = nn.Sequential(
        nn.Linear(action_dim * trace_steps, value_head_dim),
        nn.SiLU(),
        nn.Linear(value_head_dim, 1),
    )
    x_t = torch.randn((1, trace_steps, action_dim), requires_grad=True)
    scalar = value_head(x_t.reshape(1, -1)).mean()
    grad = torch.autograd.grad(scalar, x_t, allow_unused=True, create_graph=False)[
        0
    ]
    grad_norm = float(grad.abs().mean()) if grad is not None else 0.0
    return {
        "ok": grad is not None,
        "grad_norm": grad_norm,
        "value": float(scalar.item()),
        "value_head_weight_l2": float(sum(p.pow(2).sum().item() for p in value_head.parameters())),
        "value_head_shape": [int(v) for v in x_t.shape],
    }


def _gate_is_disabled(gate_json: Path | None) -> tuple[bool, str | None, dict[str, Any]]:
    if not gate_json or not gate_json.exists():
        return True, "missing_gate_json", {}
    try:
        gate = json.loads(gate_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return True, f"gate_parse_error:{exc}", {}
    if not gate.get("gate_ok", False):
        return True, "gate_ok_false", gate
    if gate.get("contract_ok") is False:
        return True, "contract_check_failed", gate
    return False, None, gate


def _write_vgg_aux_sitecustomize(path: Path, plan_path: Path) -> None:
    script = textwrap.dedent(
        """\
        # This file is generated by train_smolvla_vggflow.py for StageC only.
        from __future__ import annotations
        
        import copy
        import json
        import os
        import torch

        from torch import nn
        from lerobot.policies.smolvla import modeling_smolvla
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        
        PLAN_PATH = "__PLAN_PATH__"
        _PLAN = None
        try:
            if PLAN_PATH and os.path.exists(PLAN_PATH):
                with open(PLAN_PATH, "r", encoding="utf-8") as f:
                    _PLAN = json.load(f)
        except Exception:
            _PLAN = None


        def _safe_state(obj):
            return getattr(obj, "_vgg_aux_state", None)


        def _batch_device(batch):
            for value in batch.values():
                if torch.is_tensor(value):
                    return value.device
            return torch.device("cpu")

        def _call_denoise_step(model, x_t, prefix_pad_masks, past_key_values, t_tensor):
            for kwargs in [
                {
                    "x_t": x_t,
                    "prefix_pad_masks": prefix_pad_masks,
                    "past_key_values": past_key_values,
                    "timestep": t_tensor,
                },
                {
                    "x_t": x_t,
                    "prefix_pad_masks": prefix_pad_masks,
                    "past_key_values": past_key_values,
                    "timesteps": t_tensor,
                },
                {
                    "x_t": x_t,
                    "prefix_pad_masks": prefix_pad_masks,
                    "past_key_values": past_key_values,
                    "time": t_tensor,
                },
                {
                    "x_t": x_t,
                    "prefix_pad_masks": prefix_pad_masks,
                    "past_key_values": past_key_values,
                    "t": t_tensor,
                },
            ]:
                try:
                    out = model.denoise_step(**kwargs)
                except TypeError:
                    continue
                if isinstance(out, tuple):
                    return out[0]
                return out
            out = model.denoise_step(prefix_pad_masks, past_key_values, x_t, t_tensor)
            return out[0] if isinstance(out, tuple) else out

        def _build_prefix(model, batch):
            images, img_masks = model.prepare_images(batch)
            state = model.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            if lang_tokens is None or lang_masks is None:
                raise RuntimeError("missing language fields in batch")
            if not lang_tokens.numel():
                raise RuntimeError("language tensors are empty")
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
            )
            attention_mask = modeling_smolvla.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = model.vlm_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=model.config.use_cache,
                fill_kv_cache=True,
            )
            return state, past_key_values, prefix_pad_masks

        def _get_policy_state(policy):
            state = _safe_state(policy)
            if state is None:
                state = {
                    "plan": _PLAN,
                    "step": 0,
                    "base_model": None,
                    "value_head": None,
                }
                policy._vgg_aux_state = state
            return state

        def _ensure_value_head(policy):
            policy_state = _get_policy_state(policy)
            plan = policy_state.get("plan") or {}
            action_dim = int(plan.get("action_dim", policy.model.config.max_action_dim))
            trace_cap = max(1, int(plan.get("trace_cap", 3)))
            value_head_dim = max(1, int(plan.get("value_head_dim", 256)))
            value_head_input_dim = action_dim * min(
                trace_cap,
                int(policy.model.config.chunk_size),
            )
            vh = policy_state.get("value_head")
            if vh is None or not hasattr(vh, "to"):
                vh = nn.Sequential(
                    nn.Linear(value_head_input_dim, value_head_dim),
                    nn.SiLU(),
                    nn.Linear(value_head_dim, 1),
                )
                policy_state["value_head"] = vh
            return vh, action_dim, trace_cap

        def _ensure_base_model(policy):
            policy_state = _get_policy_state(policy)
            base_model = policy_state.get("base_model")
            if base_model is None or base_model is not policy.model:
                base_model = copy.deepcopy(policy.model)
                base_model.eval()
                for p in base_model.parameters():
                    p.requires_grad = False
                policy_state["base_model"] = base_model
            return base_model

        def _compute_vgg_terms(policy, batch):
            plan = _PLAN or {}
            policy_state = _get_policy_state(policy)
            if not plan or not plan.get("enabled", False):
                return None
            warmup = max(0, int(plan.get("match_warmup", 0)))
            match_weight = float(plan.get("match_weight", 0.0))
            value_head_weight = float(plan.get("value_head_weight", match_weight))

            if policy_state["step"] < warmup:
                policy_state["step"] += 1
                base_device = _batch_device(batch)
                return {
                    "match_loss": torch.tensor(0.0, device=base_device),
                    "value_loss": torch.tensor(0.0),
                    "aux_total": torch.tensor(0.0, device=base_device),
                    "aux_scale": 0.0,
                    "step": policy_state["step"],
                }

            model = policy.model
            base_model = _ensure_base_model(policy)
            value_head, action_dim, trace_cap = _ensure_value_head(policy)
            trace_steps = max(1, int(plan.get("value_head_steps", 6)))
            dt = -1.0 / float(trace_steps)

            state = model.prepare_state(batch)
            _, past_key_values, prefix_pad_masks = _build_prefix(model, batch)
            with torch.no_grad():
                _, base_past_key_values, _ = _build_prefix(base_model, batch)

            bsize = state.shape[0]
            device = state.device
            x_t = model.sample_noise((bsize, model.config.chunk_size, model.config.max_action_dim), device=device)
            x_t_base = x_t.detach().clone()
            value_head = value_head.to(device)

            residual = torch.tensor(0.0, device=device)
            for step in range(trace_steps):
                t_tensor = torch.full((bsize,), 1.0 + step * dt, device=device, dtype=torch.float32)
                v_t = _call_denoise_step(
                    model=model,
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    t_tensor=t_tensor,
                )
                with torch.no_grad():
                    v_base = _call_denoise_step(
                        model=base_model,
                        x_t=x_t_base,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=base_past_key_values,
                        t_tensor=t_tensor,
                    )
                if v_t.shape[-1] < action_dim or v_base.shape[-1] < action_dim:
                    return None
                residual = residual + ((v_t[:, :, :action_dim] - v_base[:, :, :action_dim]) ** 2).mean()
                x_t = x_t + dt * v_t
                x_t_base = x_t_base + dt * v_base

            match_loss = residual / float(trace_steps)
            trace_cap = min(trace_cap, int(x_t.shape[1]))
            value_input = x_t[:, :trace_cap, :action_dim].reshape(bsize, -1)
            if value_input.shape[-1] != value_head[0].in_features:
                return None
            value_loss = -value_head(value_input).mean()
            aux_scale = 1.0
            if warmup > 0:
                aux_scale = min(1.0, max(0.0, float(policy_state["step"] - warmup) / max(1, warmup)))

            policy_state["step"] += 1
            return {
                "match_loss": match_loss,
                "value_loss": value_loss,
                "aux_scale": aux_scale,
                "aux_total": aux_scale * (match_weight * match_loss + value_head_weight * value_loss),
            }

        def _patched_forward(self, batch, noise=None, time=None, reduction="mean"):
            loss, out = _ORIGINAL_FORWARD(self, batch, noise=noise, time=time, reduction=reduction)
            policy_state = _get_policy_state(self)
            if not policy_state.get("plan", {}).get("enabled", False):
                return loss, out
            if not hasattr(self, "training") or not self.training:
                return loss, out
            try:
                aux = _compute_vgg_terms(self, batch)
                if aux is None:
                    return loss, out
                aux_total = aux["aux_total"]
                if aux_total is None:
                    return loss, out
                if isinstance(loss, torch.Tensor) and loss.ndim == 0:
                    total = loss + aux_total
                else:
                    total = loss + aux_total
                out["vgg_aux_match_loss"] = float(aux["match_loss"].detach().item()) if torch.is_tensor(aux["match_loss"]) else 0.0
                out["vgg_aux_value_loss"] = float(aux["value_loss"].detach().item()) if torch.is_tensor(aux["value_loss"]) else 0.0
                out["vgg_aux_total"] = float(aux["aux_total"].detach().item()) if torch.is_tensor(aux["aux_total"]) else 0.0
                out["vgg_aux_scale"] = float(aux["aux_scale"])
                out["vgg_aux_step"] = int(policy_state["step"])
                return total, out
            except Exception:
                return loss, out

        if _PLAN is not None and _PLAN.get("enabled", False):
            _ORIGINAL_FORWARD = SmolVLAPolicy.forward
            SmolVLAPolicy.forward = _patched_forward
        """
    )
    script = script.replace("__PLAN_PATH__", str(plan_path).replace("\\", "/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")


def _build_base_policy_cmd(
    train_root: Path,
    output_dir: Path,
    args,
    extra_py_path: Path | str | None = None,
) -> str:
    ckpt = args.checkpoint
    repo_id = _infer_dataset_repo_id(train_root)
    report_to = str(getattr(args, "policy_report_to", "none")).strip()
    report_targets = [token.strip().lower() for token in report_to.replace(";", ",").split(",") if token.strip()]
    wandb_mode = os.environ.get("SMOLVLA_WANDB_MODE", "offline").strip().lower() or "offline"
    wandb_arg = f" --wandb.enable true --wandb.mode {shlex.quote(wandb_mode)}"
    if report_targets:
        report_targets = [token for token in report_targets if token != "none"]
        if report_targets and "wandb" not in report_targets:
            wandb_arg = ""
        if report_targets and "tensorboard" in report_targets and "wandb" not in report_targets:
            print("WARN: tensorboard telemetry requested but only wandb is supported in current leRobot version.")
    flags = _train_flags_from_help(args.lerobot_train_bin)
    cmd_parts: list[str] = [
        f"PYTHONPATH={shlex.quote(_compat_pythonpath(extra_py_path))}",
        shlex.quote(args.lerobot_train_bin),
        "--policy.type",
        "smolvla",
        "--policy.pretrained_path",
        shlex.quote(str(ckpt)),
        "--policy.load_vlm_weights",
        "true",
        "--policy.vlm_model_name",
        shlex.quote("HuggingFaceTB/SmolVLM2-500M-Instruct"),
        "--policy.expert_width_multiplier",
        "0.5",
        "--policy.self_attn_every_n_layers",
        "0",
        "--policy.n_action_steps",
        "1",
        "--dataset.root",
        shlex.quote(str(train_root)),
        "--dataset.repo_id",
        shlex.quote(repo_id),
        "--steps",
        str(args.max_steps),
        "--policy.push_to_hub",
        "false",
        "--output_dir",
        shlex.quote(str(output_dir)),
    ]
    if wandb_arg:
        cmd_parts.extend(shlex.split(wandb_arg))
    _append_flag_if_supported(cmd_parts, flags, "logging_steps", int(args.log_steps))
    _append_flag_if_supported(cmd_parts, flags, "save_steps", int(args.save_steps))
    return " ".join(cmd_parts)


def _prepare_train_output_dir(run_root: Path, *, leaf: str = "train_run") -> Path:
    """Avoid pre-created stage folders; lerobot-train requires a fresh output dir."""
    if not run_root.exists():
        return run_root
    candidate = run_root / leaf
    if candidate.exists():
        candidate = run_root / f"{leaf}_{int(time.time())}"
    return candidate


def _has_episode_parquets(root: Path) -> bool:
    return any(root.rglob("episode_*.parquet"))


def _resolve_legacy_split_root(root: Path) -> Path:
    """Prefer v2.1 roots for merge helper (supports auto-converted *_old splits)."""
    if _has_episode_parquets(root):
        return root
    alt = root.with_name(f"{root.name}_old")
    if alt.exists() and _has_episode_parquets(alt):
        return alt
    return root


def _convert_v21_root_to_v30(root: Path) -> None:
    import lerobot

    script = Path(lerobot.__file__).resolve().parent / "scripts" / "convert_dataset_v21_to_v30.py"
    cmd = [
        sys.executable,
        str(script),
        "--repo-id=local/stageB_mixed_dataset",
        "--root",
        str(root),
        "--push-to-hub=false",
    ]
    subprocess.check_call(cmd)


def _strict_vgg_train() -> bool:
    return os.environ.get("SMOLVLA_STRICT_VGG_TRAIN", "").strip() == "1"


def _run_vgg_aux(
    gate_json: Path | None,
    output_dir: Path,
    args,
    *,
    train_lane: str = "stageC",
) -> int:
    vgg_aux_dir = output_dir / ("vgg_aux_imagined" if train_lane == "stageD" else "vgg_aux")
    vgg_aux_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir = _prepare_train_output_dir(vgg_aux_dir)
    gate = {"gate_ok": False}
    disabled_reason: str | None = None
    strict = _strict_vgg_train()

    if gate_json and gate_json.exists():
        try:
            gate = json.loads(gate_json.read_text(encoding="utf-8"))
        except Exception as exc:
            gate = {"gate_ok": False, "error": str(exc)}
        if not gate.get("gate_ok", False):
            disabled_reason = "gate_ok false"
        if gate.get("contract_ok") is False:
            disabled_reason = "contract_check_failed"
        gic = str(gate.get("init_checkpoint", "")).strip()
        ckpt = str(args.checkpoint).strip()
        if gic and ckpt and gic != ckpt:
            disabled_reason = "gate_init_checkpoint_mismatch"

    if disabled_reason is None:
        disabled, disabled_reason_parsed, _ = _gate_is_disabled(gate_json)
        if disabled:
            disabled_reason = disabled_reason_parsed

    if disabled_reason:
        payload = {
            "enabled": False,
            "reason": disabled_reason,
            "gate": gate,
        }
        _write_json(vgg_aux_dir / "disabled_reason.json", payload)
        return 1 if strict else 0

    action_dim = int(gate.get("velocity_shape", [0, 0, 0])[-1]) if gate.get("velocity_shape") else args.value_head_dim
    if action_dim <= 0:
        action_dim = args.value_head_dim

    probe = _trace_value_head(
        action_dim=action_dim,
        value_head_dim=args.value_head_dim,
        trace_steps=max(1, args.value_head_steps),
        seed=args.value_head_seed,
    )
    _write_json(vgg_aux_dir / "value_head_probe.json", probe)
    if not probe["ok"] or probe["grad_norm"] < args.value_head_min_grad:
        payload = {
            "enabled": False,
            "reason": "value_head_probe_failed",
            "value_head_probe": probe,
            "gate": gate,
        }
        _write_json(vgg_aux_dir / "disabled_reason.json", payload)
        return 1 if strict else 0

    match_plan = {
        "enabled": True,
        "train_lane": train_lane,
        "match_weight": float(args.match_weight),
        "match_warmup": int(args.match_warmup),
        "value_head_dim": int(args.value_head_dim),
        "value_head_steps": int(args.value_head_steps),
        "value_head_seed": int(args.value_head_seed),
        "trace_cap": max(1, int(args.trace_cap)),
        "action_dim": int(action_dim),
        "value_head_weight": float(args.match_weight),
        "value_head_ok": probe["ok"],
        "value_head_grad_norm": float(probe["grad_norm"]),
        "gate": gate,
        "plan_source": "train_smolvla_vggflow.py",
    }
    _write_json(vgg_aux_dir / "match_objective_plan.json", match_plan)

    config_payload = {
        "match_weight": args.match_weight,
        "match_warmup": args.match_warmup,
        "value_head_dim": args.value_head_dim,
        "value_head_steps": args.value_head_steps,
        "value_head_seed": args.value_head_seed,
        "trace_cap": args.trace_cap,
        "gate_json": str(gate_json) if gate_json else "",
        "output_dir": str(vgg_aux_dir),
        "train_output_dir": str(train_output_dir),
    }
    _write_json(vgg_aux_dir / "vgg_aux_config.json", config_payload)

    sitecustomize = vgg_aux_dir / "sitecustomize.py"
    plan_path = vgg_aux_dir / "match_objective_plan.json"
    _write_vgg_aux_sitecustomize(sitecustomize, plan_path)

    if args.dry_run:
        env_prefix = f"SMOLVLA_VGG_AUX_PLAN={shlex.quote(str(plan_path))}"
        cmd = (
            f"{env_prefix} "
            f"{_build_base_policy_cmd(Path(os.path.expanduser(args.real_data_root)), train_output_dir, args, extra_py_path=vgg_aux_dir)}"
        )
        print(cmd)
        print(f"[train-orch] VGG auxiliary config written: {vgg_aux_dir}")
        return 0

    existing_py_path = os.environ.get("PYTHONPATH", "")
    env_prefix = f"SMOLVLA_VGG_AUX_PLAN={shlex.quote(str(plan_path))} "
    cmd = (
        f"{env_prefix}"
        f"{_build_base_policy_cmd(Path(os.path.expanduser(args.real_data_root)), train_output_dir, args, extra_py_path=vgg_aux_dir)}"
    )
    _run(cmd, vgg_aux_dir / "metrics.jsonl", args.log_steps)
    _write_json(vgg_aux_dir / "completed.json", {"enabled": True, "status": "completed"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["stageA", "stageB", "stageC", "stageD"],
        required=True,
        help="stageD is the same VGG auxiliary path as stageC (imagined-heavy dataset via --real-data-root).",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lerobot-env", required=True)
    parser.add_argument("--real-data-root", required=True)
    parser.add_argument("--jepa-data-root", default="")
    parser.add_argument("--output-dir", default="artifacts/train_stage")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--gate-json", default="")
    parser.add_argument("--match-weight", type=float, default=0.05)
    parser.add_argument("--match-warmup", type=int, default=200)
    parser.add_argument("--value-head-dim", type=int, default=256)
    parser.add_argument("--value-head-steps", type=int, default=6)
    parser.add_argument("--value-head-seed", type=int, default=0)
    parser.add_argument("--value-head-min-grad", type=float, default=1.0e-8)
    parser.add_argument("--trace-cap", type=int, default=3)
    parser.add_argument(
        "--policy-report-to",
        default=os.environ.get("SMOLVLA_TRAIN_REPORT_TO", "none"),
        help="Comma-separated telemetry targets (e.g. wandb,tensorboard). "
        "In this env, only wandb is currently wired via --wandb.enable.",
    )
    parser.add_argument(
        "--log-steps",
        type=int,
        default=int(os.environ.get("SMOLVLA_TRAIN_LOG_STEPS", "100")),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=int(os.environ.get("SMOLVLA_TRAIN_SAVE_STEPS", "2000")),
    )
    parser.add_argument("--lerobot-train-bin", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.lerobot_train_bin = (
        str(os.path.expanduser(args.lerobot_train_bin))
        if args.lerobot_train_bin
        else str(Path(args.lerobot_env) / "bin" / "lerobot-train")
    )
    real_data_root = Path(os.path.expanduser(args.real_data_root))
    jepa_data_root = Path(os.path.expanduser(args.jepa_data_root)) if args.jepa_data_root else None
    output_root = Path(os.path.expanduser(args.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    gate_json = Path(args.gate_json) if args.gate_json else None

    if args.mode in {"stageB", "stageC", "stageD"} and not real_data_root.exists():
        print(f"[train-orch] stage data root missing: {real_data_root}")
        return 1

    if args.mode == "stageA":
        if not real_data_root.exists():
            print(f"[train-orch] stageA missing real data root: {real_data_root}")
            return 1
        cmd = _build_base_policy_cmd(real_data_root, _prepare_train_output_dir(output_root), args)
        if args.dry_run:
            print(cmd)
            return 0
        _run(cmd, output_root / "metrics.jsonl", args.log_steps)
        return 0

    if args.mode == "stageB":
        if not jepa_data_root or not jepa_data_root.exists():
            print(f"[train-orch] stageB synthetic data root unavailable: {jepa_data_root}")
            return 1
        mixed_root = output_root / "mixed_lerobot_b"
        merge_py = Path(__file__).resolve().parent / "merge_lerobot_v21_datasets.py"
        merge_real_root = _resolve_legacy_split_root(real_data_root)
        merge_jepa_root = _resolve_legacy_split_root(jepa_data_root)
        if not args.dry_run:
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        str(merge_py),
                        "--real-root",
                        str(merge_real_root),
                        "--jepa-root",
                        str(merge_jepa_root),
                        "--out",
                        str(mixed_root),
                    ]
                )
            except subprocess.CalledProcessError:
                print(
                    "[train-orch] merge_lerobot_v21_datasets failed "
                    "(need pyarrow; at least one root must contain episode_*.parquet — see merge_lerobot stderr)"
                )
                return 1
            try:
                _convert_v21_root_to_v30(mixed_root)
            except subprocess.CalledProcessError:
                print(
                    "[train-orch] stageB v2.1->v3.0 conversion failed for mixed dataset "
                    f"at {mixed_root}"
                )
                return 1
        cmd = _build_base_policy_cmd(mixed_root, _prepare_train_output_dir(output_root / "jepa_mix"), args)
        if args.dry_run:
            print(f"# merged dataset would be at {mixed_root}")
            print(f"# merge real root: {merge_real_root}")
            print(f"# merge jepa root: {merge_jepa_root}")
            print(cmd)
            return 0
        _run(cmd, output_root / "jepa_mix" / "metrics.jsonl", args.log_steps)
        return 0

    if args.mode in {"stageC", "stageD"}:
        return _run_vgg_aux(gate_json, output_root, args, train_lane=args.mode)

    raise RuntimeError(f"unknown mode {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
