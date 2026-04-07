#!/usr/bin/env python3
"""Validate SmolVLA velocity-flow primitives and value-gradient prerequisites."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import site
import sys

import torch
from torch import nn


def _patch_external_datasets() -> None:
    candidates = []
    for item in site.getsitepackages() + [site.getusersitepackages() or ""]:
        if not item:
            continue
        path = Path(item) / "datasets" / "__init__.py"
        if path.exists():
            candidates.append(path)
    for path in candidates:
        spec = importlib.util.spec_from_file_location("datasets", str(path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.__file__ = str(path)
            sys.modules["datasets"] = module
            return


def _shape_to_list(value) -> list[int] | None:
    if not hasattr(value, "shape"):
        return None
    return [int(v) for v in value.shape]


def _infer_input_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _check_model_contract(policy) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    model = getattr(policy, "model", None)
    if model is None:
        return False, ["missing_policy_model"]

    required_model_attrs = ["denoise_step", "embed_prefix", "vlm_with_expert", "config"]
    for name in required_model_attrs:
        if not hasattr(model, name):
            reasons.append(f"missing_model_attr:{name}")

    required_config_attrs = ["chunk_size", "max_action_dim", "max_state_dim", "resize_imgs_with_padding"]
    for name in required_config_attrs:
        if hasattr(model, "config") and not hasattr(model.config, name):
            reasons.append(f"missing_config_attr:{name}")
    return len(reasons) == 0, reasons


def _call_denoise_step(model, x_t: torch.Tensor, prefix_pad_masks, past_key_values, t_tensor: torch.Tensor) -> torch.Tensor:
    for kwargs in [
        {"x_t": x_t, "prefix_pad_masks": prefix_pad_masks, "past_key_values": past_key_values, "timestep": t_tensor},
        {"x_t": x_t, "prefix_pad_masks": prefix_pad_masks, "past_key_values": past_key_values, "timesteps": t_tensor},
        {"x_t": x_t, "prefix_pad_masks": prefix_pad_masks, "past_key_values": past_key_values, "time": t_tensor},
        {"x_t": x_t, "prefix_pad_masks": prefix_pad_masks, "past_key_values": past_key_values, "t": t_tensor},
    ]:
        try:
            return model.denoise_step(**kwargs)
        except TypeError:
            continue
    return model.denoise_step(prefix_pad_masks, past_key_values, x_t, t_tensor)


def _build_trace(
    model,
    base_model,
    inputs,
    num_steps: int,
    device: torch.device,
    trace_max_steps: int = 0,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], list[float], torch.Tensor | None, torch.Tensor | None]:
    from lerobot.policies.smolvla import modeling_smolvla  # noqa: PLC0415

    images, img_masks, lang_tokens, lang_masks, state = inputs
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
    _, base_past_key_values = base_model.vlm_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=base_model.config.use_cache,
        fill_kv_cache=True,
    )

    bsize = state.shape[0]
    num_steps = max(1, int(num_steps))
    trace_max_steps = max(0, int(trace_max_steps))
    x_t = torch.randn(
        bsize, model.config.chunk_size, model.config.max_action_dim, device=device, requires_grad=True
    )
    x_t_base = x_t.detach().clone()

    trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    diff_stats: list[float] = []
    dt = -1.0 / num_steps
    last_v = None
    last_v_base = None

    for step in range(num_steps):
        t = 1.0 + step * dt
        t_tensor = torch.full((bsize,), t, device=device, dtype=torch.float32)

        v_t = _call_denoise_step(
            model=model,
            x_t=x_t,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            t_tensor=t_tensor,
        )
        with torch.no_grad():
            # Keep baseline model in a no-grad branch for a stable reference trajectory.
            v_base = _call_denoise_step(
                model=base_model,
                x_t=x_t_base,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=base_past_key_values,
                t_tensor=t_tensor,
            )

        if v_t.shape != v_base.shape:
            raise RuntimeError(
                f"denoise_step shape mismatch: v_t={_shape_to_list(v_t)} v_base={_shape_to_list(v_base)}"
            )

        x_t = x_t + dt * v_t
        x_t_base = x_t_base + dt * v_base
        if trace_max_steps == 0 or step < trace_max_steps:
            trace.append((x_t.detach().cpu(), v_t.detach().cpu(), v_base.detach().cpu()))

        delta = (v_t - v_base).abs().mean().item()
        if torch.isnan(torch.tensor(delta)) or torch.isinf(torch.tensor(delta)):
            diff_stats.append(0.0)
        else:
            diff_stats.append(float(delta))
        last_v = v_t
        last_v_base = v_base

    return trace, diff_stats, last_v, last_v_base


def _value_head_step(x_t: torch.Tensor, hidden_dim: int) -> dict:
    value_head = nn.Sequential(
        nn.Linear(x_t.shape[-1], hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, x_t.shape[-1]),
    ).to(x_t.device)
    scalar_value = value_head(x_t).mean()
    grad = torch.autograd.grad(scalar_value, x_t, retain_graph=False, create_graph=False, allow_unused=True)[0]
    grad_norm = float(grad.abs().mean()) if grad is not None else 0.0
    return {"value": float(scalar_value.item()), "grad_norm": grad_norm, "ok": grad is not None}


def _serialize_trace(
    trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    max_batch: int = 0,
) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for step, (x_t, v_t, v_base) in enumerate(trace):
        x_slice = x_t
        v_slice = v_t
        b_slice = v_base
        if max_batch > 0 and x_t.shape[0] > max_batch:
            x_slice = x_slice[:max_batch]
            v_slice = v_slice[:max_batch]
            b_slice = b_slice[:max_batch]
        serialized.append(
            {
                "step": step,
                "x_t": x_slice.tolist(),
                "v_t": v_slice.tolist(),
                "v_base": b_slice.tolist(),
            }
        )
    return serialized


def _clone_or_share_model(model: torch.nn.Module, target_device: torch.device) -> tuple[torch.nn.Module, bool]:
    try:
        cloned = copy.deepcopy(model).to(target_device)
        return cloned, False
    except Exception:
        return model, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    parser.add_argument("--value-head-dim", type=int, default=256)
    parser.add_argument("--value-head-grad-min", type=float, default=1.0e-8)
    parser.add_argument("--max-base-flow-diff", type=float, default=10.0)
    parser.add_argument("--emit-trace", action="store_true", help="Write optional velocity trace file")
    parser.add_argument("--trace-max-steps", type=int, default=0, help="Number of steps to serialize in trace")
    parser.add_argument("--trace-max-batch", type=int, default=0, help="Max number of batch entries to serialize")
    parser.add_argument("--trace-path", default="", help="Explicit path for trace payload if emit-trace is set")
    parser.add_argument(
        "--skip-flow-check",
        action="store_true",
        help="Skip flow rollout check; keep gate as contract/value-head-only mode.",
    )
    args = parser.parse_args()

    _patch_external_datasets()

    device = _infer_input_device(args.device)
    from lerobot.policies.smolvla import modeling_smolvla
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model_kwargs = {
        "device": device,
        "n_action_steps": 1,
        "expert_width_multiplier": 0.5,
        "self_attn_every_n_layers": 0,
        "load_vlm_weights": True,
        "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Instruct",
    }
    sig = inspect.signature(SmolVLAPolicy.from_pretrained)
    params = sig.parameters
    pretrained_keys = (
        "pretrained_name_or_path",
        "pretrained_model_name_or_path",
        "pretrained_path",
    )
    pretrained_key = None
    for key in pretrained_keys:
        if key in params:
            pretrained_key = key
            break

    supported_kwargs = {k: v for k, v in model_kwargs.items() if k in params}
    if pretrained_key:
        supported_kwargs[pretrained_key] = args.checkpoint
        policy = SmolVLAPolicy.from_pretrained(**supported_kwargs)
    else:
        policy = SmolVLAPolicy.from_pretrained(args.checkpoint, **supported_kwargs)

    policy.eval()
    model = policy.model

    # Dual-branch baseline: keep an unmodified reference copy for velocity residuals.
    model.eval()
    base, baseline_shared = _clone_or_share_model(model, device)
    base_status = "baseline_model_shared" if baseline_shared else "baseline_model_copied"
    base.eval()

    contract_ok, contract_reasons = _check_model_contract(policy)

    b = max(1, int(args.episodes))
    lang_len = 16
    h, w = model.config.resize_imgs_with_padding

    with torch.no_grad():
        # embed_prefix iterates zip(images, img_masks); each images[i] must be (B, C, H, W), not a bare 4D tensor as the iterable.
        images = [torch.rand((b, 3, h, w), device=device)]
    img_masks = [torch.ones((b,), dtype=torch.bool, device=device)]
    lang_tokens = torch.zeros((b, lang_len), dtype=torch.long, device=device)
    lang_masks = torch.ones((b, lang_len), dtype=torch.bool, device=device)
    state = torch.zeros((b, model.config.max_state_dim), dtype=torch.float32, device=device)

    if args.skip_flow_check:
        v_t_init = torch.zeros(
            (b, model.config.chunk_size, model.config.max_action_dim),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        trace: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        diffs: list[float] = []
        last_v = v_t_init
    else:
        trace, diffs, last_v, _ = _build_trace(
            model=model,
            base_model=base,
            inputs=(images, img_masks, lang_tokens, lang_masks, state),
            num_steps=args.steps,
            device=device,
            trace_max_steps=args.trace_max_steps,
        )
    if trace:
        v_sample = trace[-1][1]
    elif last_v is not None:
        v_sample = last_v
    else:
        v_sample = torch.zeros(
            (b, model.config.chunk_size, model.config.max_action_dim),
            device=device,
            dtype=torch.float32,
        )
    # Trace tensors are detached for serialization; value-head probe needs a leaf with grad.
    v_leaf = v_sample.to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
    value_info = _value_head_step(v_leaf, hidden_dim=args.value_head_dim)
    finite_diffs = [d for d in diffs if d == d and d != float("inf") and d != -float("inf")]
    base_flow_diff_max = float(max(finite_diffs)) if finite_diffs else 0.0
    base_flow_diff_mean = float(sum(finite_diffs) / len(finite_diffs)) if finite_diffs else 0.0
    value_head_grad_ok = bool(value_info["ok"]) and value_info["grad_norm"] >= args.value_head_grad_min
    base_flow_ok = base_flow_diff_max <= args.max_base_flow_diff
    gate_ok = (bool(trace) or args.skip_flow_check) and value_head_grad_ok and base_flow_ok and contract_ok

    gate_reasons = []
    if not contract_ok:
        gate_reasons.extend(contract_reasons)
    if not trace and not args.skip_flow_check:
        gate_reasons.append("velocity_trace_empty")
    if args.skip_flow_check:
        gate_reasons.append("velocity_trace_skipped")
    if not value_info["ok"]:
        gate_reasons.append("value_head_no_grad")
    if value_info["grad_norm"] < args.value_head_grad_min:
        gate_reasons.append("value_head_grad_too_small")
    if base_flow_diff_max > args.max_base_flow_diff:
        gate_reasons.append("base_flow_diff_too_large")

    report = {
        "schema_version": "smolvla_gate_v1",
        "emit_utc": datetime.now(timezone.utc).isoformat(),
        "init_checkpoint": str(args.checkpoint),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "baseline_status": base_status,
        "velocity_trace_ok": bool(trace) or args.skip_flow_check,
        "velocity_trace_skipped": bool(args.skip_flow_check),
        "contract_ok": contract_ok,
        "contract_reasons": contract_reasons,
        "velocity_shape": list(v_sample.shape),
        "base_flow_diff_max": base_flow_diff_max,
        "base_flow_diff_mean": base_flow_diff_mean,
        "value_head_ok": bool(value_info["ok"]),
        "value_head_grad_norm": float(value_info["grad_norm"]),
        "value_head_value": float(value_info["value"]),
        "device": str(device),
        "value_head_grad_ok": value_head_grad_ok,
        "base_flow_ok": base_flow_ok,
        "gate_ok": gate_ok,
        "gate_reasons": gate_reasons,
    }

    if args.emit_trace:
        trace_path = Path(args.trace_path) if args.trace_path else Path(args.output).parent / "vgg_velocity_trace.json"
        serialized_trace = _serialize_trace(trace, max_batch=args.trace_max_batch)
        payload = {
            "metadata": {
                "checkpoint": args.checkpoint,
                "episodes": args.episodes,
                "steps": args.steps,
                "trace_steps_emitted": len(serialized_trace),
                "device": str(device),
            },
            "frames": serialized_trace,
        }
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report["trace_path"] = str(trace_path)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
