#!/usr/bin/env python3
"""Shared config and transform helpers for cube-single pi0.5 LoRA pipeline."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

JOINT_0_OFFSET = float(np.pi)
DEFAULT_PROMPT = "pick up the red cube"
DEFAULT_CONFIG_NAME = "pi05_cube_single_v2_lora"
DEFAULT_BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base/params"


def _parse_image_hwc_uint8(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape={arr.shape}")
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape={arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = np.asarray(arr, dtype=np.uint8)
    return np.ascontiguousarray(arr)


@dataclasses.dataclass(frozen=True)
class UR5Inputs(_transforms.DataTransformFn):
    """Map cube UR5e dataset records to OpenPI policy input schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        joints = np.asarray(data["joints"], dtype=np.float32).reshape(-1)
        if joints.size != 6:
            raise ValueError(f"Expected joints shape (6,), got {joints.shape}")
        joints = joints.copy()
        joints[0] += JOINT_0_OFFSET

        gripper_open = float(np.clip(np.asarray(data["gripper"], dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        # OpenPI convention for this path: 0=open, 1=closed.
        gripper_model = np.asarray([1.0 - gripper_open], dtype=np.float32)
        state = np.concatenate([joints, gripper_model], axis=0)

        base_image = _parse_image_hwc_uint8(data["base_rgb"])
        wrist_image = _parse_image_hwc_uint8(data["wrist_rgb"])

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_image, np.zeros_like(base_image))
            image_masks = (np.True_, np.True_, np.False_)
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), wrist_image)
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model_type={self.model_type}")

        out = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "prompt": str(data.get("prompt", DEFAULT_PROMPT)),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            out["actions"] = actions[:7] if actions.ndim == 1 else actions[..., :7]
        return out


@dataclasses.dataclass(frozen=True)
class UR5Outputs(_transforms.DataTransformFn):
    """Keep 7D UR5e outputs only."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 1:
            return {"actions": actions[:7]}
        return {"actions": actions[..., :7]}


@dataclasses.dataclass(frozen=True)
class CubeUR5eDataConfig(_config.DataConfigFactory):
    """Data config for local cube-single LeRobot UR5e LoRA data."""

    default_prompt: str = DEFAULT_PROMPT
    base_config: _config.DataConfig | None = dataclasses.field(
        default_factory=lambda: _config.DataConfig(prompt_from_task=True)
    )

    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "base_rgb",
                        "wrist_rgb": "wrist_rgb",
                        "joints": "joints",
                        "gripper": "gripper",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )
        delta_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_mask)],
            outputs=[_transforms.AbsoluteActions(delta_mask)],
        )
        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(Path(assets_dirs), model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def build_train_config(
    *,
    repo_id: str,
    run_root: Path,
    exp_name: str,
    checkpoint_path: str = DEFAULT_BASE_CHECKPOINT,
    batch_size: int = 64,
    num_train_steps: int = 10_000,
    log_interval: int = 100,
    save_interval: int = 2_000,
    keep_period: int = 2_000,
    fsdp_devices: int = 1,
    resume: bool = False,
    overwrite: bool = False,
    wandb_enabled: bool = False,
    peak_lr: float = 5e-5,
    warmup_steps: int = 1_000,
    decay_steps: int = 1_000_000,
    decay_lr: float = 5e-5,
    seed: int = 42,
) -> _config.TrainConfig:
    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=16,
    )
    freeze_filter = pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=16,
    ).get_freeze_filter()

    run_root = Path(run_root).resolve()
    assets_base_dir = run_root / "assets"
    checkpoint_base_dir = run_root / "checkpoints"

    return _config.TrainConfig(
        name=DEFAULT_CONFIG_NAME,
        exp_name=exp_name,
        seed=seed,
        model=model_cfg,
        data=CubeUR5eDataConfig(repo_id=repo_id),
        weight_loader=_weight_loaders.CheckpointWeightLoader(checkpoint_path),
        freeze_filter=freeze_filter,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            decay_steps=decay_steps,
            decay_lr=decay_lr,
        ),
        ema_decay=None,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        log_interval=log_interval,
        save_interval=save_interval,
        keep_period=keep_period,
        fsdp_devices=fsdp_devices,
        num_workers=2,
        assets_base_dir=str(assets_base_dir),
        checkpoint_base_dir=str(checkpoint_base_dir),
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=wandb_enabled,
    )


def training_summary_dict(config: _config.TrainConfig, repo_id: str, run_root: Path) -> dict:
    return {
        "config_name": config.name,
        "exp_name": config.exp_name,
        "repo_id": repo_id,
        "run_root": str(Path(run_root).resolve()),
        "assets_dirs": str(config.assets_dirs),
        "checkpoint_dir": str(config.checkpoint_dir),
        "batch_size": int(config.batch_size),
        "num_train_steps": int(config.num_train_steps),
        "log_interval": int(config.log_interval),
        "save_interval": int(config.save_interval),
        "keep_period": int(config.keep_period) if config.keep_period is not None else None,
        "fsdp_devices": int(config.fsdp_devices),
        "resume": bool(config.resume),
        "overwrite": bool(config.overwrite),
        "wandb_enabled": bool(config.wandb_enabled),
    }

