"""
Pi0.5-base UR5e policy wrapper for OGBench joint-target control.
act() returns [1, 7]: 6 joint targets + 1 gripper (OpenPI convention: 0=open, 1=closed).
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import os
import pathlib

# Python 3.11+ only; some OpenPI / HF deps reference datetime.UTC on 3.10.
if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc

import numpy as np
import torch


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


class Pi05UR5ePolicy:
    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path
        self._openpi_policy = None
        self._openpi_failed_reason: str | None = None

        if checkpoint_path is None or str(checkpoint_path).lower().startswith("heuristic"):
            print(f"Pi05UR5ePolicy: heuristic mode (no OpenPI). ckpt={checkpoint_path}")
            return

        try:
            # Some cluster images miss a default CA bundle for aiohttp/gcsfs.
            # Set certifi bundle proactively so OpenPI checkpoint downloads work.
            if "SSL_CERT_FILE" not in os.environ or not os.environ.get("SSL_CERT_FILE"):
                try:
                    import certifi

                    ca_path = certifi.where()
                    os.environ["SSL_CERT_FILE"] = ca_path
                    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_path)
                except Exception:
                    pass

            from openpi.models import model as _model
            from openpi.policies import policy_config
            from openpi.shared import download
            from openpi.shared import normalize as _normalize
            from openpi.training import config as openpi_train_config
            from openpi import transforms as _transforms

            @dataclasses.dataclass(frozen=True)
            class UR5Inputs(_transforms.DataTransformFn):
                model_type: _model.ModelType

                def __call__(self, data: dict) -> dict:
                    gripper_pos = np.asarray(data["gripper"], dtype=np.float32).reshape(-1)
                    if gripper_pos.size == 0:
                        gripper_pos = np.zeros((1,), dtype=np.float32)
                    elif gripper_pos.size > 1:
                        gripper_pos = gripper_pos[:1]
                    state = np.concatenate([np.asarray(data["joints"], dtype=np.float32), gripper_pos], axis=0)

                    base_image = _parse_image(data["base_rgb"])
                    wrist_image = _parse_image(data["wrist_rgb"])

                    match self.model_type:
                        case _model.ModelType.PI0 | _model.ModelType.PI05:
                            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                            images = (base_image, wrist_image, np.zeros_like(base_image))
                            image_masks = (np.True_, np.True_, np.False_)
                        case _model.ModelType.PI0_FAST:
                            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                            images = (base_image, np.zeros_like(base_image), wrist_image)
                            image_masks = (np.True_, np.True_, np.True_)
                        case _:
                            raise ValueError(f"Unsupported model type: {self.model_type}")

                    inputs = {
                        "state": state,
                        "image": dict(zip(names, images, strict=True)),
                        "image_mask": dict(zip(names, image_masks, strict=True)),
                    }
                    if "actions" in data:
                        inputs["actions"] = np.asarray(data["actions"])
                    if "prompt" in data:
                        p = data["prompt"]
                        if isinstance(p, bytes):
                            p = p.decode("utf-8")
                        inputs["prompt"] = p
                    return inputs

            @dataclasses.dataclass(frozen=True)
            class UR5Outputs(_transforms.DataTransformFn):
                def __call__(self, data: dict) -> dict:
                    return {"actions": np.asarray(data["actions"][:, :7])}

            path_str = str(checkpoint_path)
            norm_stats = None
            if path_str.startswith("gs://openpi-assets/checkpoints/"):
                params_dir = download.maybe_download(path_str.rstrip("/") + "/params")
                assets_ur5e_dir = download.maybe_download(path_str.rstrip("/") + "/assets/ur5e")
                ckpt_dir = pathlib.Path(params_dir).parent
                norm_stats = _normalize.load(assets_ur5e_dir)
            else:
                ckpt_dir = path_str

            base_cfg = openpi_train_config.get_config("pi05_droid")
            delta_mask = _transforms.make_bool_mask(6, -1)
            ur5_data_cfg = openpi_train_config.SimpleDataConfig(
                assets=openpi_train_config.AssetsConfig(asset_id="ur5e"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[UR5Inputs(model_type=model.model_type)],
                    outputs=[UR5Outputs()],
                ).push(
                    inputs=[_transforms.DeltaActions(delta_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_mask)],
                ),
                base_config=openpi_train_config.DataConfig(prompt_from_task=True),
            )
            cfg = dataclasses.replace(base_cfg, name="pi05_base_ur5e_inference", data=ur5_data_cfg)
            self._openpi_policy = policy_config.create_trained_policy(cfg, ckpt_dir, norm_stats=norm_stats)
            print(f"Pi05UR5ePolicy: OpenPI loaded config='pi05_base_ur5e_inference' dir={ckpt_dir!r}")
        except Exception as exc:
            self._openpi_failed_reason = str(exc)
            print(f"Pi05UR5ePolicy: OpenPI load failed ({exc}); using zero-action fallback.")

    @classmethod
    def load_policy(cls, checkpoint_path):
        return cls(checkpoint_path)

    def _build_openpi_request(self, observation: dict) -> dict:
        from src.envs.droid.observation_openpi_ur5e import build_openpi_ur5e_request_from_tensors

        obs = observation.get("obs")
        if obs is None:
            raise ValueError("observation must include 'obs' image tensor")

        joints_6 = observation.get("joints_6")
        if joints_6 is None:
            state = observation.get("state")
            if state is None:
                raise ValueError("observation must include 'joints_6' or 7D 'state'")
            st = torch.as_tensor(state, dtype=torch.float32).view(-1).detach().cpu().numpy()
            if st.size < 7:
                raise ValueError(f"state must have at least 7 values, got {st.size}")
            joints_6 = st[:6]
            gripper_open_01 = float(st[6])
        else:
            gripper_open_01 = float(observation.get("gripper_open_01", 1.0))

        inst = observation.get("instruction", "") or ""
        wrist_obs = observation.get("wrist_obs")
        return build_openpi_ur5e_request_from_tensors(
            obs, joints_6, gripper_open_01, inst, wrist_image_chw=wrist_obs,
        )

    def _compute_action_tensor(self, observation: dict) -> torch.Tensor:
        if self._openpi_policy is not None:
            req = self._build_openpi_request(observation)
            out = self._openpi_policy.infer(req)
            actions = np.asarray(out["actions"])
            if actions.ndim == 1:
                a0 = actions
            elif actions.ndim == 2:
                a0 = actions[0]
            elif actions.ndim >= 3:
                a0 = actions[0, 0]
            else:
                raise ValueError(f"unexpected OpenPI actions shape {actions.shape}")
            a0 = np.asarray(a0, dtype=np.float32).reshape(-1)
            if a0.size < 7:
                pad = np.zeros((7 - a0.size,), dtype=np.float32)
                a0 = np.concatenate([a0, pad], axis=0)
            elif a0.size > 7:
                a0 = a0[:7]
            return torch.from_numpy(a0).view(1, 7)

        return torch.zeros(1, 7, dtype=torch.float32)

    def act(self, observation: dict) -> torch.Tensor:
        a0 = self._compute_action_tensor(observation)
        if not isinstance(a0, torch.Tensor):
            a0 = torch.as_tensor(a0, dtype=torch.float32).view(1, -1)
        return a0.float()

    def uses_openpi(self) -> bool:
        return self._openpi_policy is not None
