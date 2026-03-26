"""
Pi0.5-DROID Policy wrapper: OpenPI inference when available, else gripper heuristic.
act() always returns [1, 8] (joint velocity-style 7 + gripper 1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.envs.droid import get_droid_dataset


def _make_identity_linear_8() -> nn.Linear:
    layer = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(8))
    return layer


class Pi05DroidPolicy:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self._openpi_policy = None
        self.action_adapter: nn.Linear | None = _make_identity_linear_8()
        self._openpi_failed_reason: str | None = None

        if checkpoint_path is None or str(checkpoint_path).lower().startswith("heuristic"):
            print(f"Pi05DroidPolicy: heuristic mode (no OpenPI). ckpt={checkpoint_path}")
            return

        try:
            from openpi.policies import policy_config
            from openpi.shared import download
            from openpi.training import config as openpi_train_config

            path_str = str(checkpoint_path)
            if path_str.startswith("gs://openpi-assets/checkpoints/"):
                config_name = path_str.rstrip("/").split("/")[-1]
                ckpt_dir = download.maybe_download(path_str)
            else:
                config_name = "pi05_droid"
                ckpt_dir = path_str

            cfg = openpi_train_config.get_config(config_name)
            self._openpi_policy = policy_config.create_trained_policy(cfg, ckpt_dir)
            print(f"Pi05DroidPolicy: OpenPI loaded config={config_name!r} dir={ckpt_dir!r}")
        except Exception as exc:
            self._openpi_failed_reason = str(exc)
            print(f"Pi05DroidPolicy: OpenPI load failed ({exc}); using heuristic act().")

    @classmethod
    def load_policy(cls, checkpoint_path):
        return cls(checkpoint_path)

    def _compute_action_tensor(self, observation: dict) -> torch.Tensor:
        """Policy output [1, 8] before action_adapter."""
        if self._openpi_policy is not None:
            req = self._build_openpi_request(observation)
            out = self._openpi_policy.infer(req)
            actions = np.asarray(out["actions"])
            a0 = torch.from_numpy(actions[0].astype(np.float32)).view(1, -1)
            if a0.shape[-1] < 8:
                pad = torch.zeros(1, 8 - a0.shape[-1])
                a0 = torch.cat([a0, pad], dim=-1)
            elif a0.shape[-1] > 8:
                a0 = a0[:, :8]
            return a0

        st = observation.get("state")
        if isinstance(st, torch.Tensor) and st.numel() >= 1:
            flat = st.detach().float().view(-1)
            grip = flat[-1].view(1, 1) if flat.numel() >= 1 else torch.zeros(1, 1)
            vel = torch.zeros(1, 7)
            return torch.cat([vel, grip], dim=-1)
        return torch.zeros(1, 8)

    def act(self, observation: dict) -> torch.Tensor:
        """Return first action row [1, 8] (adapter applied)."""
        a0 = self._compute_action_tensor(observation)
        if not isinstance(a0, torch.Tensor):
            a0 = torch.as_tensor(a0, dtype=torch.float32).view(1, -1)
        a0 = a0.float()
        if self.action_adapter is not None:
            dev = self.action_adapter.weight.device
            a0 = a0.to(dev)
            a0 = self.action_adapter(a0)
        return a0

    def _build_openpi_request(self, observation: dict) -> dict:
        from src.envs.droid.observation_openpi import (
            build_openpi_droid_request,
            build_openpi_droid_request_from_tensors,
        )

        ep = observation.get("droid_episode_ref")
        t = int(observation.get("timestep", 0))
        inst = observation.get("instruction", "") or ""
        if ep is not None:
            return build_openpi_droid_request(ep, t, inst)
        obs = observation.get("obs")
        st = observation.get("state")
        if obs is None:
            raise ValueError("observation must include 'obs' or 'droid_episode_ref' for OpenPI")
        if st is None:
            st = torch.zeros(1, 14)
        return build_openpi_droid_request_from_tensors(obs, st, inst)

    def uses_openpi(self) -> bool:
        return self._openpi_policy is not None

    # ------------------------------------------------------------------
    # Success heuristic (Track 1 evaluation)
    # ------------------------------------------------------------------
    def _episode_success(self, episode: dict, task_name: str) -> float:
        gt_success = float(episode.get("success", 0.0))
        actions = episode["actions"]
        state_seq = episode.get("state", None)

        if state_seq is not None and state_seq[:, -1].abs().max() > 0.01:
            gripper = state_seq[:, -1]
            closed_thresh, open_thresh = 0.02, 0.08
            gripper_closed = gripper < closed_thresh
            gripper_open = gripper > open_thresh
        else:
            gripper = actions[:, -1]
            gripper_closed = gripper < 0.2
            gripper_open = gripper > 0.7

        activity = (gripper.diff().abs() > 0.01).float().mean().item() if len(gripper) > 1 else 0.0

        if task_name == "stacking":
            closed_steps = gripper_closed.nonzero(as_tuple=False)
            if len(closed_steps) > 0:
                first_close = closed_steps[0, 0].item()
                opened_after = gripper_open[first_close:].any().item()
                gripper_score = 1.0 if opened_after else 0.5
            else:
                gripper_score = 0.0
        else:
            gripper_score = min(activity * 2.0, 1.0)

        return 0.6 * gt_success + 0.4 * gripper_score

    def evaluate(self, task_name: str, n_episodes: int = 10, dataset=None) -> float:
        print(f"Evaluating π0.5-DROID [{task_name}] over {n_episodes} episodes (gripper heuristic)...")

        if dataset is None:
            episodes = get_droid_dataset(task_name=task_name, max_episodes=n_episodes)
        else:
            episodes = dataset[:n_episodes]

        if not episodes:
            return 0.0

        scores = [self._episode_success(ep, task_name) for ep in episodes]
        mean = float(np.mean(scores))
        print(
            f"  Success rate: {mean:.4f}  ({len(scores)} episodes, min={min(scores):.2f}, max={max(scores):.2f})"
        )
        return mean
