"""World-model datasets."""
from __future__ import annotations

import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset

from src.envs.droid import get_droid_dataset


class LiberoWorldModelDataset(Dataset):
    """LIBERO index stub (legacy)."""

    def __init__(self, data_root, index_json):
        super().__init__()
        self.data_root = data_root
        self.index_path = index_json
        self.episodes = []
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self.episodes = json.load(f)

    def __len__(self):
        return max(1, len(self.episodes))

    def __getitem__(self, idx):
        history_frames = torch.randn(3, 8, 256, 256)
        actions = torch.randn(8, 7)
        future_frames = torch.randn(3, 8, 256, 256)
        return history_frames, actions, future_frames


def get_dataloader(data_root, index_json, batch_size=8):
    dataset = LiberoWorldModelDataset(data_root, index_json)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class DroidWorldModelDataset(Dataset):
    """(frame_t, action_t, frame_{t+1}) tuples from DROID TFRecords."""

    def __init__(self, task_name: str = "stacking", max_episodes: int = 50, seed: int = 0):
        super().__init__()
        self._episodes = get_droid_dataset(task_name=task_name, max_episodes=max_episodes)
        self._pairs: list[tuple[int, int]] = []
        rng = random.Random(seed)
        for ei, ep in enumerate(self._episodes):
            obs = ep["obs"]
            t_max = obs.shape[0] - 1
            if t_max < 1:
                continue
            for t in range(t_max):
                self._pairs.append((ei, t))
        rng.shuffle(self._pairs)

    def __len__(self) -> int:
        return max(1, len(self._pairs))

    def __getitem__(self, idx: int):
        if not self._pairs:
            return (
                torch.zeros(3, 256, 256),
                torch.zeros(8),
                torch.zeros(3, 256, 256),
            )
        ei, t = self._pairs[idx % len(self._pairs)]
        ep = self._episodes[ei]
        frame_t = ep["obs"][t].clamp(0, 1)
        frame_tp = ep["obs"][t + 1].clamp(0, 1)
        if "expert_actions_8" in ep and ep["expert_actions_8"].shape[0] > t:
            act = ep["expert_actions_8"][t].float()
        else:
            a7 = ep["actions"][t].float()
            g = ep["state"][t, -1:].float()
            act = torch.cat([a7, g], dim=-1)
        return frame_t, act, frame_tp


def get_droid_wm_dataloader(
    task_name: str = "stacking",
    max_episodes: int = 50,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    ds = DroidWorldModelDataset(task_name=task_name, max_episodes=max_episodes, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
