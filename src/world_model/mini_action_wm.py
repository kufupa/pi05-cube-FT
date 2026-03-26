"""Lightweight action-conditioned next-frame predictor (MSE reconstruction)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionCondFramePredictor(nn.Module):
    """Predict next RGB frame from current frame + 8-D action; residual on input."""

    def __init__(self, action_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.fuse = nn.Conv2d(3 + action_dim, hidden, kernel_size=3, padding=1)
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=3, padding=1),
        )

    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        frame: [B, 3, H, W] in [0, 1]
        action: [B, action_dim]
        """
        b, _, h, w = frame.shape
        a = action.view(b, -1, 1, 1).expand(b, action.shape[-1], h, w)
        x = torch.cat([frame, a], dim=1)
        delta = self.net(self.fuse(x))
        return torch.sigmoid(frame + delta)
