"""Build OpenPI UR5e policy requests from OGBench observations."""

from __future__ import annotations

import numpy as np


def _to_hwc_uint8(image_chw) -> np.ndarray:
    import torch

    if isinstance(image_chw, torch.Tensor):
        arr = image_chw.detach().cpu().float()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"image tensor must be CHW or BCHW, got shape={tuple(arr.shape)}")
        arr = arr.clamp(0.0, 1.0).numpy()
    else:
        arr = np.asarray(image_chw)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"image array must be CHW or BCHW, got shape={arr.shape}")
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)

    if arr.shape[0] == 3:
        hwc = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] == 3:
        hwc = arr
    else:
        raise ValueError(f"image must have 3 channels, got shape={arr.shape}")

    if np.issubdtype(hwc.dtype, np.floating):
        hwc = (hwc * 255.0).clip(0, 255).astype(np.uint8)
    else:
        hwc = np.asarray(hwc, dtype=np.uint8)
    return np.ascontiguousarray(hwc)


def build_openpi_ur5e_request_from_tensors(
    image_chw,
    joints_6,
    gripper_open_01: float,
    instruction: str,
) -> dict:
    """Build flat UR5e request dict expected by Pi05UR5ePolicy transforms."""
    j = np.asarray(joints_6, dtype=np.float32).reshape(-1)
    if j.size != 6:
        raise ValueError(f"joints_6 must have 6 values, got {j.size}")

    base_rgb = _to_hwc_uint8(image_chw)
    # OGBench has no wrist camera; keep this slot deterministic.
    wrist_rgb = np.zeros_like(base_rgb, dtype=np.uint8)

    g_open = float(np.clip(gripper_open_01, 0.0, 1.0))
    g_model = np.array([1.0 - g_open], dtype=np.float32)
    prompt = (instruction or "").strip() or "do something"

    return {
        "base_rgb": base_rgb,
        "wrist_rgb": wrist_rgb,
        "joints": j.astype(np.float32),
        "gripper": g_model,
        "prompt": prompt,
    }
