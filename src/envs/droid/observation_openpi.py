"""Build OpenPI DROID policy requests (flat observation/* keys, 224×224 pad-resize)."""

from __future__ import annotations

import numpy as np
import cv2

try:
    from openpi_client import image_tools as _openpi_image_tools
except ImportError:
    _openpi_image_tools = None


def _fallback_resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """PIL pad-resize when openpi_client is unavailable (matches openpi_client.image_tools)."""
    from PIL import Image

    if image.shape[-3:-1] == (height, width):
        return image
    original_shape = image.shape
    batch = image.reshape(-1, *original_shape[-3:])
    out = []
    for im in batch:
        pil_im = Image.fromarray(im)
        cur_w, cur_h = pil_im.size
        if cur_w == width and cur_h == height:
            out.append(np.asarray(pil_im))
            continue
        ratio = max(cur_w / width, cur_h / height)
        rh, rw = int(cur_h / ratio), int(cur_w / ratio)
        resized = pil_im.resize((rw, rh), resample=Image.BILINEAR)
        canvas = Image.new(resized.mode, (width, height), 0)
        pad_h = max(0, (height - rh) // 2)
        pad_w = max(0, (width - rw) // 2)
        canvas.paste(resized, (pad_w, pad_h))
        out.append(np.asarray(canvas))
    stacked = np.stack(out, axis=0)
    return stacked.reshape(*original_shape[:-3], *stacked.shape[-3:])


def resize_with_pad_hwc_uint8(img_hwc: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    if _openpi_image_tools is not None:
        return _openpi_image_tools.resize_with_pad(img_hwc, height, width)
    return _fallback_resize_with_pad(img_hwc, height, width)


def decode_jpeg_bytes_to_rgb_uint8(img_bytes) -> np.ndarray:
    if isinstance(img_bytes, np.ndarray):
        if img_bytes.dtype == object:
            img_bytes = img_bytes.item()
        elif img_bytes.shape == ():
            img_bytes = img_bytes.item()
    if isinstance(img_bytes, np.bytes_):
        img_bytes = bytes(img_bytes)
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def build_openpi_droid_request_from_tensors(image_chw, state_14, instruction: str) -> dict:
    """
    Build OpenPI flat request from a single CHW float image in [0,1] and 14-D robot state.
    Used by world-model rollouts where only tensor observations are available.
    """
    import torch

    if isinstance(image_chw, torch.Tensor):
        t = image_chw.detach().cpu().float()
        if t.dim() == 4:
            t = t[0]
        chw = t.clamp(0, 1)
        hwc = (chw.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        hwc = np.transpose(hwc, (1, 2, 0))
    else:
        arr = np.asarray(image_chw, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] == 3:
            hwc = (np.transpose(arr, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            hwc = np.zeros((256, 256, 3), dtype=np.uint8)

    if isinstance(state_14, torch.Tensor):
        s = state_14.detach().cpu().float().view(-1).numpy()
    else:
        s = np.asarray(state_14, dtype=np.float32).reshape(-1)
    if s.size < 14:
        pad = np.zeros(14, dtype=np.float32)
        pad[: s.size] = s
        s = pad
    jp = s[:7].astype(np.float32)
    gp = np.array([float(s[-1])], dtype=np.float32)

    base224 = resize_with_pad_hwc_uint8(hwc, 224, 224)
    wrist224 = np.zeros((224, 224, 3), dtype=np.uint8)
    prompt = (instruction or "").strip() or "do something"

    return {
        "observation/exterior_image_1_left": base224,
        "observation/wrist_image_left": wrist224,
        "observation/joint_position": jp,
        "observation/gripper_position": gp,
        "prompt": prompt,
    }


def build_openpi_droid_request(ep: dict, t: int, instruction: str) -> dict:
    """
    Observation keys and tensor layout match external/openpi/examples/droid/main.py.
    """
    ext_seq = ep["exterior_jpeg"]
    img_bytes = ext_seq[t]
    base_rgb = decode_jpeg_bytes_to_rgb_uint8(img_bytes)

    wrist_seq = ep.get("wrist_jpeg")
    if wrist_seq is None:
        wrist_rgb = np.zeros_like(base_rgb)
    else:
        wrist_rgb = decode_jpeg_bytes_to_rgb_uint8(wrist_seq[t])

    base224 = resize_with_pad_hwc_uint8(base_rgb, 224, 224)
    wrist224 = resize_with_pad_hwc_uint8(wrist_rgb, 224, 224)

    jp = ep["joint_position"][t].detach().cpu().numpy().astype(np.float32)
    gp = ep["gripper_position"][t].detach().cpu().numpy().astype(np.float32).reshape(-1)
    if gp.size == 0:
        gp = np.zeros((1,), dtype=np.float32)
    elif gp.size > 1:
        gp = gp[:1]

    prompt = (instruction or "").strip() or "do something"

    return {
        "observation/exterior_image_1_left": base224,
        "observation/wrist_image_left": wrist224,
        "observation/joint_position": jp,
        "observation/gripper_position": gp,
        "prompt": prompt,
    }
