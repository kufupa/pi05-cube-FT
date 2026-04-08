#!/usr/bin/env python3
"""Build a SmolVLA-ready bridge dataset from JEPA-WM trajectory artifacts."""

from __future__ import annotations

import argparse
import io
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Matches jadechoghari/smolvla_metaworld pretrained config.json input_features.
_METAWORLD_ACTION_DIM = 4
_METAWORLD_STATE_DIM = 4
_METAWORLD_ENV_DIM = 39
_METAWORLD_IMAGE_HW = (480, 480)


REQUIRED_FIELDS = ("images", "state", "actions", "language", "done", "success", "action", "action_chunk")

BRIDGE_SUMMARY_FIELD_MAP = {
    "images": "observation.image",
    "state": ["observation.state", "observation.environment_state"],
    "actions/action_chunk": "action",
    "cem_plan.per_step[].step_index": "frame_index",
    "cem_plan.per_step[].policy_source": "not_exported_to_lerobot_columns",
    "cem_plan.per_step[].action_smolvla_chunk": "not_exported_to_lerobot_columns",
    "cem_plan.per_step[].action_wm_cem_plan_seq": "not_exported_to_lerobot_columns",
    "cem_plan.per_step[].latent_pred": "not_exported_to_lerobot_columns",
}


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _env_flag_int(value: Any, default: int = 1) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return 1
    if text in {"0", "false", "no", "n", "off"}:
        return 0
    try:
        return 1 if int(float(text)) != 0 else 0
    except Exception:
        return int(default)


def _resolve_wm_heavy_split_enabled_default() -> int:
    # Planned name takes precedence, legacy alias remains supported.
    raw = os.environ.get("SMOLVLA_BRIDGE_WM_HEAVY_SPLIT")
    if raw is None:
        raw = os.environ.get("SMOLVLA_BRIDGE_WM_HEAVY_SPLIT_ENABLED", "1")
    return _env_flag_int(raw, 1)


def _strict_out_dir_has_existing_content(out_dir: Path) -> bool:
    """Treat empty pre-created train/val roots as reusable; everything else blocks strict mode."""
    if not out_dir.exists():
        return False
    if not out_dir.is_dir():
        return True

    allowed_empty_root_entries = {"train", "val"}
    try:
        root_entries = list(out_dir.iterdir())
    except Exception:
        return True

    for entry in root_entries:
        if entry.name not in allowed_empty_root_entries:
            return True
        if not entry.is_dir():
            return True
        try:
            if any(entry.iterdir()):
                return True
        except Exception:
            return True
    return False


def _read_records(source: Path) -> List[Dict[str, Any]]:
    if not source.exists():
        return []
    if source.is_file():
        return _read_py_records(source)
    records: List[Dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        stem = path.name.lower()
        if (
            suffix not in {".json", ".jsonl", ".npz", ".pth", ".pt", ".pickle", ".pkl"}
            and not stem.endswith(".pth.tar")
        ):
            continue
        try:
            if path.suffix.lower() in {".json", ".jsonl"}:
                records.extend(_read_json_records(path))
            elif path.suffix.lower() == ".npz":
                records.extend(_read_npz_records(path))
            else:
                records.extend(_read_py_records(path))
        except Exception as exc:
            print(f"[bridge] skip {path}: {exc}")
    return records


def _read_records_from_manifest(source: Path) -> List[Dict[str, Any]]:
    """Read trajectory records from export_manifest.json when available."""
    if not source.is_dir():
        return _read_records(source)

    manifest_path = source / "export_manifest.json"
    if not manifest_path.is_file():
        return _read_records(source)

    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[bridge] manifest parse failed ({manifest_path}): {exc}")
        return _read_records(source)

    trajectories_file = manifest_payload.get("trajectories_file")
    if not isinstance(trajectories_file, str) or not trajectories_file.strip():
        return _read_records(source)

    raw_target = Path(trajectories_file.strip()).expanduser()
    target = raw_target if raw_target.is_absolute() else (source / raw_target)
    if target.exists():
        return _read_records(target)

    # Allow concise manifest targets like "episodes" to resolve to known payload files.
    if target.suffix == "":
        ext_candidates = [".json", ".jsonl", ".npz", ".pth", ".pt", ".pickle", ".pkl"]
        for ext in ext_candidates:
            candidate = target.with_suffix(ext)
            if candidate.is_file():
                return _read_records(candidate)
        pth_tar_candidate = target.with_name(f"{target.name}.pth.tar")
        if pth_tar_candidate.is_file():
            return _read_records(pth_tar_candidate)

        parent = target.parent
        if parent.is_dir():
            shard_files = []
            for path in sorted(parent.glob(f"{target.name}*")):
                if path.is_file():
                    suffix = path.suffix.lower()
                    stem = path.name.lower()
                    if suffix in {".json", ".jsonl", ".npz", ".pth", ".pt", ".pickle", ".pkl"} or stem.endswith(
                        ".pth.tar"
                    ):
                        shard_files.append(path)
            if shard_files:
                records: List[Dict[str, Any]] = []
                for shard in shard_files:
                    records.extend(_read_records(shard))
                return records

    return _read_records(target)


def _looks_like_model_checkpoint(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    model_keys = {"predictor", "opt", "scaler", "epoch", "proprio_encoder"}
    if any(key in payload for key in model_keys):
        return True
    state = payload.get("state_dict")
    return isinstance(state, dict) and any(
        isinstance(k, str) and k.startswith("module.")
        for k in state.keys()
    )


def _normalize_record_to_dict(record: Any) -> Dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    return dict(record)


def _read_npz_records(source: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        import numpy as np

        with np.load(source, allow_pickle=True) as npz:
            if "episodes" in npz.files:
                obj = npz["episodes"]
                if isinstance(obj, np.ndarray) and obj.ndim == 1:
                    for item in obj.tolist():
                        normalized = _normalize_record_to_dict(item)
                        if normalized is not None:
                            records.append(normalized)
                else:
                    normalized = _normalize_record_to_dict(npz["episodes"].tolist())
                    if normalized is not None:
                        records.append(normalized)
            else:
                for key in sorted(npz.files):
                    value = npz[key]
                    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] > 0:
                        for item in value.tolist():
                            normalized = _normalize_record_to_dict(item)
                            if normalized is not None:
                                records.append(normalized)
    except Exception as exc:
        print(f"[bridge] skip {source}: {exc}")
    return records


def _read_json_records(source: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        if source.suffix.lower() == ".jsonl":
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    value = json.loads(line)
                    normalized = _normalize_record_to_dict(value)
                    if normalized is not None:
                        records.append(normalized)
            return records

        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for item in payload:
                normalized = _normalize_record_to_dict(item)
                if normalized is not None:
                    records.append(normalized)
        else:
            normalized = _normalize_record_to_dict(payload)
            if normalized is not None:
                records.append(normalized)
    except Exception as exc:
        print(f"[bridge] skip {source}: {exc}")
    return records


def _read_py_records(source: Path) -> List[Dict[str, Any]]:
    if source.suffix.lower() in {".json", ".jsonl"}:
        return _read_json_records(source)
    if source.suffix.lower() == ".npz":
        return _read_npz_records(source)
    if source.suffix.lower() in {".pth", ".pt", ".pickle", ".pkl"}:
        try:
            import torch

            payload = torch.load(source, map_location="cpu")
            if _looks_like_model_checkpoint(payload):
                print(f"[bridge] skipping model checkpoint: {source}")
                return []
            if isinstance(payload, list):
                return [_normalize_record_to_dict(item) for item in payload if _normalize_record_to_dict(item) is not None]
            if isinstance(payload, dict) and "episodes" in payload:
                return [
                    _normalize_record_to_dict(item)
                    for item in payload["episodes"]
                    if _normalize_record_to_dict(item) is not None
                ]
            normalized = _normalize_record_to_dict(payload)
            return [normalized] if normalized is not None else []
        except Exception as exc:
            print(f"[bridge] skip {source}: {exc}")
            return []
    if source.suffix.lower().endswith(".tar") and source.name.lower().endswith(".pth.tar"):
        try:
            import torch

            payload = torch.load(source, map_location="cpu")
            if _looks_like_model_checkpoint(payload):
                print(f"[bridge] skipping model checkpoint: {source}")
                return []
            if isinstance(payload, list):
                return [_normalize_record_to_dict(item) for item in payload if _normalize_record_to_dict(item) is not None]
            if isinstance(payload, dict) and "episodes" in payload:
                return [
                    _normalize_record_to_dict(item)
                    for item in payload["episodes"]
                    if _normalize_record_to_dict(item) is not None
                ]
            normalized = _normalize_record_to_dict(payload)
            return [normalized] if normalized is not None else []
        except Exception as exc:
            print(f"[bridge] skip {source}: {exc}")
            return []
    return []


def _vector_names(size: int) -> List[str]:
    return [f"v{i}" for i in range(max(0, int(size)))]


def _infer_dim(value: Any) -> int:
    items = _coerce_list(value)
    if not items:
        return 0
    first = items[0]
    if isinstance(first, (list, tuple)):
        return len(first)
    return 1


def _write_split_meta_info(split_root: Path, split_name: str, records: List[Dict[str, Any]], source_files: int) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    split_meta_root = split_root / "meta"
    split_meta_root.mkdir(parents=True, exist_ok=True)

    action_dim = _infer_dim(records[0].get("action_chunk", records[0].get("actions", []))) if records else 0
    state_dim = _infer_dim(records[0].get("state", [])) if records else 0
    total_frames = 0
    for item in records:
        total_frames += len(_coerce_list(item.get("images", [])))

    info = {
        "codebase_version": "v2.1",
        "robot_type": "metaworld_push_v3",
        "total_episodes": len(records),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(records),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {split_name: f"0:{len(records)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [max(1, action_dim)],
                "names": _vector_names(action_dim),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [max(1, state_dim)],
                "names": _vector_names(state_dim),
            },
        },
        "source_files": source_files,
        "source_root": str(split_root.parent),
    }
    (split_meta_root / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _write_root_meta_info(out_dir: Path, train: List[Dict[str, Any]], val: List[Dict[str, Any]], source_files: int) -> None:
    root_meta_root = out_dir / "meta"
    root_meta_root.mkdir(parents=True, exist_ok=True)
    all_records = list(train) + list(val)
    total_frames = 0
    for item in all_records:
        total_frames += len(_coerce_list(item.get("images", [])))
    split_map = {}
    if train:
        split_map["train"] = f"0:{len(train)}"
    if val:
        split_map["val"] = f"{len(train)}:{len(train) + len(val)}"

    action_dim = _infer_dim(all_records[0].get("action_chunk", all_records[0].get("actions", []))) if all_records else 0
    state_dim = _infer_dim(all_records[0].get("state", [])) if all_records else 0

    info = {
        "codebase_version": "v2.1",
        "robot_type": "metaworld_push_v3",
        "total_episodes": len(all_records),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(all_records),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": split_map,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [max(1, action_dim)],
                "names": _vector_names(action_dim),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [max(1, state_dim)],
                "names": _vector_names(state_dim),
            },
        },
        "source_files": source_files,
        "source_root": str(out_dir),
    }
    (root_meta_root / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _lerobot_convert_script() -> Path:
    import lerobot

    p = Path(lerobot.__file__).resolve().parent / "scripts" / "convert_dataset_v21_to_v30.py"
    if not p.is_file():
        raise FileNotFoundError(f"Missing LeRobot converter at {p}")
    return p


def _prepare_split_for_lerobot_emit(split_root: Path) -> None:
    shutil.rmtree(split_root / "data", ignore_errors=True)
    shutil.rmtree(split_root / "videos", ignore_errors=True)
    shutil.rmtree(split_root / "images", ignore_errors=True)
    meta = split_root / "meta"
    if meta.is_dir():
        shutil.rmtree(meta)
    split_root.mkdir(parents=True, exist_ok=True)


def _pad_float_vec(values: List[float], dim: int) -> List[float]:
    out = [float(x) for x in values][:dim]
    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    return out


def _align_episode_lists(
    record: Dict[str, Any],
) -> Tuple[List[List[float]], List[List[float]], List[Any]]:
    states = _coerce_list(record.get("state", []))
    actions = _coerce_list(record.get("action_chunk", []))
    images = _coerce_list(record.get("images", []))
    if actions and isinstance(actions[0], (int, float)):
        actions = [actions]
    if states and isinstance(states[0], (int, float)):
        states = [states]
    n = min(len(states), len(actions))
    if n <= 0:
        return [], [], []
    states = states[:n]
    actions = actions[:n]
    if images:
        if len(images) >= n:
            images = images[:n]
        else:
            images = list(images) + [None] * (n - len(images))
    else:
        images = [None] * n
    return states, actions, images


def _frame_image_hwc_uint8(image_slot: Any) -> np.ndarray:
    h, w = _METAWORLD_IMAGE_HW
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    if image_slot is None:
        return blank
    try:
        arr = np.asarray(image_slot)
    except Exception:
        return blank
    if arr.size == 0:
        return blank
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0 + 1e-6:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        pass
    elif arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    else:
        return blank
    # Resize to (h, w) using Pillow if available, else center crop / naive shrink.
    eh, ew = arr.shape[0], arr.shape[1]
    if (eh, ew) == (h, w):
        return np.ascontiguousarray(arr)
    try:
        from PIL import Image

        pil = Image.fromarray(arr, mode="RGB")
        pil = pil.resize((w, h), resample=Image.Resampling.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)
    except Exception:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[: min(eh, h), : min(ew, w), :] = arr[: min(eh, h), : min(ew, w), :]
        return out


def _frame_image_payload(image_hwc_uint8: np.ndarray) -> Dict[str, Any]:
    """Emit HF Image feature payload (`bytes`/`path`) for Arrow struct writes."""
    try:
        from PIL import Image

        with io.BytesIO() as buffer:
            Image.fromarray(image_hwc_uint8).save(buffer, format="PNG")
            return {"bytes": buffer.getvalue(), "path": None}
    except Exception:
        return {"bytes": None, "path": ""}


def _smolvla_metaworld_features_meta() -> Dict[str, Any]:
    h, w = _METAWORLD_IMAGE_HW
    return {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "action": {
            "dtype": "float32",
            "shape": [_METAWORLD_ACTION_DIM],
            "names": _vector_names(_METAWORLD_ACTION_DIM),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [_METAWORLD_STATE_DIM],
            "names": _vector_names(_METAWORLD_STATE_DIM),
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": [_METAWORLD_ENV_DIM],
            "names": _vector_names(_METAWORLD_ENV_DIM),
        },
        "observation.image": {
            "dtype": "image",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
        },
    }


def _vector_stats_json(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        d = int(arr.shape[1]) if arr.ndim == 2 else 0
        z = [0.0] * max(d, 0)
        return {
            "min": z,
            "max": z,
            "mean": z,
            "std": z,
            "q01": z,
            "q10": z,
            "q50": z,
            "q90": z,
            "q99": z,
            "count": [0],
        }
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    m = arr.min(axis=0).astype(np.float64).tolist()
    mx = arr.max(axis=0).astype(np.float64).tolist()
    mu = arr.mean(axis=0).astype(np.float64).tolist()
    sd = arr.std(axis=0).astype(np.float64).tolist()
    return {
        "min": m,
        "max": mx,
        "mean": mu,
        "std": sd,
        "q01": mu,
        "q10": mu,
        "q50": mu,
        "q90": mu,
        "q99": mu,
        "count": [int(arr.shape[0])],
    }


def _image_stats_black_json(n_frames: int) -> Dict[str, Any]:
    zch = [[[0.0]], [[0.0]], [[0.0]]]
    out = {k: zch for k in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")}
    out["count"] = [max(1, int(n_frames))]
    return out


def _write_split_info_v21(
    split_root: Path,
    split_name: str,
    n_episodes: int,
    total_frames: int,
    source_files: int,
    out_parent: Path,
    features: Dict[str, Any],
) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    meta = split_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "metaworld_push_v3",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": n_episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {split_name: f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "source_files": source_files,
        "source_root": str(out_parent),
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_lerobot_v21_smolvla_split(
    split_root: Path,
    split_name: str,
    records: List[Dict[str, Any]],
    source_files: int,
    out_parent: Path,
    task_default: str,
) -> int:
    _prepare_split_for_lerobot_emit(split_root)
    data_dir = split_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    features = _smolvla_metaworld_features_meta()
    task_str = task_default
    if records:
        lang = records[0].get("language", "")
        if isinstance(lang, str) and lang.strip():
            task_str = lang.strip()

    episodes_meta: List[Dict[str, Any]] = []
    stats_meta: List[Dict[str, Any]] = []
    global_index = 0
    total_frames = 0
    ep_write_idx = 0

    for rec in records:
        states, actions, images = _align_episode_lists(rec)
        if not states:
            continue
        ep_idx = ep_write_idx
        ep_write_idx += 1
        t_len = len(states)
        total_frames += t_len
        actions_m = np.zeros((t_len, _METAWORLD_ACTION_DIM), dtype=np.float32)
        state_m = np.zeros((t_len, _METAWORLD_STATE_DIM), dtype=np.float32)
        env_m = np.zeros((t_len, _METAWORLD_ENV_DIM), dtype=np.float32)
        img_col: List[np.ndarray] = []
        rows: List[Dict[str, Any]] = []
        for t in range(t_len):
            raw_s = states[t]
            raw_a = actions[t]
            s_flat = list(map(float, np.asarray(raw_s, dtype=np.float64).reshape(-1).tolist()))
            a_flat = list(map(float, np.asarray(raw_a, dtype=np.float64).reshape(-1).tolist()))
            env_m[t] = np.asarray(_pad_float_vec(s_flat, _METAWORLD_ENV_DIM), dtype=np.float32)
            state_m[t] = np.asarray(_pad_float_vec(s_flat, _METAWORLD_STATE_DIM), dtype=np.float32)
            actions_m[t] = np.asarray(_pad_float_vec(a_flat, _METAWORLD_ACTION_DIM), dtype=np.float32)
            img = _frame_image_hwc_uint8(images[t])
            img_col.append(img)
            rows.append(
                {
                    "index": global_index,
                    "episode_index": ep_idx,
                    "frame_index": t,
                    "timestamp": np.float32(float(t) / 30.0),
                    "task_index": 0,
                    "action": actions_m[t].tolist(),
                    "observation.state": state_m[t].tolist(),
                    "observation.environment_state": env_m[t].tolist(),
                    "observation.image": _frame_image_payload(img),
                }
            )
            global_index += 1

        ep_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        pd.DataFrame(rows).to_parquet(ep_path, index=False)
        episodes_meta.append({"episode_index": ep_idx, "tasks": [task_str], "length": t_len})
        st_action = _vector_stats_json(actions_m)
        st_state = _vector_stats_json(state_m)
        st_env = _vector_stats_json(env_m)
        st_img = _image_stats_black_json(t_len)
        stats_meta.append(
            {
                "episode_index": ep_idx,
                "stats": {
                    "action": st_action,
                    "observation.state": st_state,
                    "observation.environment_state": st_env,
                    "observation.image": st_img,
                },
            }
        )

    meta = split_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    if not episodes_meta:
        _append_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": task_str}])
        _append_jsonl(meta / "episodes.jsonl", [])
        _append_jsonl(meta / "episodes_stats.jsonl", [])
        _write_split_info_v21(
            split_root,
            split_name,
            0,
            0,
            source_files,
            out_parent,
            features,
        )
        return 0

    _append_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": task_str}])
    _append_jsonl(meta / "episodes.jsonl", episodes_meta)
    _append_jsonl(meta / "episodes_stats.jsonl", stats_meta)
    _write_split_info_v21(
        split_root,
        split_name,
        len(episodes_meta),
        total_frames,
        source_files,
        out_parent,
        features,
    )
    return total_frames


def _run_lerobot_v21_to_v30(split_root: Path) -> None:
    script = _lerobot_convert_script()
    cmd = [
        sys.executable,
        str(script),
        "--repo-id=local/bridge_dataset",
        "--root",
        str(split_root.resolve()),
        "--push-to-hub=false",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stdout + "\n" + proc.stderr
        raise RuntimeError(f"LeRobot v2.1→v3.0 conversion failed for {split_root}:\n{msg}")


def _split_has_emitted_episodes(split_root: Path) -> bool:
    data_dir = split_root / "data" / "chunk-000"
    if not data_dir.is_dir():
        return False
    return any(data_dir.glob("episode_*.parquet"))


def _write_root_meta_smolvla(
    out_dir: Path,
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    source_files: int,
) -> None:
    features = _smolvla_metaworld_features_meta()
    total_frames = 0
    for item in train + val:
        st, ac, _ = _align_episode_lists(item)
        total_frames += min(len(st), len(ac))
    split_map: Dict[str, str] = {}
    if train:
        split_map["train"] = f"0:{len(train)}"
    if val:
        split_map["val"] = f"{len(train)}:{len(train) + len(val)}"
    root_meta = out_dir / "meta"
    root_meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "metaworld_push_v3",
        "total_episodes": len(train) + len(val),
        "total_frames": total_frames,
        "total_tasks": 1,
        "splits": split_map,
        "fps": 30,
        "features": features,
        "source_files": source_files,
        "source_root": str(out_dir),
        "note": "Root meta is informational; train/ and val/ hold LeRobotDataset roots after v3 conversion.",
    }
    (root_meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _normalize(record: Dict[str, Any]) -> Dict[str, Any]:
    wm_score = _compute_wm_completeness_score(record)
    out = {
        "images": record.get("images", []),
        "state": record.get("state", []),
        "action_chunk": record.get("actions", record.get("action", record.get("action_chunk", []))),
        "language": record.get("language", ""),
        "done": bool(record.get("done", False)),
        "success": bool(record.get("success", False)),
        "wm_completeness_score": wm_score,
        "meta": {k: v for k, v in record.items() if k not in REQUIRED_FIELDS},
    }
    confidence = _safe_float(record.get("confidence", record.get("meta", {}).get("confidence", 1.0)), 1.0)
    out["meta"]["confidence"] = confidence
    out["meta"]["wm_completeness_score"] = wm_score
    return out


def _read_manifest_export_mode(source: Path) -> str | None:
    """If source is a dir with export_manifest.json, return export_mode string."""
    if not source.is_dir():
        return None
    manifest = source / "export_manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        mode = data.get("export_mode")
        return str(mode) if mode else None
    except Exception:
        return None


def _pair_key_hash(item: Dict[str, Any]) -> int:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    pk = meta.get("pair_key") or item.get("pair_key") or ""
    h = hashlib.sha256(str(pk).encode("utf-8")).hexdigest()[:12]
    return int(h, 16) % 1000


def _extract_per_step_telemetry(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = item.get("cem_plan")
    if isinstance(plan, dict):
        per_step = plan.get("per_step")
        if isinstance(per_step, list):
            return [step for step in per_step if isinstance(step, dict)]

    meta = item.get("meta")
    if isinstance(meta, dict):
        meta_plan = meta.get("cem_plan")
        if isinstance(meta_plan, dict):
            per_step = meta_plan.get("per_step")
            if isinstance(per_step, list):
                return [step for step in per_step if isinstance(step, dict)]
    return []


def _step_has_cem_plan(step: Dict[str, Any]) -> bool:
    if _safe_float(step.get("cem_iterations", 0), 0.0) > 0.0:
        return True
    seq = _coerce_list(step.get("action_wm_cem_plan_seq", []))
    if len(seq) > 0:
        return True
    planner_metadata = step.get("planner_metadata")
    if isinstance(planner_metadata, dict):
        if "horizon" in planner_metadata or "population" in planner_metadata:
            return True
    return False


def _step_has_latent(step: Dict[str, Any]) -> bool:
    latents = _coerce_list(step.get("latent_pred", []))
    return len(latents) > 0


def _step_wm_success(step: Dict[str, Any]) -> bool:
    planner_metadata = step.get("planner_metadata")
    if not isinstance(planner_metadata, dict):
        planner_metadata = {}
    if planner_metadata.get("wm_skipped"):
        return False
    if planner_metadata.get("wm_step_error"):
        return False
    src = str(step.get("policy_source", "")).strip().lower()
    if src == "cem_mpc_wm":
        return True
    return _step_has_cem_plan(step) or _step_has_latent(step)


def _compute_wm_completeness_score(item: Dict[str, Any]) -> float:
    per_step = _extract_per_step_telemetry(item)
    if not per_step:
        return 0.0

    total = 0.0
    for step in per_step:
        wm_success = 1.0 if _step_wm_success(step) else 0.0
        latent_present = 1.0 if _step_has_latent(step) else 0.0
        plan_present = 1.0 if _step_has_cem_plan(step) else 0.0
        total += (wm_success + latent_present + plan_present) / 3.0
    return float(total / len(per_step))


def _wm_completeness_from_item(item: Dict[str, Any]) -> float:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    if "wm_completeness_score" in meta:
        return _safe_float(meta.get("wm_completeness_score"), 0.0)
    if "wm_completeness_score" in item:
        return _safe_float(item.get("wm_completeness_score"), 0.0)
    return _compute_wm_completeness_score(item)


def _wm_split_tiebreak(item: Dict[str, Any], fallback_index: int) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    pair_key = meta.get("pair_key") or item.get("pair_key") or ""
    language = str(item.get("language", ""))
    steps = len(_coerce_list(item.get("action_chunk", item.get("actions", []))))
    digest_src = f"{pair_key}|{language}|{steps}|{fallback_index}"
    return hashlib.sha256(digest_src.encode("utf-8")).hexdigest()


def _mean_wm_completeness(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    scores = [_wm_completeness_from_item(item) for item in records]
    return float(sum(scores) / len(scores))


def _split_wm_heavy(
    records: List[Dict[str, Any]],
    val_ratio: float,
    score_margin: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    if not records:
        return [], [], {
            "train_count": 0,
            "val_count": 0,
            "mean_score_train": 0.0,
            "mean_score_val": 0.0,
            "val_fraction": 0.0,
            "score_margin": float(max(0.0, score_margin)),
        }

    ratio = 0.0 if val_ratio < 0.0 else 1.0 if val_ratio > 1.0 else float(val_ratio)
    margin = float(max(0.0, score_margin))
    ranked: List[Dict[str, Any]] = []
    for idx, item in enumerate(records):
        ranked.append(
            {
                "idx": idx,
                "item": item,
                "score": _wm_completeness_from_item(item),
                "tiebreak": _wm_split_tiebreak(item, idx),
            }
        )
    ranked.sort(key=lambda row: (-row["score"], row["tiebreak"], row["idx"]))

    if ratio <= 0.0:
        train = [row["item"] for row in ranked]
        val: List[Dict[str, Any]] = []
    elif ratio >= 1.0:
        train = []
        val = [row["item"] for row in ranked]
    else:
        n_total = len(ranked)
        n_val = max(1, int(n_total * ratio))
        if n_total > 1:
            n_val = min(n_val, n_total - 1)
        cutoff_score = ranked[n_val - 1]["score"]

        hard_val_idx = {row["idx"] for row in ranked if row["score"] > cutoff_score + margin}
        hard_train_idx = {row["idx"] for row in ranked if row["score"] < cutoff_score - margin}
        border = [
            row
            for row in ranked
            if row["idx"] not in hard_val_idx and row["idx"] not in hard_train_idx
        ]
        needed = max(0, n_val - len(hard_val_idx))
        selected_val_idx = set(hard_val_idx)
        selected_val_idx.update(row["idx"] for row in border[:needed])

        val = [row["item"] for row in ranked if row["idx"] in selected_val_idx]
        train = [row["item"] for row in ranked if row["idx"] not in selected_val_idx]

    stats = {
        "train_count": len(train),
        "val_count": len(val),
        "mean_score_train": _mean_wm_completeness(train),
        "mean_score_val": _mean_wm_completeness(val),
        "val_fraction": ratio,
        "score_margin": margin,
    }
    return train, val, stats


def _split_by_pair_key_hash(
    records: List[Dict[str, Any]], val_ratio: float
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic train/val split from pair_key (cem_paired_push_v3)."""
    if not records:
        return [], []
    thr = max(0, min(1000, int(round(val_ratio * 1000))))
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for item in records:
        if _pair_key_hash(item) < thr:
            val.append(item)
        else:
            train.append(item)
    if thr > 0 and not val and len(train) > 1:
        val.append(train.pop())
    if thr < 1000 and not train and len(val) > 1:
        train.append(val.pop())
    return train, val


def _split(records: List[Dict[str, Any]], val_ratio: float) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        return [], []
    records = list(records)
    random.shuffle(records)
    ratio = 0.0 if val_ratio < 0.0 else 1.0 if val_ratio > 1.0 else val_ratio
    n_total = len(records)
    if ratio == 0.0:
        return records, []
    if ratio == 1.0:
        return [], records
    n_val = max(1, int(n_total * ratio))
    n_val = min(n_val, n_total - 1)
    if n_val <= 0:
        return records, []
    return records[:-n_val], records[-n_val:]


def _write_placeholder_dataset(out_dir: Path) -> None:
    (out_dir / "train" / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "val" / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    _write_split_manifest_preview(out_dir / "train" / "manifest.json", [])
    _write_split_manifest_preview(out_dir / "val" / "manifest.json", [])
    _write_split_meta_info(out_dir / "train", "train", [], 0)
    _write_split_meta_info(out_dir / "val", "val", [], 0)
    (out_dir / "bridge_summary.json").write_text(
        json.dumps(
            {
                "train_records": 0,
                "val_records": 0,
                "filtered_confidence": 0,
                "filtered_empty": 0,
                "source_files": 0,
                "empty_inputs": True,
                "split_policy": {
                    "name": "wm_heavy_top_fraction",
                    "wm_heavy_enabled": True,
                    "wm_heavy_jepa_fraction": 0.6,
                    "wm_heavy_val_fraction": 0.6,
                    "wm_score_margin": 0.0,
                    "wm_heavy_score_margin": 0.0,
                    "deterministic_tiebreaker": "sha256(pair_key|language|steps|index)",
                },
                "split_counts": {"train": 0, "val": 0},
                "wm_completeness_mean": {"train": 0.0, "val": 0.0},
                "wm_heavy_split_policy": "wm_heavy_top_fraction",
                "mean_wm_score_train": 0.0,
                "mean_wm_score_val": 0.0,
                "n_train_episodes": 0,
                "n_val_episodes": 0,
                "quality_metrics": {
                    "image_nonblank_ratio": 0.0,
                    "heuristic_fallback_episode_ratio": 0.0,
                    "action_std_mean": 0.0,
                },
                "field_map_exporter_to_lerobot": BRIDGE_SUMMARY_FIELD_MAP,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_root_meta_info(out_dir, [], [], 0)


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    return list(value) if hasattr(value, "__iter__") else [value]


def _write_split_manifest_preview(path: Path, records: List[Dict[str, Any]]) -> None:
    sample_pair_keys: List[str] = []
    sample_step_counts: List[int] = []
    for item in records[:10]:
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        pair_key = meta.get("pair_key") or item.get("pair_key") or ""
        sample_pair_keys.append(str(pair_key))
        steps = _coerce_list(item.get("action_chunk", item.get("actions", [])))
        if steps and isinstance(steps[0], (int, float)):
            steps = [steps]
        sample_step_counts.append(int(len(steps)))
    preview = {
        "record_count": int(len(records)),
        "sample_pair_keys": sample_pair_keys,
        "sample_step_counts": sample_step_counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preview, indent=2), encoding="utf-8")


def _episode_policy_sources(item: Dict[str, Any]) -> List[str]:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    plan = meta.get("cem_plan") if isinstance(meta, dict) else {}
    per_step = plan.get("per_step") if isinstance(plan, dict) else []
    sources: List[str] = []
    for step in _coerce_list(per_step):
        if not isinstance(step, dict):
            continue
        src = step.get("policy_source")
        if isinstance(src, str) and src.strip():
            sources.append(src.strip())
    if sources:
        return sources
    nested_meta = meta.get("meta") if isinstance(meta, dict) else {}
    fallback_src = nested_meta.get("policy") if isinstance(nested_meta, dict) else None
    if isinstance(fallback_src, str) and fallback_src.strip():
        return [fallback_src.strip()]
    return []


def _is_nonblank_numeric_image(image: Any) -> bool:
    if image is None:
        return False
    try:
        arr = np.asarray(image)
    except Exception:
        return False
    if arr.size <= 0:
        return False
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return False
    try:
        finite = np.isfinite(arr)
    except Exception:
        return False
    if not bool(np.any(finite)):
        return False
    safe = np.where(finite, arr, 0)
    return bool(np.any(safe != 0))


def _compute_quality_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    total_frames = 0
    nonblank_frames = 0
    total_episodes = len(records)
    heuristic_episodes = 0
    flat_actions: List[float] = []

    for item in records:
        images = _coerce_list(item.get("images", []))
        for image in images:
            total_frames += 1
            if _is_nonblank_numeric_image(image):
                nonblank_frames += 1

        sources = _episode_policy_sources(item)
        if any(src in {"heuristic_fallback", "heuristic"} for src in sources):
            heuristic_episodes += 1

        action_steps = _coerce_list(item.get("action_chunk", item.get("actions", [])))
        if action_steps and isinstance(action_steps[0], (int, float)):
            action_steps = [action_steps]
        for step in action_steps:
            try:
                vec = np.asarray(step, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if vec.size == 0:
                continue
            flat_actions.extend(vec.tolist())

    action_std_mean = float(np.std(np.asarray(flat_actions, dtype=np.float32))) if flat_actions else 0.0
    image_nonblank_ratio = float(nonblank_frames / total_frames) if total_frames > 0 else 0.0
    heuristic_ratio = float(heuristic_episodes / total_episodes) if total_episodes > 0 else 0.0
    return {
        "image_nonblank_ratio": image_nonblank_ratio,
        "heuristic_fallback_episode_ratio": heuristic_ratio,
        "action_std_mean": action_std_mean,
    }


def _enforce_quality_gates(
    metrics: Dict[str, float],
    *,
    min_image_coverage: float,
    max_heuristic_ratio: float,
    min_action_std: float,
) -> None:
    image_cov = float(metrics.get("image_nonblank_ratio", 0.0))
    heuristic_ratio = float(metrics.get("heuristic_fallback_episode_ratio", 0.0))
    action_std = float(metrics.get("action_std_mean", 0.0))
    if image_cov < min_image_coverage:
        raise RuntimeError(
            f"image_nonblank_ratio below threshold: {image_cov:.4f} < {min_image_coverage:.4f}"
        )
    if heuristic_ratio > max_heuristic_ratio:
        raise RuntimeError(
            "heuristic_fallback_episode_ratio above threshold: "
            f"{heuristic_ratio:.4f} > {max_heuristic_ratio:.4f}"
        )
    if action_std < min_action_std:
        raise RuntimeError(f"action_std_mean below threshold: {action_std:.4f} < {min_action_std:.4f}")


def _is_valid_record(item: Dict[str, Any], min_actions: int) -> bool:
    if min_actions > 0:
        return len(_coerce_list(item.get("action_chunk", []))) >= min_actions
    return len(_coerce_list(item.get("images", []))) > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    wm_split_enabled_default = _resolve_wm_heavy_split_enabled_default()
    wm_split_fraction_default = _safe_float(
        os.environ.get(
            "SMOLVLA_BRIDGE_WM_HEAVY_JEPA_FRACTION",
            os.environ.get(
                "SMOLVLA_BRIDGE_WM_HEAVY_VAL_FRACTION",
                os.environ.get("SMOLVLA_BRIDGE_VAL_RATIO", "0.6"),
            ),
        ),
        0.6,
    )
    wm_split_margin_default = _safe_float(
        os.environ.get(
            "SMOLVLA_BRIDGE_WM_SCORE_MARGIN",
            os.environ.get("SMOLVLA_BRIDGE_WM_HEAVY_SCORE_MARGIN", "0.0"),
        ),
        0.0,
    )
    parser.add_argument("--jepa-source", default="", help="JEPA trajectory export path")
    parser.add_argument("--out-dir", required=True, help="Output dataset root")
    parser.add_argument(
        "--fail-on-path-reuse",
        type=int,
        default=_env_flag_int(os.environ.get("SMOLVLA_FAIL_ON_PATH_REUSE", "0"), 0),
        help="When 1, fail if --out-dir already contains non-empty bridge content.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--wm-heavy-split-enabled",
        type=int,
        default=wm_split_enabled_default,
        help="When 1, split by wm_completeness_score top fraction into val (JEPA) root.",
    )
    parser.add_argument(
        "--wm-heavy-val-fraction",
        type=float,
        default=wm_split_fraction_default,
        help="Top fraction (by wm_completeness_score) assigned to val split when WM-heavy policy is enabled.",
    )
    parser.add_argument(
        "--wm-heavy-score-margin",
        type=float,
        default=wm_split_margin_default,
        help="Boundary margin around WM-heavy split cutoff score before deterministic tie-break resolution.",
    )
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--min-action-length", type=int, default=1)
    parser.add_argument("--min-image-coverage", type=float, default=0.95)
    parser.add_argument("--max-heuristic-ratio", type=float, default=0.10)
    parser.add_argument("--min-action-std", type=float, default=0.02)
    parser.add_argument(
        "--no-convert-v30",
        action="store_true",
        help="Keep v2.1 on-disk layout (debug only; lerobot-train expects v3.0).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    strict_fail_on_reuse = bool(_env_flag_int(args.fail_on_path_reuse, 0))
    if strict_fail_on_reuse and _strict_out_dir_has_existing_content(out_dir):
        print(f"[bridge] strict mode: refusing non-empty out-dir reuse: {out_dir}")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    if args.jepa_source:
        source = Path(args.jepa_source).expanduser().resolve()
        if source.is_file():
            records.extend(_read_py_records(source))
        elif source.is_dir():
            records.extend(_read_records_from_manifest(source))

    if not records:
        print("[bridge] no source files found; writing empty placeholders")
        _write_placeholder_dataset(out_dir)
        return 0

    def _passes_confidence(item: Dict[str, Any]) -> bool:
        conf = item.get("meta", {}).get("confidence", item.get("confidence", 1.0))
        return _safe_float(conf, 1.0) >= args.min_confidence

    filtered = [_normalize(item) for item in records if _passes_confidence(item)]
    filtered_conf = len(filtered)
    filtered = [item for item in filtered if _is_valid_record(item, args.min_action_length)]
    filtered_empty = len(filtered)
    quality_metrics = _compute_quality_metrics(filtered)
    _enforce_quality_gates(
        quality_metrics,
        min_image_coverage=float(args.min_image_coverage),
        max_heuristic_ratio=float(args.max_heuristic_ratio),
        min_action_std=float(args.min_action_std),
    )

    if args.val_ratio < 0:
        val_ratio = 0.0
    elif args.val_ratio > 1:
        val_ratio = 1.0
    else:
        val_ratio = args.val_ratio
    wm_heavy_split_enabled = bool(int(args.wm_heavy_split_enabled))
    wm_heavy_val_fraction = (
        0.0
        if args.wm_heavy_val_fraction < 0
        else 1.0
        if args.wm_heavy_val_fraction > 1
        else float(args.wm_heavy_val_fraction)
    )
    wm_heavy_score_margin = max(0.0, float(args.wm_heavy_score_margin))

    manifest_mode = None
    if args.jepa_source:
        manifest_mode = _read_manifest_export_mode(Path(args.jepa_source).expanduser().resolve())
    wm_split_stats = {
        "train_count": 0,
        "val_count": 0,
        "mean_score_train": 0.0,
        "mean_score_val": 0.0,
        "val_fraction": wm_heavy_val_fraction if wm_heavy_split_enabled else val_ratio,
        "score_margin": wm_heavy_score_margin if wm_heavy_split_enabled else 0.0,
    }
    if wm_heavy_split_enabled:
        train, val, wm_split_stats = _split_wm_heavy(
            filtered,
            val_ratio=wm_heavy_val_fraction,
            score_margin=wm_heavy_score_margin,
        )
    elif manifest_mode == "cem_paired_push_v3":
        train, val = _split_by_pair_key_hash(filtered, val_ratio)
    else:
        train, val = _split(filtered, val_ratio)

    if args.max_train > 0:
        cap = min(args.max_train, len(filtered))
        effective_val_ratio = wm_heavy_val_fraction if wm_heavy_split_enabled else val_ratio
        train = train[:cap]
        val = val[: max(1, int(len(filtered) * effective_val_ratio))]
    train = [item for item in train if _is_valid_record(item, args.min_action_length)]
    val = [item for item in val if _is_valid_record(item, args.min_action_length)]
    wm_split_stats["train_count"] = len(train)
    wm_split_stats["val_count"] = len(val)
    wm_split_stats["mean_score_train"] = _mean_wm_completeness(train)
    wm_split_stats["mean_score_val"] = _mean_wm_completeness(val)

    # Do not rebalance after deterministic pair_key split for cem_paired exports.
    if args.train_ratio < 1.0 and manifest_mode != "cem_paired_push_v3" and not wm_heavy_split_enabled:
        if len(train) > 0 and len(val) == 0:
            val_size = max(1, math.floor(len(train) * (1.0 - args.train_ratio)))
            move_n = min(len(train), val_size)
            val = train[-move_n:]
            train = train[:-move_n]

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    _write_split_manifest_preview(out_dir / "train" / "manifest.json", train)
    _write_split_manifest_preview(out_dir / "val" / "manifest.json", val)
    task_default = "push the puck to the goal"
    try:
        _write_lerobot_v21_smolvla_split(
            out_dir / "train", "train", train, len(records), out_dir, task_default
        )
        _write_lerobot_v21_smolvla_split(
            out_dir / "val", "val", val, len(records), out_dir, task_default
        )
    except Exception as exc:
        print(f"[bridge] LeRobot v2.1 emit failed: {exc}")
        return 1
    if not args.no_convert_v30:
        try:
            if _split_has_emitted_episodes(out_dir / "train"):
                _run_lerobot_v21_to_v30(out_dir / "train")
            if _split_has_emitted_episodes(out_dir / "val"):
                _run_lerobot_v21_to_v30(out_dir / "val")
        except Exception as exc:
            print(f"[bridge] {exc}")
            return 1
    _write_root_meta_smolvla(out_dir, train, val, len(records))
    wm_split_stats["train_count"] = len(train)
    wm_split_stats["val_count"] = len(val)
    wm_split_stats["mean_score_train"] = _mean_wm_completeness(train)
    wm_split_stats["mean_score_val"] = _mean_wm_completeness(val)
    if wm_heavy_split_enabled:
        split_method = "wm_heavy_top_fraction"
    else:
        split_method = "pair_key_hash" if manifest_mode == "cem_paired_push_v3" else "shuffle"
    summary = {
        "train_records": len(train),
        "val_records": len(val),
        "filtered_confidence": filtered_conf,
        "filtered_empty": filtered_empty,
        "source_files": len(records),
        "empty_inputs": False,
        "out_dir": str(out_dir),
        "min_confidence": args.min_confidence,
        "min_action_length": args.min_action_length,
        "split_method": split_method,
        "split_policy": {
            "name": "wm_heavy_top_fraction" if wm_heavy_split_enabled else split_method,
            "wm_heavy_enabled": wm_heavy_split_enabled,
            "wm_heavy_jepa_fraction": float(wm_heavy_val_fraction),
            "wm_heavy_val_fraction": float(wm_heavy_val_fraction),
            "wm_score_margin": float(wm_heavy_score_margin),
            "wm_heavy_score_margin": float(wm_heavy_score_margin),
            "deterministic_tiebreaker": "sha256(pair_key|language|steps|index)",
        },
        "split_counts": {
            "train": int(wm_split_stats.get("train_count", len(train))),
            "val": int(wm_split_stats.get("val_count", len(val))),
        },
        "wm_completeness_mean": {
            "train": float(wm_split_stats.get("mean_score_train", 0.0)),
            "val": float(wm_split_stats.get("mean_score_val", 0.0)),
        },
        "wm_heavy_split_policy": "wm_heavy_top_fraction" if wm_heavy_split_enabled else split_method,
        "mean_wm_score_train": float(wm_split_stats.get("mean_score_train", 0.0)),
        "mean_wm_score_val": float(wm_split_stats.get("mean_score_val", 0.0)),
        "n_train_episodes": int(wm_split_stats.get("train_count", len(train))),
        "n_val_episodes": int(wm_split_stats.get("val_count", len(val))),
        "manifest_export_mode": manifest_mode,
        "lerobot_layout": "smolvla_metaworld_jadechoghari",
        "lerobot_codebase": "v3.0" if not args.no_convert_v30 else "v2.1",
        "quality_metrics": quality_metrics,
        "quality_thresholds": {
            "min_image_coverage": float(args.min_image_coverage),
            "max_heuristic_ratio": float(args.max_heuristic_ratio),
            "min_action_std": float(args.min_action_std),
        },
        "field_map_exporter_to_lerobot": BRIDGE_SUMMARY_FIELD_MAP,
    }
    (out_dir / "bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[bridge] train={len(train)} val={len(val)} output={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

