#!/usr/bin/env python3
"""Build a SmolVLA-ready bridge dataset from JEPA-WM trajectory artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List


REQUIRED_FIELDS = ("images", "state", "actions", "language", "done", "success", "action", "action_chunk")


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_records(source: Path) -> List[Dict[str, Any]]:
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


def _normalize(record: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "images": record.get("images", []),
        "state": record.get("state", []),
        "action_chunk": record.get("actions", record.get("action", record.get("action_chunk", []))),
        "language": record.get("language", ""),
        "done": bool(record.get("done", False)),
        "success": bool(record.get("success", False)),
        "meta": {k: v for k, v in record.items() if k not in REQUIRED_FIELDS},
    }
    confidence = _safe_float(record.get("confidence", record.get("meta", {}).get("confidence", 1.0)), 1.0)
    out["meta"]["confidence"] = confidence
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
    (out_dir / "train" / "manifest.json").write_text("[]\n", encoding="utf-8")
    (out_dir / "val" / "manifest.json").write_text("[]\n", encoding="utf-8")
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


def _is_valid_record(item: Dict[str, Any], min_actions: int) -> bool:
    if min_actions > 0:
        return len(_coerce_list(item.get("action_chunk", []))) >= min_actions
    return len(_coerce_list(item.get("images", []))) > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-source", default="", help="JEPA trajectory export path")
    parser.add_argument("--out-dir", required=True, help="Output dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--min-action-length", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    if args.jepa_source:
        source = Path(args.jepa_source).expanduser().resolve()
        if source.is_file():
            records.extend(_read_py_records(source))
        elif source.is_dir():
            records.extend(_read_records(source))

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

    if args.val_ratio < 0:
        val_ratio = 0.0
    elif args.val_ratio > 1:
        val_ratio = 1.0
    else:
        val_ratio = args.val_ratio

    manifest_mode = None
    if args.jepa_source:
        manifest_mode = _read_manifest_export_mode(Path(args.jepa_source).expanduser().resolve())
    if manifest_mode == "cem_paired_push_v3":
        train, val = _split_by_pair_key_hash(filtered, val_ratio)
    else:
        train, val = _split(filtered, val_ratio)

    if args.max_train > 0:
        cap = min(args.max_train, len(filtered))
        train = train[:cap]
        val = val[: max(1, int(len(filtered) * val_ratio))]
    train = [item for item in train if _is_valid_record(item, args.min_action_length)]
    val = [item for item in val if _is_valid_record(item, args.min_action_length)]

    # Do not rebalance after deterministic pair_key split for cem_paired exports.
    if args.train_ratio < 1.0 and manifest_mode != "cem_paired_push_v3":
        if len(train) > 0 and len(val) == 0:
            val_size = max(1, math.floor(len(train) * (1.0 - args.train_ratio)))
            move_n = min(len(train), val_size)
            val = train[-move_n:]
            train = train[:-move_n]

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "train" / "manifest.json").write_text(
        json.dumps(train, indent=2), encoding="utf-8"
    )
    (out_dir / "val" / "manifest.json").write_text(
        json.dumps(val, indent=2), encoding="utf-8"
    )
    _write_split_meta_info(out_dir / "train", "train", train, len(records))
    _write_split_meta_info(out_dir / "val", "val", val, len(records))
    _write_root_meta_info(out_dir, train, val, len(records))
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
        "manifest_export_mode": manifest_mode,
    }
    (out_dir / "bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[bridge] train={len(train)} val={len(val)} output={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

