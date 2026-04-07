#!/usr/bin/env python3
"""Merge two LeRobot v2.1-style local roots (meta/info.json + episode_*.parquet) into one trainable root.

Real + JEPA (e.g. train/ + val/) are concatenated with renumbered episode indices. Requires pyarrow.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def _load_meta(root: Path) -> dict[str, Any] | None:
    p = root / "meta" / "info.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _collect_parquets(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("episode_*.parquet") if p.is_file())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _parse_episode_index(path: Path) -> int | None:
    m = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _episode_meta_map(root: Path, filename: str) -> dict[int, dict[str, Any]]:
    rows = _load_jsonl(root / "meta" / filename)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            idx = int(row.get("episode_index"))
        except Exception:
            continue
        out[idx] = row
    return out


def _load_tasks_rows(root: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(root / "meta" / "tasks.jsonl")
    if rows:
        return rows
    return [{"task_index": 0, "task": "push the puck to the goal"}]


def _row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _copy_episode_with_reindex(src: Path, dst: Path, *, episode_index: int, index_offset: int) -> tuple[int, int]:
    """Copy one episode parquet while rewriting key index columns to merged-global values."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(src)
    frame_count = int(table.num_rows)

    def _replace_or_append_col(name: str, values: list[int]) -> None:
        nonlocal table
        arr = pa.array(values, type=pa.int64())
        col_idx = table.schema.get_field_index(name)
        if col_idx == -1:
            table = table.append_column(name, arr)
            return
        table = table.set_column(col_idx, name, arr)

    _replace_or_append_col("episode_index", [episode_index] * frame_count)
    _replace_or_append_col("frame_index", list(range(frame_count)))
    _replace_or_append_col("index", list(range(index_offset, index_offset + frame_count)))

    pq.write_table(table, dst)
    return frame_count, index_offset + frame_count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", type=Path, required=True)
    ap.add_argument("--jepa-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    real = args.real_root.resolve()
    jepa = args.jepa_root.resolve()
    out = args.out.resolve()

    real_eps = _collect_parquets(real)
    jepa_eps = _collect_parquets(jepa)
    if not real_eps and not jepa_eps:
        print("[merge_lerobot] no episode_*.parquet under real or jepa roots; cannot build mixed dataset")
        return 1
    if not real_eps:
        print(f"[merge_lerobot] warn: no episode_*.parquet under --real-root {real} (mix is JEPA-only)")
    if not jepa_eps:
        print(f"[merge_lerobot] warn: no episode_*.parquet under --jepa-root {jepa} (mix is real-only)")

    meta = _load_meta(real) or _load_meta(jepa)
    if not meta:
        print("[merge_lerobot] missing meta/info.json under both roots")
        return 1

    tasks_rows = _load_tasks_rows(real) if _load_tasks_rows(real) else _load_tasks_rows(jepa)
    if not tasks_rows:
        tasks_rows = [{"task_index": 0, "task": "push the puck to the goal"}]
    task_values = [str(row.get("task", "")).strip() for row in tasks_rows if str(row.get("task", "")).strip()]
    fallback_task = task_values[0] if task_values else "push the puck to the goal"
    fallback_tasks = [fallback_task]

    episodes_meta_by_root = {
        str(real): _episode_meta_map(real, "episodes.jsonl"),
        str(jepa): _episode_meta_map(jepa, "episodes.jsonl"),
    }
    stats_meta_by_root = {
        str(real): _episode_meta_map(real, "episodes_stats.jsonl"),
        str(jepa): _episode_meta_map(jepa, "episodes_stats.jsonl"),
    }

    data_dir = out / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    idx = 0
    episodes_rows_out: list[dict[str, Any]] = []
    stats_rows_out: list[dict[str, Any]] = []
    for src in real_eps + jepa_eps:
        src_root = real if real in src.parents else jepa
        src_key = str(src_root)
        src_ep_idx = _parse_episode_index(src)
        dst = data_dir / f"episode_{idx:06d}.parquet"
        frame_count = 0
        try:
            frame_count, total_frames = _copy_episode_with_reindex(
                src,
                dst,
                episode_index=idx,
                index_offset=total_frames,
            )
        except Exception as exc:
            # Preserve prior behavior if pyarrow mutation fails for any reason.
            shutil.copy2(src, dst)
            try:
                frame_count = _row_count(dst)
                total_frames += frame_count
            except Exception:
                frame_count = 0
            print(f"[merge_lerobot] warn: fallback copy without reindex for {src}: {exc}")
        src_ep = episodes_meta_by_root.get(src_key, {}).get(src_ep_idx or -1, {})
        src_tasks = src_ep.get("tasks", fallback_tasks)
        if not isinstance(src_tasks, list) or not src_tasks:
            src_tasks = fallback_tasks
        src_len = int(src_ep.get("length", frame_count if frame_count > 0 else 0))
        episodes_rows_out.append(
            {
                "episode_index": idx,
                "tasks": src_tasks,
                "length": src_len,
            }
        )
        src_stats = stats_meta_by_root.get(src_key, {}).get(src_ep_idx or -1, {})
        stats_payload = src_stats.get("stats") if isinstance(src_stats, dict) else None
        if not isinstance(stats_payload, dict):
            stats_payload = {}
        stats_rows_out.append({"episode_index": idx, "stats": stats_payload})
        idx += 1

    _write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)
    _write_jsonl(meta_dir / "episodes.jsonl", episodes_rows_out)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", stats_rows_out)

    meta = dict(meta)
    meta["total_episodes"] = idx
    meta["total_frames"] = total_frames
    meta["source_root"] = str(out)
    meta["splits"] = {"train": f"0:{idx}"}
    meta["total_chunks"] = 1
    (out / "meta" / "info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[merge_lerobot] wrote {idx} episodes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
