#!/usr/bin/env python3
"""Create a trimmed copy of a cube run with videos capped to N seconds.

Behavior:
- Input: latest run under run-root by default, or explicit --run-dir.
- Output: sibling dir named run_copy_trimmed_<input_run_name> by default.
- Copies all non-video files/directories from source run.
- Trims only videos/*.mp4 to the first --seconds.
- Writes trim_report.json with counts, timing, and any fallback/failures.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_run_root() -> Path:
    return _project_root() / "cube_dataset" / "finetune_runs" / "pi05_cube_single_v1" / "data_gen"


def _resolve_ffmpeg() -> Path:
    try:
        import imageio_ffmpeg
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio_ffmpeg is required for bundled ffmpeg lookup") from exc
    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not exe.exists():
        raise FileNotFoundError(f"ffmpeg executable not found: {exe}")
    return exe


def _find_latest_run(run_root: Path) -> Path:
    if not run_root.exists():
        raise FileNotFoundError(f"run-root not found: {run_root}")
    candidates = [
        p
        for p in run_root.iterdir()
        if p.is_dir() and p.name.startswith("run_") and not p.name.startswith("run_copy_trimmed_")
    ]
    if not candidates:
        raise RuntimeError(f"No source run directories found under: {run_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _default_output_dir(src_run: Path) -> Path:
    return src_run.parent / f"run_copy_trimmed_{src_run.name}"


def _run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = (p.stderr or "").strip()
    return p.returncode, stderr


def _trim_stream_copy(ffmpeg: Path, src: Path, dst: Path, seconds: float) -> tuple[bool, str]:
    cmd = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-t",
        f"{seconds:.3f}",
        "-map",
        "0",
        "-c",
        "copy",
        str(dst),
    ]
    code, err = _run_cmd(cmd)
    return code == 0, err


def _trim_reencode(ffmpeg: Path, src: Path, dst: Path, seconds: float) -> tuple[bool, str]:
    cmd = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-t",
        f"{seconds:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    code, err = _run_cmd(cmd)
    return code == 0, err


def _validate_playable(ffmpeg: Path, video: Path) -> tuple[bool, str]:
    cmd = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(video),
        "-f",
        "null",
        "-",
    ]
    code, err = _run_cmd(cmd)
    return code == 0, err


def _copy_run_non_video(src_run: Path, dst_run: Path, overwrite: bool) -> None:
    if dst_run.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination exists: {dst_run}. Use --overwrite to replace it."
            )
        shutil.rmtree(dst_run)
    dst_run.mkdir(parents=True, exist_ok=True)

    src_videos = src_run / "videos"
    for root, dirs, files in os.walk(src_run):
        root_path = Path(root)
        rel = root_path.relative_to(src_run)
        dst_root = dst_run / rel
        dst_root.mkdir(parents=True, exist_ok=True)

        for d in dirs:
            (dst_root / d).mkdir(parents=True, exist_ok=True)

        for f in files:
            src_file = root_path / f
            rel_file = src_file.relative_to(src_run)
            # Skip top-level videos/*.mp4 so we can replace with trimmed outputs.
            if src_file.parent == src_videos and src_file.suffix.lower() == ".mp4":
                continue
            shutil.copy2(src_file, dst_root / f)


@dataclass
class TrimStats:
    total: int = 0
    stream_copy_ok: int = 0
    fallback_reencode_ok: int = 0
    failed: int = 0
    validated_ok: int = 0
    validated_failed: int = 0


def trim_run(
    src_run: Path,
    dst_run: Path,
    seconds: float,
    overwrite: bool,
) -> dict:
    ffmpeg = _resolve_ffmpeg()
    started = time.time()

    _copy_run_non_video(src_run, dst_run, overwrite=overwrite)

    src_videos = src_run / "videos"
    if not src_videos.exists():
        raise FileNotFoundError(f"Missing source videos dir: {src_videos}")
    dst_videos = dst_run / "videos"
    dst_videos.mkdir(parents=True, exist_ok=True)

    src_mp4s = sorted(src_videos.glob("*.mp4"))
    stats = TrimStats(total=len(src_mp4s))
    failures: list[dict] = []

    for idx, src_mp4 in enumerate(src_mp4s, start=1):
        dst_mp4 = dst_videos / src_mp4.name
        print(f"[{idx:03d}/{len(src_mp4s):03d}] trim {src_mp4.name}", flush=True)

        ok, err = _trim_stream_copy(ffmpeg, src_mp4, dst_mp4, seconds=seconds)
        if ok:
            stats.stream_copy_ok += 1
        else:
            ok2, err2 = _trim_reencode(ffmpeg, src_mp4, dst_mp4, seconds=seconds)
            if ok2:
                stats.fallback_reencode_ok += 1
            else:
                stats.failed += 1
                failures.append(
                    {
                        "file": src_mp4.name,
                        "stream_copy_error": err,
                        "fallback_reencode_error": err2,
                    }
                )
                continue

        if not dst_mp4.exists() or dst_mp4.stat().st_size <= 0:
            stats.failed += 1
            failures.append(
                {"file": src_mp4.name, "error": "trimmed file missing or empty"}
            )
            continue

        playable, perr = _validate_playable(ffmpeg, dst_mp4)
        if playable:
            stats.validated_ok += 1
        else:
            stats.validated_failed += 1
            failures.append({"file": src_mp4.name, "error": f"validate failed: {perr}"})

    # Count parity check.
    dst_mp4s = sorted(dst_videos.glob("*.mp4"))
    if len(dst_mp4s) != len(src_mp4s):
        failures.append(
            {
                "error": "video count mismatch",
                "source_count": len(src_mp4s),
                "dest_count": len(dst_mp4s),
            }
        )

    ended = time.time()
    report = {
        "source_run": str(src_run.resolve()),
        "dest_run": str(dst_run.resolve()),
        "seconds": seconds,
        "ffmpeg": str(ffmpeg),
        "started_unix": started,
        "ended_unix": ended,
        "elapsed_sec": ended - started,
        "counts": {
            "source_videos": len(src_mp4s),
            "dest_videos": len(dst_mp4s),
            "stream_copy_ok": stats.stream_copy_ok,
            "fallback_reencode_ok": stats.fallback_reencode_ok,
            "failed": stats.failed,
            "validated_ok": stats.validated_ok,
            "validated_failed": stats.validated_failed,
        },
        "failures": failures,
    }
    (dst_run / "trim_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit source run directory. If omitted, latest run under --run-root is used.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Explicit destination run directory. Default: sibling run_copy_trimmed_<run_name>.",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_run = args.run_dir.resolve() if args.run_dir else _find_latest_run(args.run_root.resolve())
    dst_run = args.out_dir.resolve() if args.out_dir else _default_output_dir(src_run)

    print(f"source run: {src_run}")
    print(f"dest run  : {dst_run}")
    print(f"trim sec  : {args.seconds}")

    report = trim_run(
        src_run=src_run,
        dst_run=dst_run,
        seconds=float(args.seconds),
        overwrite=bool(args.overwrite),
    )
    failed = int(report["counts"]["failed"]) + int(report["counts"]["validated_failed"])
    if failed > 0 or report["counts"]["source_videos"] != report["counts"]["dest_videos"]:
        print("DONE WITH ERRORS. See trim_report.json for details.", file=sys.stderr)
        raise SystemExit(2)
    print("DONE OK.")


if __name__ == "__main__":
    main()

