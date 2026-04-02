#!/usr/bin/env python3
"""Preflight checks for local-first pi0.5 LoRA pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _check_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, "ok"
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Source trimmed run directory")
    parser.add_argument("--run-root", type=Path, required=True, help="Target local_build/<run_tag> directory")
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    checks: dict[str, dict] = {}
    failed = False

    required_imports = [
        "jax",
        "flax",
        "torch",
        "openpi",
        "imageio_ffmpeg",
        "certifi",
    ]
    for mod in required_imports:
        ok, msg = _check_import(mod)
        checks[f"import:{mod}"] = {"ok": ok, "message": msg}
        failed = failed or (not ok)

    # Runtime and environment checks.
    try:
        import certifi

        cert_path = certifi.where()
        checks["certifi_path"] = {"ok": bool(cert_path), "value": cert_path}
        if cert_path:
            os.environ.setdefault("SSL_CERT_FILE", cert_path)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
    except Exception as exc:  # pragma: no cover
        checks["certifi_path"] = {"ok": False, "message": str(exc)}
        failed = True

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        checks["ffmpeg_exe"] = {"ok": Path(ffmpeg_exe).exists(), "value": ffmpeg_exe}
        failed = failed or (not Path(ffmpeg_exe).exists())
    except Exception as exc:  # pragma: no cover
        checks["ffmpeg_exe"] = {"ok": False, "message": str(exc)}
        failed = True

    try:
        import jax

        devices = [str(d) for d in jax.devices()]
        checks["jax_devices"] = {"ok": len(devices) >= 1, "count": len(devices), "devices": devices}
        failed = failed or (len(devices) < 1)
    except Exception as exc:  # pragma: no cover
        checks["jax_devices"] = {"ok": False, "message": str(exc)}
        failed = True

    # Input/output path checks.
    checks["run_dir_exists"] = {"ok": run_dir.exists(), "path": str(run_dir)}
    checks["run_dir_videos"] = {"ok": (run_dir / "videos").exists(), "path": str(run_dir / "videos")}
    checks["run_dir_metadata"] = {"ok": (run_dir / "metadata").exists(), "path": str(run_dir / "metadata")}
    checks["run_dir_manifest"] = {"ok": (run_dir / "manifest").exists(), "path": str(run_dir / "manifest")}
    checks["run_ready_file"] = {
        "ok": (run_dir / "RUN_READY_FOR_TRAINING.md").exists(),
        "path": str(run_dir / "RUN_READY_FOR_TRAINING.md"),
    }
    for k in ("run_dir_exists", "run_dir_videos", "run_dir_metadata", "run_dir_manifest", "run_ready_file"):
        failed = failed or (not checks[k]["ok"])

    free_gb = _disk_free_gb(run_root)
    checks["disk_free"] = {"ok": free_gb >= args.min_free_gb, "free_gb": free_gb, "min_free_gb": args.min_free_gb}
    failed = failed or (free_gb < args.min_free_gb)

    # OpenPI train entrypoint check.
    ok_train, msg_train = _check_import("openpi.training.config")
    checks["openpi_training_config_import"] = {"ok": ok_train, "message": msg_train}
    failed = failed or (not ok_train)

    report = {
        "status": "fail" if failed else "pass",
        "run_dir": str(run_dir),
        "run_root": str(run_root),
        "checks": checks,
    }
    # Stage L0 ledger bootstrap.
    project_root = Path(__file__).resolve().parents[2]
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root), text=True).strip()
    except Exception:
        pass
    manifest = {
        "source_run_dir": str(run_dir),
        "run_root": str(run_root),
        "git_sha": git_sha,
        "env": {
            "HOSTNAME": os.environ.get("HOSTNAME"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SSL_CERT_FILE": os.environ.get("SSL_CERT_FILE"),
            "REQUESTS_CA_BUNDLE": os.environ.get("REQUESTS_CA_BUNDLE"),
        },
        "command_templates": {
            "preflight": "check_preflight.py --run-dir <trimmed_run> --run-root <local_build/run_tag>",
            "validate": "validate_trimmed_run.py --run-dir <trimmed_run> --run-root <local_build/run_tag>",
            "convert": "convert_run_to_lerobot_cube.py --run-dir <trimmed_run> --run-root <local_build/run_tag> --source-npz <npz>",
            "train": "train_pi05_lora_cube.py --run-root <local_build/run_tag> --repo-id <repo> --exp-name <exp>",
        },
    }
    out = run_root / "preflight_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if failed:
        print("Preflight failed. See preflight_report.json", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

