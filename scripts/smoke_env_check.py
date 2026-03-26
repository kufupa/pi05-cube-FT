#!/usr/bin/env python3
"""
Pre-PBS import verification. Exit 0 only if all required modules import.
Run from project root: uv run python scripts/smoke_env_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_paths() -> None:
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    openpi_src = root / "external" / "openpi" / "src"
    if openpi_src.is_dir() and str(openpi_src) not in sys.path:
        sys.path.insert(0, str(openpi_src))


def _try(name: str, fn) -> bool:
    try:
        fn()
        print(f"[ok] {name}")
        return True
    except Exception as exc:
        print(f"[fail] {name}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    _ensure_paths()
    root = _root()
    os.chdir(root)

    checks = [
        ("torch", lambda: __import__("torch")),
        ("transformers", lambda: __import__("transformers")),
        ("diffusers", lambda: __import__("diffusers")),
        ("bitsandbytes", lambda: __import__("bitsandbytes")),
        ("tensorflow_datasets", lambda: __import__("tensorflow_datasets")),
        ("openpi", lambda: __import__("openpi")),
    ]

    ok = all(_try(n, f) for n, f in checks)
    if not ok:
        print(
            "\nFix: cd "
            + str(root)
            + " && uv sync && uv pip install bitsandbytes tensorflow tensorflow-datasets",
            file=sys.stderr,
        )
        return 1
    print("\nAll smoke_env_check imports passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
