#!/usr/bin/env python3
"""Verify OpenPI DROID policy outputs 8-D actions. Requires GPU/OpenPI env."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
openpi_src = Path(__file__).resolve().parents[1] / "external" / "openpi" / "src"
if openpi_src.is_dir():
    sys.path.insert(0, str(openpi_src))


def main() -> int:
    try:
        from openpi.policies import policy_config
        from openpi.policies.droid_policy import make_droid_example
        from openpi.shared import download
        from openpi.training import config as openpi_train_config
    except Exception as exc:
        print(f"FATAL: OpenPI import failed: {exc}", file=sys.stderr)
        return 1

    config_name = "pi05_droid"
    ckpt_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{config_name}")
    cfg = openpi_train_config.get_config(config_name)
    policy = policy_config.create_trained_policy(cfg, ckpt_dir)

    ex = make_droid_example()
    out = policy.infer(ex)
    actions = __import__("numpy").asarray(out["actions"])
    if actions.shape[-1] != 8:
        print(f"FATAL: expected actions[...,8], got {actions.shape}", file=sys.stderr)
        return 2
    if not __import__("numpy").isfinite(actions).all():
        print("FATAL: non-finite actions", file=sys.stderr)
        return 3
    print("test_openpi_policy: OK (8-D actions, finite).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
