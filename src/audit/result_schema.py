"""Shared result schema helpers for VLAW audit outputs."""

import json
import os
import time
from pathlib import Path

SCHEMA_VERSION = "1.0"


def build_provenance(
    metric_backend,
    checkpoint,
    config_path,
    task,
    episodes=None,
    extra=None,
):
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_unix": int(time.time()),
        "host": os.uname().nodename,
        "metric_backend": metric_backend,
        "checkpoint": checkpoint,
        "config_path": config_path,
        "task": task,
    }
    if episodes is not None:
        provenance["episodes"] = episodes
    if extra:
        provenance.update(extra)
    return provenance


def write_json(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
