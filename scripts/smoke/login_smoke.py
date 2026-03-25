#!/usr/bin/env python3
"""Low-RAM login-node smoke test for control path integrity."""

import argparse
import ast
import json
import time
from pathlib import Path

import yaml

from src.vla.pi05_droid import Pi05DroidPolicy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/droid_single_task_vlaw.yaml")
    parser.add_argument("--output", default="logs/login_smoke.json")
    parser.add_argument("--max-seconds", type=int, default=60)
    args = parser.parse_args()

    t0 = time.time()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["vlaw"]

    # Keep this test cheap: instantiate policy only, static-check heavy wrappers.
    policy = Pi05DroidPolicy(cfg["base_policy_ckpt"])
    wm_text = Path("src/world_model/models.py").read_text(encoding="utf-8")
    rm_text = Path("src/reward_model/models.py").read_text(encoding="utf-8")
    wm_tree = ast.parse(wm_text)
    rm_tree = ast.parse(rm_text)
    wm_has_class = any(isinstance(node, ast.ClassDef) and node.name == "CtrlWorldModel" for node in ast.walk(wm_tree))
    rm_has_class = any(isinstance(node, ast.ClassDef) and node.name == "QwenRewardModel" for node in ast.walk(rm_tree))

    report = {
        "status": "pass",
        "config": args.config,
        "world_model_wrapper": "CtrlWorldModel",
        "reward_model_wrapper": "QwenRewardModel",
        "policy_wrapper": policy.__class__.__name__,
        "world_model_class_found": wm_has_class,
        "reward_model_class_found": rm_has_class,
        "elapsed_s": round(time.time() - t0, 3),
    }

    if not wm_has_class or not rm_has_class:
        report["status"] = "fail"
        report["failure"] = "wrapper_classes_missing"

    if report["elapsed_s"] > args.max_seconds:
        report["status"] = "fail"
        report["failure"] = f"runtime_exceeded:{args.max_seconds}s"

    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out_path)}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
