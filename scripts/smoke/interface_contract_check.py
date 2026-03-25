#!/usr/bin/env python3
"""Contract checks for VLAW wrappers (mock/backend swap safety)."""

import argparse
import ast
import json
from pathlib import Path

import torch

from src.vla.pi05_droid import Pi05DroidPolicy


def _fail(msg):
    raise AssertionError(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/interface_contract.json")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    checks: dict[str, bool] = {}
    details: dict[str, str] = {}
    ok = True

    try:
        policy = Pi05DroidPolicy("gs://openpi-assets/checkpoints/pi05_droid")
        act = policy.act({"obs": torch.randn(1, 3, 256, 256)})
        if not isinstance(act, torch.Tensor):
            _fail("policy.act must return torch.Tensor")
        if tuple(act.shape) != (1, 7):
            _fail(f"policy.act shape expected (1,7), got {tuple(act.shape)}")
        checks["policy.act_shape"] = True
    except Exception as exc:
        ok = False
        checks["policy.act_shape"] = False
        details["policy.act_shape"] = str(exc)

    # Keep these checks static to avoid expensive model initialization on login node.
    try:
        wm_path = Path("src/world_model/models.py")
        wm_tree = ast.parse(wm_path.read_text(encoding="utf-8"))
        wm_has_rollout = False
        wm_has_expected_keys = False
        for node in ast.walk(wm_tree):
            if isinstance(node, ast.FunctionDef) and node.name == "rollout":
                wm_has_rollout = True
        text = wm_path.read_text(encoding="utf-8")
        wm_has_expected_keys = all(k in text for k in ["\"observation\"", "\"action\"", "\"dreamt_image\""])
        if not wm_has_rollout or not wm_has_expected_keys:
            _fail("world model rollout contract symbols missing")
        checks["world_model.rollout_contract"] = True
    except Exception as exc:
        ok = False
        checks["world_model.rollout_contract"] = False
        details["world_model.rollout_contract"] = str(exc)

    try:
        rm_path = Path("src/reward_model/models.py")
        rm_tree = ast.parse(rm_path.read_text(encoding="utf-8"))
        rm_has_score = any(isinstance(node, ast.FunctionDef) and node.name == "score" for node in ast.walk(rm_tree))
        rm_text = rm_path.read_text(encoding="utf-8")
        rm_mentions_shape = "torch.rand(B, 1)" in rm_text or "unsqueeze(-1)" in rm_text
        if not rm_has_score or not rm_mentions_shape:
            _fail("reward model score contract symbols missing")
        checks["reward_model.score_contract"] = True
    except Exception as exc:
        ok = False
        checks["reward_model.score_contract"] = False
        details["reward_model.score_contract"] = str(exc)

    report = {
        "status": "pass" if ok else "fail",
        "checks": checks,
        "details": details,
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(out)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
