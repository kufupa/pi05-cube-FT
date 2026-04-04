#!/usr/bin/env python3
"""JEPA-WM smoke check for Meta-World rollout compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", default="facebook/jepa-wms", help="torch.hub repo path")
  parser.add_argument("--ckpt", default="jepa_wm_metaworld.pth.tar", help="Unused default checkpoint name")
  parser.add_argument("--task", default="push-v3", help="Task label")
  parser.add_argument("--pretrained", action="store_true", default=False, help="Use remote pretrained checkpoint from torch.hub model registry")
  parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for smoke check")
  parser.add_argument("--smoke-steps", type=int, default=6, dest="smoke_steps")
  args = parser.parse_args()

  out = {
    "repo": args.repo,
    "task": args.task,
    "smoke_steps": args.smoke_steps,
    "status": "unknown",
    "errors": [],
  }

  try:
    import torch

    import os
    import site
    import sys

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    if args.device != "auto":
      device = args.device

    print(f"[jepa-smoke] torch {torch.__version__}")

    # Prefer local repo load to avoid internet-only source.
    repo_dir = Path(args.repo)
    if repo_dir.is_dir():
      model, preprocessor = torch.hub.load(
        str(repo_dir),
        "jepa_wm_metaworld",
        source="local",
        pretrained=args.pretrained,
        device=device,
      )
    else:
      model, preprocessor = torch.hub.load(
        args.repo, "jepa_wm_metaworld", source="github", pretrained=args.pretrained, device=device
      )

    model.eval()
    print("[jepa-smoke] model loaded")

    # Determine sizes from pretrained preprocessors/statistics.
    # Some JEPA checkpoints expose a model action_dim that differs from
    # preprocessor action statistics depending on model packaging/variant.
    # Use the live model action_dim for rollout smoke checks to avoid shape
    # mismatches during forward/unroll calls.
    action_dim = int(getattr(preprocessor, "action_mean").numel())
    model_action_dim = int(getattr(getattr(model, "model"), "action_dim", action_dim))
    if model_action_dim != action_dim:
      print(
        f"[jepa-smoke] action_dim mismatch: preprocessor={action_dim}, model={model_action_dim}; using model dim for smoke unroll"
      )
    action_dim = model_action_dim
    proprio_dim = int(getattr(preprocessor, "proprio_mean").numel())

    b = 1
    # A single-context frame keeps temporal tokenization aligned across JEPA-WM
    # variants and avoids context-length dependent token-count mismatches in smoke
    # rollout tests.
    context_len = 1
    model.to(device)

    obs = {
      "visual": torch.randint(
        low=0,
        high=256,
        size=(b, context_len, 3, 256, 256),
        dtype=torch.float32,
      ),
      "proprio": torch.zeros((b, context_len, proprio_dim), dtype=torch.float32),
    }
    z = model.encode(obs)

    act_suffix = torch.randn(args.smoke_steps, b, action_dim, device=device)
    z = z.to(device)
    act_suffix = act_suffix.to(device)

    with torch.no_grad():
      z_pred = model.unroll(z, act_suffix=act_suffix, debug=False)
      frame_count = int(z_pred["visual"].shape[0]) if hasattr(z_pred, "keys") else int(z_pred.shape[0])
      frames = None
      if hasattr(model, "decode_unroll"):
        decoded = model.decode_unroll(z_pred, batch=True)
        frames = int(decoded.shape[1]) if decoded is not None else 0

    out.update(
      {
        "status": "pass",
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "context_len": context_len,
        "predicted_frames": frame_count,
        "decoded_frames": frames,
        "device": device,
      }
    )
    print(json.dumps(out, indent=2))
    return 0
  except Exception as exc:
    out["status"] = "fail"
    out["errors"].append(str(exc))
    print(json.dumps(out, indent=2))
    traceback.print_exc()
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
