"""Finetune ActionCondFramePredictor on DROID frame pairs."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.world_model.datasets import get_droid_wm_dataloader
from src.world_model.models import CtrlWorldModel


def finetune_predictor(
    wm: CtrlWorldModel,
    *,
    task_name: str,
    max_episodes: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    save_path: str,
    seed: int = 0,
) -> None:
    wm.to_device(device)
    wm.predictor.train()
    opt = torch.optim.Adam(wm.predictor.parameters(), lr=lr)
    dl = get_droid_wm_dataloader(
        task_name=task_name,
        max_episodes=max_episodes,
        batch_size=batch_size,
        seed=seed,
    )
    it = iter(dl)
    log_every = max(1, steps // 10)
    for step in range(steps):
        try:
            ft, act, ftp = next(it)
        except StopIteration:
            it = iter(dl)
            try:
                ft, act, ftp = next(it)
            except StopIteration:
                print("[train_world_model] Empty dataloader; skipping training.")
                return
        ft = ft.to(device)
        act = act.to(device)
        ftp = ftp.to(device)
        opt.zero_grad(set_to_none=True)
        pred = wm.predictor(ft, act)
        loss = F.mse_loss(pred, ftp)
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"  [WM train] step {step}/{steps} loss={loss.item():.6f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    wm.save_predictor(save_path)
    print(f"[train_world_model] Saved predictor to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="ctrl_world_droid")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    wm_cfg = config.get("world_model", {})
    vlaw_cfg = config.get("vlaw", {})
    checkpoint_dir = wm_cfg.get("checkpoint_dir", "yjguo/Ctrl-World")
    steps = int(wm_cfg.get("steps", 500))
    batch_size = int(wm_cfg.get("batch_size", 4))
    lr = float(wm_cfg.get("lr", 1e-4))
    max_episodes = int(wm_cfg.get("max_episodes", vlaw_cfg.get("n_real_rollouts_wm", 20)))
    task_name = vlaw_cfg.get("task_name", "stacking")
    save_path = wm_cfg.get("save_path", "output/wm_predictor.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = CtrlWorldModel(checkpoint_dir=checkpoint_dir)
    finetune_predictor(
        wm,
        task_name=task_name,
        max_episodes=max_episodes,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
