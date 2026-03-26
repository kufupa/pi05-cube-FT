import argparse
import json
import os
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.audit.result_schema import build_provenance, write_json
from src.envs.droid import get_droid_dataset
from src.reward_model.models import QwenRewardModel
from src.vla.pi05_droid import Pi05DroidPolicy, _make_identity_linear_8
from src.world_model.models import CtrlWorldModel
from src.world_model.train_world_model import finetune_predictor


def collect_real_data(task: str, n_rollouts: int):
    print(f"[STAGE 1] collect_real_data: up to {n_rollouts} episodes for {task!r}")
    return get_droid_dataset(task_name=task, max_episodes=n_rollouts)


def dream_synthetic_trajectories(wm, policy, real_trajectories, n_traj: int):
    print(f"[STAGE 3] dream_synthetic_trajectories: n_traj={n_traj}")
    syn_trajectories = []
    if not real_trajectories:
        return syn_trajectories
    for i in range(n_traj):
        real_traj = real_trajectories[i % len(real_trajectories)]
        initial_obs = {
            "obs": real_traj["obs"][0].unsqueeze(0),
            "state": real_traj["state"][0:1],
            "instruction": real_traj.get("instruction", ""),
            "timestep": 0,
            "droid_episode_ref": real_traj,
        }
        rolled_out = wm.rollout(initial_obs, policy, horizon=5, n_traj=1)[0]
        frames = torch.stack([step["dreamt_image"].squeeze(0).cpu() for step in rolled_out["steps"]])
        actions = torch.stack([step["action"].squeeze(0).cpu() for step in rolled_out["steps"]])
        t = frames.shape[0]
        rs = real_traj["state"]
        if rs.shape[0] >= t:
            st_slice = rs[:t].clone()
        else:
            pad = torch.zeros(t - rs.shape[0], rs.shape[1], dtype=rs.dtype)
            st_slice = torch.cat([rs, pad], dim=0)
        syn_traj = {
            "obs": frames,
            "actions": actions[:, :7] if actions.shape[-1] >= 7 else actions,
            "state": st_slice,
            "expert_actions_8": actions,
            "success": 0.0,
            "instruction": real_traj.get("instruction", ""),
        }
        syn_trajectories.append(syn_traj)
    return syn_trajectories


def filter_with_reward_model(rm, trajectories, threshold: float):
    print(f"[STAGE 4a] filter_with_reward_model: threshold={threshold}, n_in={len(trajectories)}")
    filtered = []
    for traj in trajectories:
        obs = traj["obs"]
        clip = obs.permute(1, 0, 2, 3).unsqueeze(0)
        prob = rm.score_trajectory(clip, traj["instruction"])
        p = float(prob.view(-1)[0].item())
        if p > threshold:
            traj = dict(traj)
            traj["success"] = p
            filtered.append(traj)
    print(f"[STAGE 4a] kept {len(filtered)} trajectories.")
    return filtered


def fine_tune_policy(policy: Pi05DroidPolicy, trajectories: list, steps: int, device: torch.device):
    print(f"[STAGE 4b] fine_tune_policy (action adapter): steps={steps}, trajs={len(trajectories)}")
    if not trajectories or steps <= 0:
        return policy
    if policy.action_adapter is None:
        policy.action_adapter = _make_identity_linear_8()
    policy.action_adapter = policy.action_adapter.to(device)
    policy.action_adapter.train()
    opt = torch.optim.Adam(policy.action_adapter.parameters(), lr=1e-3)
    for s in range(steps):
        traj = trajectories[s % len(trajectories)]
        t_max = traj["obs"].shape[0]
        if t_max < 1:
            continue
        t = random.randrange(t_max)
        frame = traj["obs"][t : t + 1].to(device)
        st = (
            traj["state"][t : t + 1].to(device)
            if "state" in traj
            else torch.zeros(1, 14, device=device)
        )
        inst = traj.get("instruction", "")
        obs_dict = {"obs": frame, "state": st, "instruction": inst}
        with torch.no_grad():
            base = policy._compute_action_tensor(obs_dict).to(device)
        if "expert_actions_8" in traj and traj["expert_actions_8"].shape[0] > t:
            tgt = traj["expert_actions_8"][t].view(1, -1).float().to(device)
        else:
            a7 = traj["actions"][t].view(1, -1).float().to(device)
            g = st[:, -1:].float().to(device)
            tgt = torch.cat([a7, g], dim=-1)
        if tgt.shape[-1] < 8:
            pad = torch.zeros(1, 8 - tgt.shape[-1], device=device)
            tgt = torch.cat([tgt, pad], dim=-1)
        elif tgt.shape[-1] > 8:
            tgt = tgt[:, :8]
        adapted = policy.action_adapter(base)
        loss = F.mse_loss(adapted, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return policy


def save_rollout_gif(syn_trajectories: list, path: str) -> None:
    if not syn_trajectories:
        return
    last = syn_trajectories[-1]
    frames_rgb = []
    for i in range(last["obs"].shape[0]):
        im = last["obs"][i].detach().cpu().float().clamp(0, 1)
        hwc = (im.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        frames_rgb.append(hwc)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames_rgb, fps=4)
    print(f"[artifact] wrote {path} ({len(frames_rgb)} frames)")


def evaluate_baseline(policy_cls, checkpoint, task, n_eval=5):
    policy = policy_cls.load_policy(checkpoint)
    return policy.evaluate(task, n_episodes=n_eval), policy


def _policy_backend(p: Pi05DroidPolicy) -> str:
    if p.uses_openpi():
        return "openpi"
    return "gripper_heuristic"


def _rm_backend() -> str:
    if os.environ.get("VLAW_MOCK_REWARD", "").strip() in ("1", "true", "True", "yes"):
        return "mock"
    if not torch.cuda.is_available():
        return "mock"
    return "qwen2_vl_4bit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to store loop artifacts (WM ckpt, GIF, etc.).",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results_vlaw.json",
        help="Path to write final metrics JSON.",
    )
    parser.add_argument(
        "--base-real-path",
        type=str,
        default="results_base_real.json",
        help="Optional existing real-eval JSON to ingest as Base-Real.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run + VLAW_MOCK_REWARD for login CPU (sets env inside process).",
    )
    args = parser.parse_args()

    # Transformers pulls optional TensorFlow for image helpers; that collides with tfrecord's
    # embedded example protos (duplicate tensorflow.BytesList). Torch-only is enough for Qwen2-VL.
    os.environ.setdefault("USE_TF", "0")

    if args.smoke:
        os.environ["VLAW_MOCK_REWARD"] = "1"

    with open(args.config, "r") as f:
        full = yaml.safe_load(f)
    config = full["vlaw"]
    wm_cfg = full.get("world_model", {})

    task = config["task_name"]
    n_real = int(config["n_real_rollouts_wm"])
    wm_steps = int(config["wm_finetune_steps"])
    n_syn = int(config["n_synthetic_trajectories"])
    rm_thresh = float(config["rm_threshold"])
    policy_steps = int(config["policy_finetune_steps"])
    vlaw_iters = int(config["vlaw_iterations"])

    if args.smoke:
        n_real = 1
        wm_steps = min(10, wm_steps)
        n_syn = 1
        policy_steps = min(10, policy_steps)
        vlaw_iters = 1
        rm_thresh = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    base_real_path = Path(args.base_real_path)

    policy_ckpt = config["base_policy_ckpt"]
    if args.smoke:
        policy_ckpt = "heuristic-fallback"

    wm = CtrlWorldModel(config["world_model_ckpt"])
    wm_ckpt_default = output_dir / "wm_predictor.pt"
    wm_ckpt_path = wm_cfg.get("save_path", str(wm_ckpt_default))
    if Path(wm_ckpt_path).exists():
        try:
            wm.load_predictor(wm_ckpt_path)
            print(f"[init] Loaded WM predictor from {wm_ckpt_path}")
        except Exception as exc:
            print(f"[init] Could not load WM ckpt: {exc}")

    rm = QwenRewardModel(config["reward_model_ckpt"])
    policy = Pi05DroidPolicy(policy_ckpt)

    results = {
        "result_type": "vlaw_loop",
        "task": task,
        "metrics": {},
        "metrics_meta": {},
        "provenance": build_provenance(
            metric_backend="vlaw_pipeline",
            checkpoint=policy_ckpt,
            config_path=args.config,
            task=task,
            extra={
                "config_base_policy_ckpt": config["base_policy_ckpt"],
                "world_model_ckpt": config["world_model_ckpt"],
                "reward_model_ckpt": config["reward_model_ckpt"],
                "rm_threshold": rm_thresh,
                "smoke": bool(args.smoke),
            },
        ),
    }

    print("[STAGE 0] Evaluating base policy (heuristic metric)")
    base_score, _ = evaluate_baseline(Pi05DroidPolicy, policy_ckpt, task)
    results["metrics"]["Base"] = base_score
    results["metrics_meta"]["Base"] = {
        "backend": _policy_backend(policy),
        "rm_backend": _rm_backend(),
        "episodes": 5,
    }

    print("[STAGE 1] Real rollouts")
    real_data = collect_real_data(task, n_rollouts=n_real)
    if not real_data:
        print("FATAL: no DROID episodes; check data/droid_sample/droid_100/")
        raise SystemExit(2)

    print("[STAGE 2–4] VLAW iterations (WM train → dream → RM → adapter per iter)")
    for loop_iter in range(1, vlaw_iters + 1):
        print(f"[STAGE 2] VLAW iter {loop_iter}: world model finetune")
        wm_max_eps = int(wm_cfg.get("max_episodes", n_real))
        if args.smoke:
            wm_max_eps = max(n_real, 1)
        finetune_predictor(
            wm,
            task_name=task,
            max_episodes=wm_max_eps,
            steps=wm_steps,
            batch_size=int(wm_cfg.get("batch_size", 4)),
            lr=float(wm_cfg.get("lr", 1e-4)),
            device=device,
            save_path=wm_ckpt_path,
        )
        wm.release_cuda()

        print(f"[STAGE 3] VLAW iter {loop_iter}: synthetic rollouts")
        syn_data = dream_synthetic_trajectories(wm, policy, real_data, n_traj=n_syn)
        save_rollout_gif(syn_data, str(output_dir / "wm_rollout_preview.gif"))

        print(f"[STAGE 4a] VLAW iter {loop_iter}: reward filter")
        filtered_syn = filter_with_reward_model(rm, syn_data, threshold=rm_thresh)
        rm.to_cpu_and_clear()

        combined = real_data + filtered_syn
        print(f"[STAGE 4b] VLAW iter {loop_iter}: policy adapter")
        policy = fine_tune_policy(policy, combined, steps=policy_steps, device=device)

        metric_name = f"VLAW-{loop_iter}"
        results["metrics"][metric_name] = policy.evaluate(task, dataset=real_data)
        results["metrics_meta"][metric_name] = {
            "backend": _policy_backend(policy),
            "rm_backend": _rm_backend(),
            "episodes": len(real_data),
        }

    if base_real_path.exists():
        try:
            base_real = json.loads(base_real_path.read_text(encoding="utf-8"))
            if isinstance(base_real, dict) and "success_rate" in base_real:
                results["metrics"]["Base-Real"] = float(base_real["success_rate"])
                prov = base_real.get("provenance") if isinstance(base_real.get("provenance"), dict) else {}
                results["metrics_meta"]["Base-Real"] = {
                    "backend": prov.get("metric_backend", "openpi_inference"),
                    "primary_metric": prov.get("primary_metric"),
                    "episodes": int(base_real.get("episodes", 0)),
                    "source_file": str(base_real_path),
                }
        except Exception as exc:
            print(f"[WARN] Could not ingest results_base_real.json: {exc}")

    write_json(str(results_path), results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
