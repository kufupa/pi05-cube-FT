"""
Real π0.5-DROID evaluation using OpenPI — runs inside the openpi conda env on PBS GPU node.
Outputs JSON to stdout / --output file, consumed by run_vlaw_loop.py.

Uses the flat observation/* contract from external/openpi/examples/droid/main.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit.result_schema import build_provenance, write_json
from src.envs.droid import get_droid_dataset
from src.envs.droid.observation_openpi import build_openpi_droid_request
from src.vla.pi05_droid import Pi05DroidPolicy


def _stacking_gripper_heuristic_update(episode_success: float, pred_chunk: np.ndarray) -> float:
    """Gripper open if last dim > 0.5 after inference (matches DROID main.py binarization rule)."""
    raw_g = float(pred_chunk[0, -1])
    g_open = raw_g > 0.5
    if not g_open:
        episode_success = max(episode_success, 0.5)
    if g_open and episode_success >= 0.5:
        episode_success = 1.0
    return episode_success


def run_openpi_smoke(policy, episodes: list) -> None:
    from openpi.policies.droid_policy import make_droid_example

    ex = make_droid_example()
    out = policy.infer(ex)
    actions = np.asarray(out["actions"])
    if actions.shape[-1] != 8:
        raise AssertionError(f"smoke: expected actions[...,8], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise AssertionError("smoke: non-finite actions from make_droid_example()")

    if episodes:
        req = build_openpi_droid_request(episodes[0], 0, episodes[0].get("instruction", ""))
        out2 = policy.infer(req)
        a2 = np.asarray(out2["actions"])
        if a2.shape[-1] != 8:
            raise AssertionError(f"smoke: real request expected [...,8], got {a2.shape}")
        if not np.isfinite(a2).all():
            raise AssertionError("smoke: non-finite actions from dataset timestep")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="stacking")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--output", default="results_base_real.json")
    parser.add_argument("--config_name", default="pi05_droid")
    parser.add_argument("--subsample", type=int, default=10, help="Run every N-th timestep per episode.")
    parser.add_argument("--smoke-only", action="store_true", help="Run shape probes and exit.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    openpi_ready = True
    policy = None
    try:
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as _config

        print(f"Loading OpenPI policy: {args.config_name}")
        ckpt_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{args.config_name}")
        config = _config.get_config(args.config_name)
        policy = policy_config.create_trained_policy(config, ckpt_dir)
        print("Policy loaded.")
    except Exception as exc:
        openpi_ready = False
        policy = None
        print(f"[WARN] OpenPI load failed; falling back to heuristic baseline: {exc}")

    episodes = get_droid_dataset(
        task_name=args.task, max_episodes=args.n_episodes, debug=args.debug
    )
    print(f"Loaded {len(episodes)} episodes.")

    if openpi_ready and policy is not None:
        run_openpi_smoke(policy, episodes)
        print("OpenPI smoke checks passed.")
        if args.smoke_only:
            return

    if not openpi_ready:
        heuristic = Pi05DroidPolicy.load_policy("heuristic-fallback")
        mean = heuristic.evaluate(args.task, n_episodes=args.n_episodes, dataset=episodes)
        result = {
            "result_type": "baseline_openpi_fallback",
            "task": args.task,
            "success_rate": float(mean),
            "dataset_success_rate": float(np.mean([float(ep["success"]) for ep in episodes]))
            if episodes
            else 0.0,
            "gripper_heuristic_success_rate": None,
            "action_agreement_mse_mean": None,
            "episodes": len(episodes),
            "provenance": build_provenance(
                metric_backend="gripper_heuristic_fallback",
                checkpoint=f"gs://openpi-assets/checkpoints/{args.config_name}",
                config_path="scripts/eval_pi05_droid_real.py",
                task=args.task,
                episodes=len(episodes),
                extra={
                    "config_name": args.config_name,
                    "primary_metric": "none_openpi_unavailable",
                    "metric_notes": {
                        "success_rate": "Pi05DroidPolicy gripper heuristic blend (Track 1).",
                    },
                },
            ),
        }
        write_json(args.output, result)
        print(json.dumps(result))
        return

    dataset_success_rate = (
        float(np.mean([float(ep["success"]) for ep in episodes])) if episodes else 0.0
    )

    action_mses_episode: list[float] = []
    gripper_heuristic_scores: list[float] = []

    for ep in episodes:
        obs_seq = ep["obs"]
        instruction = ep.get("instruction", "")
        mses: list[float] = []
        episode_grip = 0.0

        for t in range(0, obs_seq.shape[0], args.subsample):
            request_data = build_openpi_droid_request(ep, t, instruction)
            try:
                out = policy.infer(request_data)
                actions = np.asarray(out["actions"])
            except Exception as e:
                print(f"  Inference error at step {t}: {e}")
                continue

            if actions.ndim != 2 or actions.shape[-1] != 8:
                print(f"  Unexpected actions shape {actions.shape} at t={t}")
                continue

            if args.task.lower().find("stack") >= 0 or "stack" in instruction.lower():
                episode_grip = _stacking_gripper_heuristic_update(episode_grip, actions)

            h = actions.shape[0]
            expert = ep["expert_actions_8"][t : t + h].detach().cpu().numpy()
            L = min(h, expert.shape[0])
            if L <= 0:
                continue
            mse = float(((actions[:L] - expert[:L]) ** 2).mean())
            mses.append(mse)
            if args.debug:
                print(f"  t={t} chunk_mse={mse:.6f} actions.shape={actions.shape}")

        if mses:
            action_mses_episode.append(float(np.mean(mses)))
        else:
            action_mses_episode.append(float("nan"))

        gripper_heuristic_scores.append(episode_grip)

    finite_mses = [m for m in action_mses_episode if np.isfinite(m)]
    action_mse_mean = float(np.mean(finite_mses)) if finite_mses else float("nan")

    # Primary headline: imitation alignment (high when MSE is low)
    if finite_mses:
        episode_scores = [float(np.exp(-m)) for m in action_mses_episode if np.isfinite(m)]
        success_rate = float(np.mean(episode_scores)) if episode_scores else 0.0
    else:
        success_rate = 0.0

    gripper_heuristic_rate = (
        float(np.mean(gripper_heuristic_scores)) if gripper_heuristic_scores else 0.0
    )

    print(
        f"Primary success_rate (exp(-chunk_MSE), subsample={args.subsample}): {success_rate:.4f}"
    )
    print(f"Dataset success label mean: {dataset_success_rate:.4f}")
    print(f"Mean action-chunk MSE vs logged expert: {action_mse_mean:.6f}")
    print(f"Gripper heuristic (stacking proxy): {gripper_heuristic_rate:.4f}")

    result = {
        "result_type": "baseline_openpi_real",
        "task": args.task,
        "success_rate": success_rate,
        "dataset_success_rate": dataset_success_rate,
        "gripper_heuristic_success_rate": gripper_heuristic_rate,
        "action_agreement_mse_mean": action_mse_mean,
        "episodes": len(episodes),
        "provenance": build_provenance(
            metric_backend="openpi_inference",
            checkpoint=f"gs://openpi-assets/checkpoints/{args.config_name}",
            config_path="scripts/eval_pi05_droid_real.py",
            task=args.task,
            episodes=len(episodes),
            extra={
                "config_name": args.config_name,
                "subsample": args.subsample,
                "primary_metric": "action_agreement_exp_neg_mean_chunk_mse",
                "metric_notes": {
                    "success_rate": (
                        "Mean over episodes of exp(-mean_MSE) between first predicted action chunk "
                        "and logged expert (joint_velocity + gripper_position), same subsampling."
                    ),
                    "dataset_success_rate": "Mean of per-episode RLDS final reward (sanity baseline).",
                    "gripper_heuristic_success_rate": (
                        "Stacking proxy on predicted actions: close (<0.5 open) then open (>0.5) "
                        "pattern on subsampled steps (secondary)."
                    ),
                    "action_agreement_mse_mean": (
                        "Mean squared-error vs logged 8D expert command (primary diagnostic)."
                    ),
                },
            },
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
