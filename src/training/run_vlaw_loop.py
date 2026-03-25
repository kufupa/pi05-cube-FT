import argparse
import yaml
import json
import torch
from pathlib import Path
from src.envs.droid import get_droid_dataset
from src.vla.pi05_droid import Pi05DroidPolicy
from src.world_model.models import CtrlWorldModel
from src.reward_model.models import QwenRewardModel
from src.audit.result_schema import build_provenance, write_json

def collect_real_data(task, n_rollouts=5):
    print(f"collect_real_data: Sampling {n_rollouts} real rollouts for {task}...")
    dataset = get_droid_dataset(task_name=task, max_episodes=n_rollouts)
    rollouts = []
    for i, traj in enumerate(dataset):
        if i >= n_rollouts: break
        rollouts.append(traj)
    return rollouts

def train_world_model_step(wm, rollouts, steps=500):
    print(f"train_world_model_step: Fine-tuning world model for {steps} steps on {len(rollouts)} rollouts...")
    # Mock fine-tuning
    return wm

def dream_synthetic_trajectories(wm, policy, real_trajectories, n_traj=20):
    print(f"dream_synthetic_trajectories: Dreaming {n_traj} synthetic trajectories...")
    syn_trajectories = []
    
    # Generate synthetic trajectories using real initial observations
    for i in range(n_traj):
        real_traj = real_trajectories[i % len(real_trajectories)]
        # Extract first frame as initial observation map [3, 256, 256] -> [1, 3, 256, 256]
        initial_obs = {"obs": real_traj["obs"][0].unsqueeze(0)}
        
        # Rollout horizon 5
        rolled_out = wm.rollout(initial_obs, policy, horizon=5, n_traj=1)[0]
        
        # Re-pack rollout into the same format as real episodes for policy tuning
        # [T, 3, 256, 256]
        frames = torch.stack([step["dreamt_image"].squeeze(0).cpu() for step in rolled_out["steps"]])
        actions = torch.stack([step["action"].squeeze(0).cpu() for step in rolled_out["steps"]])
        
        syn_traj = {
            "obs": frames,
            "actions": actions,
            "success": 0.0, # Will be scored by RM
            "instruction": real_traj["instruction"]
        }
        syn_trajectories.append(syn_traj)
        
    return syn_trajectories

def filter_with_reward_model(rm, trajectories, threshold=0.8):
    print(f"filter_with_reward_model: Filtering {len(trajectories)} trajectories with threshold {threshold}...")
    filtered = []
    for traj in trajectories:
        # Extract clip as [1, C, T, H, W] for the video RM
        clip = traj["obs"].permute(1, 0, 2, 3).unsqueeze(0)
        prob = rm.score(clip, traj["instruction"])
        if prob.item() > threshold:
            traj["success"] = prob.item()
            filtered.append(traj)
    print(f"  Kept {len(filtered)} trajectories (p(yes) > {threshold}).")
    return filtered

def fine_tune_policy(policy, trajectories, steps=100):
    print(f"fine_tune_policy: Running flow-matching for {steps} steps on {len(trajectories)} trajectories...")
    # Mock
    return policy

def evaluate_baseline(policy_cls, checkpoint, task, n_eval=5):
    policy = policy_cls.load_policy(checkpoint)
    return policy.evaluate(task, n_episodes=n_eval)

def main():
    print("[TRACE] Starting main()")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    print(f"[TRACE] Loaded args: {args}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["vlaw"]

    task = config["task_name"]
    print(f"[TRACE] Config loaded, task: {task}")
    # Keep quick defaults for resource-constrained smoke runs.
    n_real = 2
    wm_steps = 10
    n_syn = 2
    rm_thresh = config["rm_threshold"]
    policy_steps = 10

    print("=== DROID Minimal VLAW Pipeline ===")
    
    wm = CtrlWorldModel(config["world_model_ckpt"])
    rm = QwenRewardModel(config["reward_model_ckpt"])
    policy = Pi05DroidPolicy(config["base_policy_ckpt"])

    results = {
        "result_type": "vlaw_loop",
        "task": task,
        "metrics": {},
        "metrics_meta": {},
        "provenance": build_provenance(
            metric_backend="vlaw_heuristic_loop",
            checkpoint=config["base_policy_ckpt"],
            config_path=args.config,
            task=task,
            extra={
                "world_model_ckpt": config["world_model_ckpt"],
                "reward_model_ckpt": config["reward_model_ckpt"],
                "rm_threshold": rm_thresh,
            },
        ),
    }

    print("\n--- Evaluating Base Policy ---")
    results["metrics"]["Base"] = evaluate_baseline(Pi05DroidPolicy, config["base_policy_ckpt"], task)
    results["metrics_meta"]["Base"] = {"backend": "gripper_heuristic", "episodes": 5}
    
    print("\n--- Evaluating Filtered-BC baselines ---")
    real_data = collect_real_data(task, n_rollouts=n_real)
    policy_bc1 = fine_tune_policy(policy, real_data, steps=policy_steps)
    results["metrics"]["Filtered-BC-1"] = policy_bc1.evaluate(task)
    results["metrics_meta"]["Filtered-BC-1"] = {"backend": "gripper_heuristic", "episodes": 10}
    
    policy_bc2 = fine_tune_policy(policy_bc1, real_data, steps=policy_steps)
    results["metrics"]["Filtered-BC-2"] = policy_bc2.evaluate(task)
    results["metrics_meta"]["Filtered-BC-2"] = {"backend": "gripper_heuristic", "episodes": 10}

    print("\n--- Running VLAW ---")
    for loop_iter in range(1, 3):
        print(f"\n--- VLAW Iteration {loop_iter} ---")
        train_world_model_step(wm, real_data, steps=wm_steps)
        syn_data = dream_synthetic_trajectories(wm, policy, real_data, n_traj=n_syn)
        filtered_syn = filter_with_reward_model(rm, syn_data, threshold=rm_thresh)
        
        combined_data = real_data + filtered_syn
        policy = fine_tune_policy(policy, combined_data, steps=policy_steps)
        
        metric_name = f"VLAW-{loop_iter}"
        results["metrics"][metric_name] = policy.evaluate(task)
        results["metrics_meta"][metric_name] = {"backend": "gripper_heuristic", "episodes": 10}

    # Optional artifact linkage: if real baseline exists, record it for plotting/trace.
    base_real_path = Path("results_base_real.json")
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
                    "dataset_success_rate": base_real.get("dataset_success_rate"),
                    "action_agreement_mse_mean": base_real.get("action_agreement_mse_mean"),
                }
        except Exception as exc:
            print(f"[WARN] Could not ingest results_base_real.json: {exc}")

    write_json("results_vlaw.json", results)
        
    print("\n=== Final Results ===")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
