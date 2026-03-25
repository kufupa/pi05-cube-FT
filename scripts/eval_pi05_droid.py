import argparse
from src.vla.pi05_droid import Pi05DroidPolicy
from src.audit.result_schema import build_provenance, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="openpi", help="Path to pi0.5 checkpoint")
    parser.add_argument("--task", type=str, default="stacking", help="DROID task name")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--output", type=str, default="results_base.json", help="Output JSON path")
    args = parser.parse_args()

    policy = Pi05DroidPolicy.load_policy(args.checkpoint)
    success_rate = policy.evaluate(args.task, args.episodes)

    results = {
        "result_type": "baseline_heuristic",
        "model": "Base pi0.5-DROID",
        "task": args.task,
        "success_rate": success_rate,
        "episodes": args.episodes,
        "provenance": build_provenance(
            metric_backend="gripper_heuristic",
            checkpoint=args.checkpoint,
            config_path="scripts/eval_pi05_droid.py",
            task=args.task,
            episodes=args.episodes,
        ),
    }

    write_json(args.output, results)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
