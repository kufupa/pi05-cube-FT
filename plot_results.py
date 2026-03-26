import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results_vlaw.json")
    parser.add_argument("--out", type=str, default="vlaw_results_plot.png")
    parser.add_argument("--base-real", type=str, default="results_base_real.json")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)["metrics"]

    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = []
    for label in labels:
        if "Base-Real" in label:
            colors.append("mediumpurple")
        elif label == "Base":
            colors.append("skyblue")
        elif label.startswith("Filtered-BC"):
            colors.append("lightgreen")
        else:
            colors.append("coral")
    bars = ax.bar(labels, values, color=colors)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success Rate")
    ax.set_title("VLAW Pipeline: Success Rate vs Approach (Stacking Task)")

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            round(yval, 4),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

    if Path(args.base_real).exists():
        print(f"Detected {args.base_real} (Base-Real should appear if ingested by run_vlaw_loop).")


if __name__ == "__main__":
    main()
