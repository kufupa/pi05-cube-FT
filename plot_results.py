import json
import matplotlib.pyplot as plt
from pathlib import Path

with open('results_vlaw.json', 'r') as f:
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
ax.set_ylabel('Success Rate')
ax.set_title('VLAW Pipeline: Success Rate vs Approach (Stacking Task)')

# Add value labels
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('vlaw_results_plot.png', dpi=300)
print("Saved plot to vlaw_results_plot.png")

if Path("results_base_real.json").exists():
    print("Detected results_base_real.json (Base-Real should appear if ingested by run_vlaw_loop).")
