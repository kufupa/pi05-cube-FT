import sys
import os
sys.path.append(os.getcwd())

from src.vla.pi05_droid import Pi05DroidPolicy
from src.envs.droid import get_droid_dataset

print("Starting OpenPI diagnostic...")
try:
    print("Initializing policy...")
    policy = Pi05DroidPolicy()
    print("Policy initialized.")
    
    print("Loading episodes...")
    episodes = get_droid_dataset('stacking', max_episodes=2)
    print(f"Loaded {len(episodes)} episodes.")
    
    print("Evaluating...")
    score = policy.evaluate('stacking', n_episodes=2, dataset=episodes)
    print(f"Real π0.5 score: {score}")
except Exception as e:
    import traceback
    traceback.print_exc()
