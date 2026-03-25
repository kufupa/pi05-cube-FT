import torch
from src.vla.pi05_libero import Pi05LiberoPolicy

def main():
    config = {"pi05_checkpoint_dir": "checkpoints/pi05/libero_base"}
    print("Loading pi0.5 policy...")
    policy_wrapper = Pi05LiberoPolicy(config)
    
    # Create fake observations
    obs_dict = {
        "rgb": torch.zeros((1, 3, 256, 256), dtype=torch.float32),
        "state": torch.zeros((1, 15), dtype=torch.float32),
        "prev_action": torch.zeros((1, 7), dtype=torch.float32),
    }
    instruction = "stack red block on blue"
    
    print(f"Running inference with instruction: '{instruction}'")
    action = policy_wrapper.act(obs_dict, instruction)
    
    assert action.shape == torch.Size([1, 7]), f"Expected shape [1, 7], got {action.shape}"
    print("SUCCESS: π₀.₅ LIBERO loaded and rollout shape correct.")

if __name__ == "__main__":
    main()
