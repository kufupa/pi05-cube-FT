import torch
import os
from src.reward_model.models import VLRewardModel

def main():
    checkpoint_dir = "./checkpoints/reward_model/qwen2vl7b_base"
    print(f"Loading VLRewardModel from {checkpoint_dir}")
    model = VLRewardModel(checkpoint_dir)
    
    instruction = "stack red on blue"
    # dummy video [T, C, H, W]
    video_frames = torch.randn(1, 16, 3, 256, 256)
    
    print(f"Feeding dummy video shape {video_frames.shape} and instruction '{instruction}'")
    prob = model.score(video_frames, instruction)
    
    print(f"Success prob: {prob.item():.4f}")
    assert 0 <= prob.item() <= 1, "Probability out of bounds"
    print("Reward model smoke test passed!")

if __name__ == "__main__":
    main()
