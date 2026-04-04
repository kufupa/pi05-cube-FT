import torch
import os
from src.reward_model.models import QwenRewardModel

def main():
    checkpoint_dir = "./checkpoints/reward_model/qwen2vl7b_base"
    os.environ.setdefault("VLAW_MOCK_REWARD", "1")
    print(f"Loading QwenRewardModel from {checkpoint_dir} (mock={os.environ.get('VLAW_MOCK_REWARD')})")
    model = QwenRewardModel(checkpoint_dir)

    instruction = "stack red on blue"
    # Dummy video in expected layout [B, C, T, H, W]
    video_frames = torch.rand(1, 3, 4, 64, 64)

    print(f"Feeding dummy video shape {video_frames.shape} and instruction '{instruction}'")
    prob = model.score(video_frames, instruction)

    assert prob.shape == (1, 1), f"Expected [1,1], got {tuple(prob.shape)}"
    print(f"Success prob: {prob.item():.4f}")
    assert 0 <= prob.item() <= 1, "Probability out of bounds"
    print("Reward model smoke test passed!")

if __name__ == "__main__":
    main()
