import torch
import os
from src.world_model.models import CtrlWorldLikeModel

def main():
    checkpoint_dir = "./checkpoints/world_model/ctrl_world_base"
    print(f"Loading CtrlWorldLikeModel from {checkpoint_dir}")
    model = CtrlWorldLikeModel(checkpoint_dir)
    
    B, C, T, H, W = 1, 3, 4, 256, 256
    action_dim = 7
    
    print(f"Feeding dummy history frames of shape {(B, C, T, H, W)} and actions of shape {(B, T, action_dim)}")
    history_frames = torch.randn(B, C, T, H, W)
    actions = torch.randn(B, T, action_dim)
    
    predicted_frames = model(history_frames, actions)
    
    print(f"Predicted frames shape: {predicted_frames.shape}")
    assert predicted_frames.shape == (B, C, T, H, W), "Shape mismatch"
    print("World model smoke test passed!")

if __name__ == "__main__":
    main()
