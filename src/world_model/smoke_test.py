import torch
import os
from src.world_model.models import CtrlWorldLikeModel

def main():
    checkpoint_dir = "./checkpoints/world_model/ctrl_world_base"
    print(f"Loading CtrlWorldLikeModel from {checkpoint_dir}")
    model = CtrlWorldLikeModel(checkpoint_dir)
    
    B, C, T, H, W = 1, 3, 4, 64, 64
    action_dim = 7
    
    print(f"Feeding dummy history frames of shape {(B, C, T, H, W)} and actions of shape {(B, T, action_dim)}")
    history_frames = torch.randn(B, C, T, H, W)
    actions = torch.randn(B, T, action_dim)
    
    predicted_frames = model(history_frames, actions)
    
    print(f"Forward output shape: {tuple(predicted_frames.shape)}")
    assert predicted_frames.numel() == 1, "Expected scalar proxy output from CtrlWorldLikeModel.forward"
    assert torch.isfinite(predicted_frames).all(), "Forward output contains non-finite values"
    print("World model smoke test passed!")

if __name__ == "__main__":
    main()
