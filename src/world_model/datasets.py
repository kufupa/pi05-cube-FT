import json
import os
import torch
from torch.utils.data import Dataset, DataLoader

class LiberoWorldModelDataset(Dataset):
    """
    Dataloader stub that reads LIBERO episodes and returns short video clips + action chunks
    suitable for an action-conditioned video diffusion model.
    """
    def __init__(self, data_root, index_json):
        super().__init__()
        self.data_root = data_root
        self.index_path = index_json
        self.episodes = []
        
        # In a real scenario, we load the index JSON
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self.episodes = json.load(f)
        
    def __len__(self):
        # Stub length
        return max(1, len(self.episodes))

    def __getitem__(self, idx):
        # Stub: Return dummy (history_frames, actions)
        # Real implementation would sample contiguous frames and action chunks from an episode
        history_frames = torch.randn(3, 8, 256, 256) # (C, T, H, W)
        actions = torch.randn(8, 7) # (T, action_dim)
        future_frames = torch.randn(3, 8, 256, 256) # (C, T, H, W)
        
        return history_frames, actions, future_frames

def get_dataloader(data_root, index_json, batch_size=8):
    dataset = LiberoWorldModelDataset(data_root, index_json)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
