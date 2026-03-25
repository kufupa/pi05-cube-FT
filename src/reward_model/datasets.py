import torch
from torch.utils.data import Dataset, DataLoader

class DroidRewardDataset(Dataset):
    """
    Dataset for fine-tuning Qwen3-VL on yes/no task completion.
    Yields (video_frames, instruction, label).
    """
    def __init__(self, data_root, task_name="stacking"):
        super().__init__()
        self.data_root = data_root
        self.task_name = task_name
        # Pseudo-dataset of real successful / failed trajectories
        self.samples = []
        for i in range(200):
            # Alternate positive and negative examples
            self.samples.append({
                "video": torch.randn(3, 16, 256, 256), # C, T, H, W
                "instruction": f"complete the {task_name} task",
                "label": 1 if i % 2 == 0 else 0
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["video"], sample["instruction"], torch.tensor(sample["label"], dtype=torch.float32)

def get_rm_dataloader(data_root, task_name="stacking", batch_size=2):
    dataset = DroidRewardDataset(data_root, task_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
