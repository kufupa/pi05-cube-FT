import argparse
import yaml
import torch
from src.reward_model.models import QwenRewardModel
from src.reward_model.datasets import get_rm_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--exp_name", type=str, default="qwen_rm_droid", help="Experiment name")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    rm_config = config.get("reward_model", {})
    checkpoint_dir = rm_config.get("checkpoint_dir", "Qwen/Qwen3-VL-4B-Instruct")
    data_root = rm_config.get("data_root", "./data/droid/stacking/labeled")
    steps = rm_config.get("steps", 200)
    batch_size = rm_config.get("batch_size", 128)

    dataloader = get_rm_dataloader(data_root, task_name="stacking", batch_size=batch_size)
    model = QwenRewardModel(checkpoint_dir=checkpoint_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    print(f"Starting Qwen3-VL reward model fine-tuning for {steps} steps...")
    step = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    while step < steps:
        for video, instruction, label in dataloader:
            optimizer.zero_grad()
            
            # Predict success prob
            # This is a stub for the VLM loss computation
            logits = torch.randn(len(label), 1, requires_grad=True)
            loss = criterion(logits, label.unsqueeze(-1))
            
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step}/{steps} - Loss: {loss.item():.4f}")
            
            step += 1
            if step >= steps:
                break
                
    print("Training finished.")

if __name__ == "__main__":
    main()
