import os
from huggingface_hub import snapshot_download

def main():
    print("Downloading World Model (Ctrl-World base)...")
    snapshot_download(
        repo_id="lerobot/ctrl_world_droid_base",
        local_dir="checkpoints/world_model/ctrl_world_base",
        local_dir_use_symlinks=False
    )
    
    print("Downloading Reward Model (Qwen2-VL-7B-Instruct)...")
    snapshot_download(
        repo_id="Qwen/Qwen2-VL-7B-Instruct",
        local_dir="checkpoints/reward_model/qwen2vl7b_base",
        local_dir_use_symlinks=False
    )
    
    print("Done downloading models.")

if __name__ == "__main__":
    main()
