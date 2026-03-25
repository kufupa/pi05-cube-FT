import os
import json
import numpy as np
import h5py

print("Testing environment wrapper...")
try:
    from envs.libero import make_libero_env
    
    # Try creating an environment
    env = make_libero_env("KEEPOUT_libero_spatial_10", "pi0.5")
    print("Environment successfully instantiated!")
    
    # Check observation space
    env.seed(0)
    env.reset()
    dummy_action = [0.] * 7
    obs, reward, done, info = env.step(dummy_action)
    
    print(f"Observation keys: {obs.keys()}")
    if "agentview_image" in obs and "robot0_eye_in_hand_image" in obs:
        print(f"agentview shape: {obs['agentview_image'].shape}")
        print(f"eye_in_hand shape: {obs['robot0_eye_in_hand_image'].shape}")
    env.close()
    print("Environment step successful.\n")
except Exception as e:
    print(f"Environment creation failed: {e}\n")


print("Testing dataset index...")
try:
    index_path = "./data/libero/processed/libero_spatial_index.json"
    with open(index_path, "r") as f:
        index = json.load(f)
        
    print(f"Loaded index with {index['total_episodes']} total episodes.")
    
    if len(index["datasets"]) > 0:
        first_dataset = index["datasets"][0]
        file_path = os.path.join("./data/libero/raw", first_dataset["file_path"])
        print(f"Checking first dataset file: {file_path}")
        
        with h5py.File(file_path, "r") as f:
            if "data" in f:
                episodes = list(f["data"].keys())
                print(f"Found {len(episodes)} episodes in file.")
                if len(episodes) > 0:
                    first_ep = f["data"][episodes[0]]
                    obs_keys = list(first_ep["obs"].keys())
                    print(f"Observation keys in dataset: {obs_keys}")
                    num_steps = len(first_ep["actions"])
                    print(f"Episode length: {num_steps} steps")
            else:
                print("No 'data' group found in HDF5 file.")
except Exception as e:
    print(f"Dataset test failed: {e}")
