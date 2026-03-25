import os
import json
import glob
import h5py
from pathlib import Path

def create_dataset_index():
    # Use environment variable for data root, defaulting to ./data/libero/raw
    data_root = os.environ.get("LIBERO_DATA_ROOT", "./data/libero/raw")
    spatial_data_dir = os.path.join(data_root, "libero_spatial")
    
    # Check if the directory exists
    if not os.path.exists(spatial_data_dir):
        print(f"Warning: Directory {spatial_data_dir} does not exist. No index created.")
        return
        
    index_data = {
        "datasets": [],
        "total_episodes": 0
    }
    
    # Scan for HDF5 files
    hdf5_files = glob.glob(os.path.join(spatial_data_dir, "**/*.hdf5"), recursive=True)
    
    for file_path in hdf5_files:
        try:
            with h5py.File(file_path, "r") as f:
                # Count episodes by looking at 'data' group keys
                if "data" in f:
                    episodes = list(f["data"].keys())
                    num_episodes = len(episodes)
                else:
                    num_episodes = 0
                    
                rel_path = os.path.relpath(file_path, data_root)
                index_data["datasets"].append({
                    "file_path": rel_path,
                    "num_episodes": num_episodes,
                    "episodes": episodes if num_episodes > 0 else []
                })
                index_data["total_episodes"] += num_episodes
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # Save the index to processed dir
    processed_dir = os.environ.get("LIBERO_PROCESSED_ROOT", "./data/libero/processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    out_file = os.path.join(processed_dir, "libero_spatial_index.json")
    with open(out_file, "w") as f:
        json.dump(index_data, f, indent=4)
        
    print(f"Created dataset index at {out_file} with {index_data['total_episodes']} total episodes.")

if __name__ == "__main__":
    create_dataset_index()
