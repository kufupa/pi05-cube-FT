import h5py
import numpy as np
import os

data_dir = "data/droid/stacking"
os.makedirs(data_dir, exist_ok=True)

for i in range(20):
    filepath = os.path.join(data_dir, f"episode_{i:04d}.hdf5")
    with h5py.File(filepath, "w") as f:
        # Create Dummy Data
        steps = 5
        # observations/images
        obs = f.create_group("observations")
        obs.create_dataset("images", data=np.random.randint(0, 256, size=(steps, 3, 256, 256), dtype=np.uint8))
        
        # actions
        f.create_dataset("actions", data=np.random.randn(steps, 7).astype(np.float32))
        
        # success (give vary success ground truths so it shows up in metrics)
        # e.g. success rate around 40%
        success_val = 1.0 if np.random.rand() < 0.4 else 0.0
        f.create_dataset("success", data=np.array(success_val, dtype=np.float32))

print(f"Generated 20 dummy HDF5 episodes in {data_dir}")
