import sys
sys.path.append("external/openpi/src")
sys.path.append("external/openpi/packages/openpi-models/src")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from openpi.training.droid_rlds_dataset import DroidRldsDataset, RLDSDataset

dataset_cfg = RLDSDataset(name="droid", version="0.1.0", weight=1.0)

try:
    loader = DroidRldsDataset(
        data_dir="gs://gresearch/robotics",
        batch_size=1,
        datasets=[dataset_cfg],
        shuffle=False
    )
    
    print("Loader initialized!")
    
    for batch in loader:
        print("Success! Got batch:")
        print(batch.keys())
        if 'observation' in batch:
            print(f"Image shape: {batch['observation']['image'].shape}")
            print(f"Action shape: {batch['actions'].shape}")
        break

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed: {e}")
