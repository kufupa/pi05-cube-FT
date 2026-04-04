import os

import torch

from src.vla.pi05_ur5e import Pi05UR5ePolicy

def main():
    checkpoint = os.environ.get("PI05_SMOKE_CHECKPOINT", "checkpoints/pi05/libero_base")
    require_openpi = os.environ.get("PI05_SMOKE_REQUIRE_OPENPI", "").strip().lower() in {"1", "true", "yes"}

    print(f"Loading pi0.5 policy wrapper from: {checkpoint}")
    policy_wrapper = Pi05UR5ePolicy(checkpoint)

    # Fake observation matching UR5e wrapper expectations.
    obs_dict = {
        "obs": torch.zeros((3, 256, 256), dtype=torch.float32),
        "wrist_obs": torch.zeros((3, 256, 256), dtype=torch.float32),
        "joints_6": torch.zeros((6,), dtype=torch.float32),
        "gripper_open_01": 1.0,
        "instruction": "stack red block on blue",
    }

    print("Running inference...")
    action = policy_wrapper.act(obs_dict)

    assert action.shape == torch.Size([1, 7]), f"Expected shape [1, 7], got {action.shape}"
    if require_openpi:
        assert policy_wrapper.uses_openpi(), "OpenPI failed to load but PI05_SMOKE_REQUIRE_OPENPI=1"
    print(f"SUCCESS: pi0.5 wrapper produced action shape [1,7], openpi_loaded={policy_wrapper.uses_openpi()}")

if __name__ == "__main__":
    main()
