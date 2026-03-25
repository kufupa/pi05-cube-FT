import torch
from lerobot.common.policies.pi0_5_policy import Pi05Policy

def load_pi05_libero(config):
    return Pi05Policy.from_pretrained(config["pi05_checkpoint_dir"])

class Pi05LiberoPolicy:
    def __init__(self, config):
        self.policy = load_pi05_libero(config)

    def act(self, observation_dict, instruction_str):
        # Forward pass through the policy
        # The policy expects inputs usually formatted for its custom forward/act
        # We will wrap it as required.
        with torch.no_grad():
            action = self.policy.act(observation_dict, instruction_str)
        return action
