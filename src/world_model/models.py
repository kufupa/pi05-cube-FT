import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download

from src.world_model.mini_action_wm import ActionCondFramePredictor


class CtrlWorldModel(nn.Module):
    """
    Action-conditioned next-frame model + optional Ctrl-World hub weight probe.
    """

    def __init__(self, checkpoint_dir="yjguo/Ctrl-World"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hub_state_dict = None
        self.predictor = ActionCondFramePredictor(action_dim=8, hidden=64)

        try:
            print(f"[CtrlWorldModel] Attempting hf_hub_download({checkpoint_dir!r}, checkpoint-10000.pt)")
            ckpt_path = hf_hub_download(repo_id=checkpoint_dir, filename="checkpoint-10000.pt")
            self.hub_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"[CtrlWorldModel] Hub checkpoint loaded ({len(self.hub_state_dict)} keys); mini-predictor trains separately.")
        except Exception as e:
            print(f"[CtrlWorldModel] Hub load skipped: {e}")

    def load_predictor(self, path: str, map_location=None) -> None:
        loc = map_location or self.device
        try:
            data = torch.load(path, map_location=loc, weights_only=True)
        except TypeError:
            data = torch.load(path, map_location=loc)
        self.predictor.load_state_dict(data)

    def save_predictor(self, path: str) -> None:
        torch.save(self.predictor.state_dict(), path)

    def rollout(self, initial_obs, policy, horizon=10, n_traj=1):
        synthetic_trajectories = []
        device = self.device
        self.predictor.eval()

        for _ in range(n_traj):
            traj = {"steps": []}
            obs = dict(initial_obs)
            for t in range(horizon):
                obs.setdefault("timestep", t)
                action = policy.act(obs)

                frame = obs.get("obs")
                if frame is None:
                    frame = torch.randn(1, 3, 256, 256, device=device)
                else:
                    frame = frame.to(device).float()
                    if frame.dim() == 3:
                        frame = frame.unsqueeze(0)
                act = action.to(device).float()
                if act.dim() == 1:
                    act = act.unsqueeze(0)

                with torch.no_grad():
                    dreamt = self.predictor(frame, act)

                traj["steps"].append(
                    {
                        "observation": obs,
                        "action": action.detach().cpu(),
                        "dreamt_image": dreamt.detach().cpu(),
                    }
                )
                st = obs.get("state")
                if isinstance(st, torch.Tensor):
                    next_state = st.clone()
                else:
                    next_state = torch.zeros(1, 14, device=device)
                obs = {
                    "obs": dreamt.detach(),
                    "state": next_state,
                    "instruction": obs.get("instruction", ""),
                    "timestep": t + 1,
                    "droid_episode_ref": obs.get("droid_episode_ref"),
                }
            synthetic_trajectories.append(traj)
        return synthetic_trajectories

    def forward(self, history_frames, actions):
        if self.hub_state_dict:
            first_param = next(iter(self.hub_state_dict.values()))
            if isinstance(first_param, torch.Tensor) and first_param.numel() > 0:
                return history_frames.mean() * 0.0
        dummy = torch.tensor(1.0, requires_grad=True, device=self.device)
        return history_frames.mean() * dummy

    def release_cuda(self) -> None:
        self.predictor.cpu()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self, device: torch.device | str | None = None) -> None:
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.predictor.to(dev)


# Backward compatibility
CtrlWorldLikeModel = CtrlWorldModel


if __name__ == "__main__":
    from src.vla.pi05_droid import Pi05DroidPolicy

    print("Testing CtrlWorldModel rollout...")
    wm = CtrlWorldModel()
    wm.to_device("cpu")
    policy = Pi05DroidPolicy("heuristic-fallback")
    initial_obs = {
        "obs": torch.randn(1, 3, 256, 256),
        "state": torch.randn(1, 14),
        "instruction": "test",
    }
    trajs = wm.rollout(initial_obs, policy, horizon=3, n_traj=1)
    print(f"Generated {len(trajs)} trajectories, steps={len(trajs[0]['steps'])}")
