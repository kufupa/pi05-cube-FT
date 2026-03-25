import torch
import numpy as np
from pathlib import Path
import tfrecord
import cv2


# DROID 100 RLDS (flat arrays per episode):
# - steps/observation/joint_position  -> (T*7,) observational joint pos
# - steps/observation/gripper_position -> (T,) gripper opening
# - steps/observation/exterior_image_1_left / wrist_image_left -> (T,) JPEG bytes
# - steps/action_dict/joint_velocity + gripper_position -> executed command as (T*7,) + (T,)
#   Concatenated as expert_actions_8 [T,8] for OpenPI / offline alignment (velocity + gripper pos).
# If only steps/observation/robot_state [T*14] exists elsewhere, split is documented as:
#   robot_state[:, :7] = joint positions, robot_state[:, 7:8] = gripper (fallback only).
#
# steps/action is [T,7] (not 8); it does not match joint_velocity byte-for-byte — use action_dict for 8-D expert.


def _decode_instruction(raw_episode, T):
    for key in (
        "steps/language_instruction",
        "steps/language_instruction_2",
        "steps/language_instruction_3",
    ):
        inst_raw = raw_episode.get(key)
        if inst_raw is None or len(inst_raw) == 0:
            continue
        for t in range(min(T, len(inst_raw))):
            row = inst_raw[t]
            if isinstance(row, bytes):
                s = row.decode("utf-8", errors="replace").strip()
            else:
                s = str(row).strip()
            if s:
                return s
    return ""


def get_droid_dataset(task_name="stacking", split="train", max_episodes=20, debug=False):
    """Load real DROID RLDS episodes natively using tfrecord package."""
    data_dir = Path("data/droid_sample/droid_100")
    episodes = []

    # We will grab any TFRecords we can find up to max_episodes
    for episode_path in data_dir.glob("**/*.tfrecord*"):
        if len(episodes) >= max_episodes:
            break

        try:
            loader = tfrecord.tfrecord_loader(str(episode_path), None, None)

            for raw_episode in loader:
                if len(episodes) >= max_episodes:
                    break

                flat_actions = raw_episode.get("steps/action")
                if flat_actions is None:
                    continue
                actions = torch.from_numpy(flat_actions.reshape(-1, 7)).float()
                T = actions.shape[0]

                image_bytes_seq = raw_episode.get("steps/observation/exterior_image_1_left")
                if image_bytes_seq is None:
                    image_bytes_seq = raw_episode.get("steps/observation/wrist_image_left")
                if image_bytes_seq is None:
                    continue

                jp_obs = raw_episode.get("steps/observation/joint_position")
                gp_obs = raw_episode.get("steps/observation/gripper_position")
                state_raw = raw_episode.get("steps/observation/robot_state")

                if jp_obs is not None and gp_obs is not None:
                    joint_obs = torch.from_numpy(jp_obs.reshape(T, 7).astype(np.float32))
                    grip_obs = torch.from_numpy(np.asarray(gp_obs, dtype=np.float32).reshape(T, 1))
                elif state_raw is not None:
                    rs = state_raw.reshape(T, 14).astype(np.float32)
                    joint_obs = torch.from_numpy(rs[:, :7])
                    grip_obs = torch.from_numpy(rs[:, 7:8])
                    if debug:
                        print(
                            "[droid] Using robot_state fallback split: [:7] joints, [7:8] gripper "
                            f"(episode T={T})"
                        )
                else:
                    joint_obs = torch.zeros(T, 7, dtype=torch.float32)
                    grip_obs = torch.zeros(T, 1, dtype=torch.float32)
                    if debug:
                        print(f"[droid] Warning: no observation joints; zero-filled (T={T})")

                images = []
                for img_bytes in image_bytes_seq:
                    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    img_np = cv2.resize(img_np, (256, 256))  # resize to uniform shape for pipeline
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                    images.append(img_tensor)

                images_tensor = torch.stack(images)  # [T, 3, H, W]

                rewards = raw_episode.get("steps/reward", [])
                success = float(rewards[-1]) if len(rewards) > 0 else 0.0

                instruction = _decode_instruction(raw_episode, T)

                state = torch.zeros(T, 14, dtype=torch.float32)
                state[:, :7] = joint_obs
                state[:, -1] = grip_obs.squeeze(-1)

                jv = raw_episode.get("steps/action_dict/joint_velocity")
                gpa = raw_episode.get("steps/action_dict/gripper_position")
                if jv is not None and gpa is not None:
                    expert8 = np.concatenate(
                        [jv.reshape(T, 7).astype(np.float32), gpa.reshape(T, 1).astype(np.float32)],
                        axis=-1,
                    )
                    expert_actions_8 = torch.from_numpy(expert8)
                else:
                    expert_actions_8 = torch.zeros(T, 8, dtype=torch.float32)
                    expert_actions_8[:, :7] = actions
                    expert_actions_8[:, -1:] = grip_obs

                wrist_bytes_seq = raw_episode.get("steps/observation/wrist_image_left")

                episodes.append({
                    "obs": images_tensor,
                    "actions": actions,
                    "state": state,
                    "joint_position": joint_obs,
                    "gripper_position": grip_obs,
                    "exterior_jpeg": image_bytes_seq,
                    "wrist_jpeg": wrist_bytes_seq,
                    "expert_actions_8": expert_actions_8,
                    "success": success,
                    "instruction": instruction,
                })
                print(f"Loaded episode: {T} steps | task: {instruction!r}")
                
        except Exception as e:
            print(f"Error reading {episode_path}: {e}")
            continue
            
    return episodes
