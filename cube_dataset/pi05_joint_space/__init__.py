"""π0.5 joint-target cube env registration."""

from __future__ import annotations

from cube_dataset.pi05_joint_space.cube_env_joint_target import CubeEnvJointTargetDelta


def register_joint_target_envs() -> None:
    """Register cube-single-joint-target-v0 if not already registered."""
    from gymnasium.envs.registration import register, registry

    env_id = "cube-single-joint-target-v0"
    if env_id in registry:
        return
    register(
        id=env_id,
        entry_point="cube_dataset.pi05_joint_space.cube_env_joint_target:CubeEnvJointTargetDelta",
        max_episode_steps=200,
        kwargs=dict(env_type="single", joint_scale=0.05),
    )


__all__ = ["CubeEnvJointTargetDelta", "register_joint_target_envs"]
