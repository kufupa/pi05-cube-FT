"""Joint-target-delta control for OGBench CubeEnv (6 arm + gripper in [0,1])."""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box

from ogbench.manipspace.envs.cube_env import CubeEnv


class CubeEnvJointTargetDelta(CubeEnv):
    """UR5 arm: delta joint targets from clipped [-1,1]^6 scaled by joint_scale; gripper ctrl = 255 * [0,1]."""

    def __init__(
        self,
        env_type: str = "single",
        permute_blocks: bool = True,
        joint_scale: float = 0.05,
        **kwargs,
    ):
        self._joint_scale = float(joint_scale)
        self._cached_jnt_limits = False
        self._jnt_low: np.ndarray | None = None
        self._jnt_high: np.ndarray | None = None
        super().__init__(env_type=env_type, permute_blocks=permute_blocks, **kwargs)

    @property
    def action_space(self) -> Box:
        low = np.array([-1.0] * 6 + [0.0], dtype=np.float32)
        high = np.array([1.0] * 7, dtype=np.float32)
        return Box(low=low, high=high, shape=(7,), dtype=np.float32)

    def _cache_joint_limits(self) -> None:
        if self._cached_jnt_limits:
            return
        lows: list[float] = []
        highs: list[float] = []
        for jid in self._arm_joint_ids:
            r = self.model.jnt_range[jid]
            lows.append(float(r[0]))
            highs.append(float(r[1]))
        self._jnt_low = np.asarray(lows, dtype=np.float64)
        self._jnt_high = np.asarray(highs, dtype=np.float64)
        self._cached_jnt_limits = True

    def set_control(self, action) -> None:
        """Write PD position targets; do not use EE IK path."""
        self._cache_joint_limits()
        assert self._jnt_low is not None and self._jnt_high is not None
        a = np.asarray(action, dtype=np.float64).reshape(7)
        arm = np.clip(a[:6], -1.0, 1.0)
        g = float(np.clip(a[6], 0.0, 1.0))
        q = self._data.qpos[self._arm_joint_ids].astype(np.float64)
        delta = self._joint_scale * arm
        q_target = np.clip(q + delta, self._jnt_low, self._jnt_high)
        self._data.ctrl[self._arm_actuator_ids] = q_target
        self._data.ctrl[self._gripper_actuator_ids] = 255.0 * g
