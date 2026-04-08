import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "jepa_cem_paired_pushv3_export.py"
SPEC = importlib.util.spec_from_file_location("jepa_export", MODULE)
jepa_export = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = jepa_export
SPEC.loader.exec_module(jepa_export)


class _DummyActionSpace:
    shape = (4,)


class DummyEnv:
    def __init__(self, frame: np.ndarray):
        self._frame = frame
        self.action_space = _DummyActionSpace()

    def render(self):
        return self._frame


class ExporterContiguousTests(unittest.TestCase):
    def test_negative_stride_input_becomes_contiguous(self):
        frame = np.zeros((12, 12, 3), dtype=np.uint8)[:, :, ::-1]
        out = jepa_export._as_contiguous_rgb_uint8(frame)
        self.assertTrue(out.flags["C_CONTIGUOUS"])
        self.assertEqual(out.shape, (12, 12, 3))

    def test_collect_step_image_uses_render_fallback(self):
        env = DummyEnv(np.ones((8, 8, 3), dtype=np.uint8) * 255)
        img = jepa_export._collect_step_image(obs={}, env=env)
        self.assertEqual(img.shape, (8, 8, 3))
        self.assertTrue(img.flags["C_CONTIGUOUS"])

    def test_cem_primary_action_selection(self):
        out = jepa_export._select_executed_action(
            obs=np.zeros(16, dtype=np.float32),
            env=DummyEnv(np.zeros((8, 8, 3), dtype=np.uint8)),
            action_wm_cem_first=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            action_smolvla_raw=np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
            env_action_dim=4,
            wm_available=True,
        )
        self.assertEqual(out["policy_source"], "cem_mpc_wm")
        self.assertEqual(len(out["action_executed"]), 4)

    def test_smolvla_selected_when_cem_missing(self):
        out = jepa_export._select_executed_action(
            obs=np.zeros(16, dtype=np.float32),
            env=DummyEnv(np.zeros((8, 8, 3), dtype=np.uint8)),
            action_wm_cem_first=None,
            action_smolvla_raw=np.array([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
            env_action_dim=4,
            wm_available=True,
        )
        self.assertEqual(out["policy_source"], "smolvla")

    def test_heuristic_fallback_policy_source(self):
        out = jepa_export._select_executed_action(
            obs=np.zeros(16, dtype=np.float32),
            env=DummyEnv(np.zeros((8, 8, 3), dtype=np.uint8)),
            action_wm_cem_first=None,
            action_smolvla_raw=None,
            env_action_dim=4,
            wm_available=True,
        )
        self.assertEqual(out["policy_source"], "heuristic_fallback")

    def test_heuristic_policy_source_when_wm_absent(self):
        out = jepa_export._select_executed_action(
            obs=np.zeros(16, dtype=np.float32),
            env=DummyEnv(np.zeros((8, 8, 3), dtype=np.uint8)),
            action_wm_cem_first=None,
            action_smolvla_raw=None,
            env_action_dim=4,
            wm_available=False,
        )
        self.assertEqual(out["policy_source"], "heuristic")


if __name__ == "__main__":
    unittest.main()
