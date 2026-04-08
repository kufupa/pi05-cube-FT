import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


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
            execution_policy="cem_primary",
        )
        self.assertEqual(out["policy_source"], "cem_mpc_wm")
        np.testing.assert_allclose(
            np.asarray(out["action_executed"], dtype=np.float32),
            np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )

    def test_smolvla_primary_action_selection(self):
        out = jepa_export._select_executed_action(
            obs=np.zeros(16, dtype=np.float32),
            env=DummyEnv(np.zeros((8, 8, 3), dtype=np.uint8)),
            action_wm_cem_first=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            action_smolvla_raw=np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            env_action_dim=4,
            wm_available=True,
            execution_policy="smolvla_primary",
        )
        self.assertEqual(out["policy_source"], "smolvla")
        np.testing.assert_allclose(
            np.asarray(out["action_executed"], dtype=np.float32),
            np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )

    def test_smolvla_selected_when_cem_missing(self):
        out = jepa_export._select_executed_action(
            obs=np.zeros(16, dtype=np.float32),
            env=DummyEnv(np.zeros((8, 8, 3), dtype=np.uint8)),
            action_wm_cem_first=None,
            action_smolvla_raw=np.array([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
            env_action_dim=4,
            wm_available=True,
            execution_policy="cem_primary",
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
            execution_policy="cem_primary",
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
            execution_policy="cem_primary",
        )
        self.assertEqual(out["policy_source"], "heuristic")

    def test_cem_first_action_raises_when_all_unrolls_fail(self):
        class _FailingModel:
            def unroll(self, *args, **kwargs):
                raise RuntimeError("forced unroll failure")

        with self.assertRaises(RuntimeError):
            jepa_export.cem_first_action(
                model=_FailingModel(),
                z=torch.zeros((1, 1, 8), dtype=torch.float32),
                action_dim=4,
                horizon=2,
                pop_size=2,
                cem_iters=2,
                device=torch.device("cpu"),
                rng=np.random.default_rng(0),
            )

    def test_cem_first_action_truncates_latents_when_full_export_disabled(self):
        latent_dim = 300

        class _LatentModel:
            def unroll(self, *args, **kwargs):
                return torch.arange(latent_dim, dtype=torch.float32).view(1, 1, latent_dim)

        _, payload = jepa_export.cem_first_action(
            model=_LatentModel(),
            z=torch.zeros((1, 1, 8), dtype=torch.float32),
            action_dim=4,
            horizon=1,
            pop_size=1,
            cem_iters=1,
            device=torch.device("cpu"),
            rng=np.random.default_rng(0),
            full_latents_export=False,
        )
        self.assertEqual(len(payload["latent_pred"]), 256)
        self.assertEqual(int(payload["latent_pred_dim"]), latent_dim)

    def test_cem_first_action_keeps_full_latents_when_enabled(self):
        latent_dim = 300

        class _LatentModel:
            def unroll(self, *args, **kwargs):
                return torch.arange(latent_dim, dtype=torch.float32).view(1, 1, latent_dim)

        _, payload = jepa_export.cem_first_action(
            model=_LatentModel(),
            z=torch.zeros((1, 1, 8), dtype=torch.float32),
            action_dim=4,
            horizon=1,
            pop_size=1,
            cem_iters=1,
            device=torch.device("cpu"),
            rng=np.random.default_rng(0),
            full_latents_export=True,
        )
        self.assertEqual(len(payload["latent_pred"]), latent_dim)
        self.assertEqual(int(payload["latent_pred_dim"]), latent_dim)


if __name__ == "__main__":
    unittest.main()
