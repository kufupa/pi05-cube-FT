import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / "bridge_builder.py"
SPEC = importlib.util.spec_from_file_location("bridge_builder", MODULE)
bridge_builder = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = bridge_builder
SPEC.loader.exec_module(bridge_builder)


class BridgeWMHeavySplitTests(unittest.TestCase):
    def test_wm_completeness_score_uses_per_step_telemetry(self):
        episode = {
            "cem_plan": {
                "per_step": [
                    {
                        "cem_iterations": 2,
                        "latent_pred": [0.1, 0.2],
                        "planner_metadata": {},
                    },
                    {
                        "cem_iterations": 0,
                        "latent_pred": [],
                        "planner_metadata": {"wm_step_error": "boom"},
                    },
                    {
                        "cem_iterations": 0,
                        "latent_pred": [0.4],
                        "planner_metadata": {"wm_skipped": True},
                    },
                ]
            }
        }

        score = bridge_builder._compute_wm_completeness_score(episode)
        expected = (1.0 + 0.0 + (1.0 / 3.0)) / 3.0
        self.assertAlmostEqual(score, expected, places=6)

    def test_wm_heavy_split_is_deterministic_and_top_fraction(self):
        records = [
            {"pair_key": "pair-c", "meta": {"wm_completeness_score": 0.95}},
            {"pair_key": "pair-a", "meta": {"wm_completeness_score": 0.90}},
            {"pair_key": "pair-b", "meta": {"wm_completeness_score": 0.20}},
            {"pair_key": "pair-z", "meta": {"wm_completeness_score": 0.1}},
        ]
        expected_margin = 0.30

        train_a, val_a, stats_a = bridge_builder._split_wm_heavy(
            records,
            val_ratio=0.50,
            score_margin=0.0,
        )
        train_b, val_b, stats_b = bridge_builder._split_wm_heavy(
            records,
            val_ratio=0.50,
            score_margin=0.0,
        )

        self.assertEqual([x["pair_key"] for x in train_a], [x["pair_key"] for x in train_b])
        self.assertEqual([x["pair_key"] for x in val_a], [x["pair_key"] for x in val_b])
        self.assertEqual(len(val_a), 2)
        self.assertEqual(stats_a["train_count"], len(train_a))
        self.assertEqual(stats_a["val_count"], len(val_a))
        self.assertAlmostEqual(stats_a["mean_score_train"], stats_b["mean_score_train"], places=8)
        self.assertAlmostEqual(stats_a["mean_score_val"], stats_b["mean_score_val"], places=8)
        self.assertGreaterEqual(
            stats_a["mean_score_val"] - stats_a["mean_score_train"],
            expected_margin,
        )

    def test_placeholder_summary_includes_wm_heavy_contract_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bridge"
            bridge_builder._write_placeholder_dataset(out_dir)
            summary = json.loads((out_dir / "bridge_summary.json").read_text(encoding="utf-8"))

        self.assertIn("split_policy", summary)
        self.assertIn("split_counts", summary)
        self.assertIn("wm_completeness_mean", summary)
        self.assertIn("wm_heavy_jepa_fraction", summary["split_policy"])
        self.assertIn("wm_score_margin", summary["split_policy"])
        self.assertIn("wm_heavy_split_policy", summary)
        self.assertIn("mean_wm_score_train", summary)
        self.assertIn("mean_wm_score_val", summary)
        self.assertIn("n_train_episodes", summary)
        self.assertIn("n_val_episodes", summary)

    def test_toggle_env_alias_plan_name_is_supported(self):
        with mock.patch.dict(
            os.environ,
            {
                "SMOLVLA_BRIDGE_WM_HEAVY_SPLIT": "0",
                "SMOLVLA_BRIDGE_WM_HEAVY_SPLIT_ENABLED": "1",
            },
            clear=True,
        ):
            self.assertEqual(bridge_builder._resolve_wm_heavy_split_enabled_default(), 0)

        with mock.patch.dict(
            os.environ,
            {
                "SMOLVLA_BRIDGE_WM_HEAVY_SPLIT_ENABLED": "0",
            },
            clear=True,
        ):
            self.assertEqual(bridge_builder._resolve_wm_heavy_split_enabled_default(), 0)


if __name__ == "__main__":
    unittest.main()
