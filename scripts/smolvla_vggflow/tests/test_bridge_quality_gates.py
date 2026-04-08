import importlib.util
import json
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

EXPORTER_MODULE = Path(__file__).resolve().parents[1] / "jepa_cem_paired_pushv3_export.py"
EXPORTER_SPEC = importlib.util.spec_from_file_location("jepa_export", EXPORTER_MODULE)
jepa_export = importlib.util.module_from_spec(EXPORTER_SPEC)
assert EXPORTER_SPEC and EXPORTER_SPEC.loader
sys.modules[EXPORTER_SPEC.name] = jepa_export
EXPORTER_SPEC.loader.exec_module(jepa_export)


class BridgeQualityGateTests(unittest.TestCase):
    def test_blank_images_are_rejected(self):
        metrics = {
            "image_nonblank_ratio": 0.0,
            "heuristic_fallback_episode_ratio": 1.0,
            "action_std_mean": 0.0,
        }
        with self.assertRaises(RuntimeError):
            bridge_builder._enforce_quality_gates(
                metrics,
                min_image_coverage=0.95,
                max_heuristic_ratio=0.1,
                min_action_std=0.02,
            )

    def test_exporter_quality_gate_rejects_policy_error_spike(self):
        metrics = {
            "wm_step_error_rate": 0.0,
            "policy_exec_error_rate": 0.8,
            "episodes_with_images": 2,
            "total_episodes": 2,
            "heuristic_fallback_episode_ratio": 0.0,
        }
        with self.assertRaises(RuntimeError):
            jepa_export._enforce_export_quality_gates(
                metrics,
                max_wm_error_rate=0.05,
                max_policy_error_rate=0.05,
                require_images=True,
                max_heuristic_ratio=0.1,
            )

    def test_bridge_split_writer_handles_empty_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_root = Path(tmp) / "train"
            out_parent = Path(tmp)
            frames = bridge_builder._write_lerobot_v21_smolvla_split(
                split_root=split_root,
                split_name="train",
                records=[],
                source_files=0,
                out_parent=out_parent,
                task_default="push the puck to the goal",
            )
            self.assertEqual(frames, 0)
            info_path = split_root / "meta" / "info.json"
            self.assertTrue(info_path.is_file())
            info = json.loads(info_path.read_text(encoding="utf-8"))
            self.assertEqual(int(info.get("total_episodes", -1)), 0)

    def test_compute_quality_metrics_treats_invalid_images_as_blank(self):
        records = [
            {
                "images": [None, {"not": "an_image"}, "invalid-image"],
                "action_chunk": [[0.0, 0.0, 0.0, 0.0]],
                "meta": {},
            }
        ]
        metrics = bridge_builder._compute_quality_metrics(records)
        self.assertEqual(float(metrics["image_nonblank_ratio"]), 0.0)

    def test_bridge_builder_strict_out_dir_rejects_existing_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bridge"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "existing.txt").write_text("stale", encoding="utf-8")
            with mock.patch.object(
                sys,
                "argv",
                [
                    "bridge_builder.py",
                    "--out-dir",
                    str(out_dir),
                    "--fail-on-path-reuse",
                    "1",
                ],
            ):
                rc = bridge_builder.main()
            self.assertNotEqual(rc, 0)

    def test_bridge_builder_strict_out_dir_allows_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bridge"
            out_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(
                sys,
                "argv",
                [
                    "bridge_builder.py",
                    "--out-dir",
                    str(out_dir),
                    "--fail-on-path-reuse",
                    "1",
                ],
            ):
                rc = bridge_builder.main()
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "bridge_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
