import importlib.util
import json
import sys
import tempfile
import types
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
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.is_tensor = lambda _: False
    sys.modules["torch"] = torch_stub
sys.modules[EXPORTER_SPEC.name] = jepa_export
EXPORTER_SPEC.loader.exec_module(jepa_export)


class BridgeQualityGateTests(unittest.TestCase):
    def test_bridge_reads_manifest_trajectories_file_for_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes_dir = root / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    "images": [[[0, 0, 0]]],
                    "state": [[0.0, 0.0, 0.0, 0.0]],
                    "actions": [[0.0, 0.0, 0.0, 0.0]],
                    "language": "push",
                    "done": True,
                    "success": True,
                }
            ]
            (episodes_dir / "shard_0001.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            (root / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "export_mode": "cem_paired_push_v3",
                        "trajectories_file": "episodes",
                    }
                ),
                encoding="utf-8",
            )

            records = bridge_builder._read_records_from_manifest(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].get("language"), "push")

    def test_bridge_reads_manifest_trajectories_file_for_pt_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "episodes_0001.pt").write_bytes(b"placeholder")
            (root / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "export_mode": "cem_paired_push_v3",
                        "trajectories_file": "episodes",
                    }
                ),
                encoding="utf-8",
            )
            torch_stub = types.ModuleType("torch")
            torch_stub.load = lambda *args, **kwargs: [
                {
                    "images": [[[0, 0, 0]]],
                    "state": [[0.0, 0.0, 0.0, 0.0]],
                    "actions": [[0.0, 0.0, 0.0, 0.0]],
                    "language": "pt-shard",
                    "done": True,
                    "success": True,
                }
            ]
            with mock.patch.dict(sys.modules, {"torch": torch_stub}):
                records = bridge_builder._read_records_from_manifest(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].get("language"), "pt-shard")

    def test_bridge_manifest_target_without_records_fails_fast_in_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "src"
            root.mkdir(parents=True, exist_ok=True)
            (root / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "export_mode": "cem_paired_push_v3",
                        "trajectories_file": "episodes",
                    }
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmp) / "bridge"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "bridge_builder.py",
                    "--jepa-source",
                    str(root),
                    "--out-dir",
                    str(out_dir),
                ],
            ):
                rc = bridge_builder.main()
            self.assertNotEqual(rc, 0)
            self.assertFalse((out_dir / "bridge_summary.json").exists())

    def test_split_manifest_preview_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "manifest.json"
            records = []
            for i in range(12):
                records.append(
                    {
                        "meta": {"pair_key": f"pair-{i}"},
                        "action_chunk": [[0.0, 0.0, 0.0, 0.0] for _ in range(i + 1)],
                    }
                )
            bridge_builder._write_split_manifest_preview(out, records)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload.keys()),
                {"record_count", "sample_pair_keys", "sample_step_counts"},
            )
            self.assertEqual(payload["record_count"], 12)
            self.assertEqual(len(payload["sample_pair_keys"]), 10)
            self.assertEqual(len(payload["sample_step_counts"]), 10)
            self.assertEqual(payload["sample_pair_keys"][0], "pair-0")
            self.assertEqual(payload["sample_step_counts"][0], 1)

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
