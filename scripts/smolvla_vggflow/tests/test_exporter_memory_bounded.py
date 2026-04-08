import importlib.util
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def _build_torch_import_stub() -> types.ModuleType:
    torch_stub = types.ModuleType("torch")
    torch_stub.is_tensor = lambda _: False
    torch_stub.Tensor = object
    torch_stub.device = lambda *_args, **_kwargs: "cpu"
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)

    def _save(obj, path):
        Path(path).write_bytes(pickle.dumps(obj))

    def _load(path):
        return pickle.loads(Path(path).read_bytes())

    torch_stub.save = _save
    torch_stub.load = _load
    return torch_stub


MODULE = Path(__file__).resolve().parents[1] / "jepa_cem_paired_pushv3_export.py"
SPEC = importlib.util.spec_from_file_location("jepa_export_memory_bounded", MODULE)
jepa_export = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader

_original_torch_module = sys.modules.get("torch")
if _original_torch_module is None:
    sys.modules["torch"] = _build_torch_import_stub()
try:
    sys.modules[SPEC.name] = jepa_export
    SPEC.loader.exec_module(jepa_export)
finally:
    if _original_torch_module is None:
        del sys.modules["torch"]
    else:
        sys.modules["torch"] = _original_torch_module


def test_encode_image_payload_returns_uint8_array():
    frame = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)
    out = jepa_export._encode_image_payload(frame)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)
    assert out.flags["C_CONTIGUOUS"]


def test_encode_latent_payload_not_python_list():
    latent = np.arange(512, dtype=np.float32)
    out = jepa_export._encode_latent_payload(latent, full_latents_export=True)
    assert not isinstance(out, list)
    if jepa_export.torch.is_tensor(out):
        assert int(out.numel()) == 512
    else:
        arr = np.asarray(out)
        assert int(arr.size) == 512


def test_encode_latent_payload_truncates_when_full_export_disabled():
    latent = np.arange(512, dtype=np.float32)
    out = jepa_export._encode_latent_payload(latent, full_latents_export=False)
    assert not isinstance(out, list)
    arr = np.asarray(out)
    assert int(arr.size) == 256
    np.testing.assert_array_equal(arr, latent[:256])


def test_encode_latent_payload_torch_branch_not_python_list():
    class _FakeTensor:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *_shape):
            self._values = self._values.reshape(-1)
            return self

        def numel(self):
            return int(self._values.size)

        def __getitem__(self, item):
            return _FakeTensor(self._values[item])

    original_torch = jepa_export.torch
    jepa_export.torch = types.SimpleNamespace(is_tensor=lambda value: isinstance(value, _FakeTensor))
    try:
        out = jepa_export._encode_latent_payload(_FakeTensor(np.arange(300, dtype=np.float32)), full_latents_export=False)
    finally:
        jepa_export.torch = original_torch

    assert not isinstance(out, list)
    assert isinstance(out, _FakeTensor)
    assert int(out.numel()) == 256


def test_episode_shard_writer_writes_episode_file():
    with tempfile.TemporaryDirectory() as td:
        writer = jepa_export.EpisodeShardWriter(Path(td), episodes_per_shard=1)
        episode = {
            "meta": {"episode_index": 0},
            "state": [[0.0, 1.0, 2.0]],
            "actions": [[0.0, 0.0, 0.0, 0.0]],
            "images": [],
        }
        file_path = writer.write_episode(episode)
        assert isinstance(file_path, Path)
        assert file_path.is_file()
        written_files = writer.finalize()
        assert written_files == [file_path]
        restored = jepa_export.torch.load(file_path)
        assert isinstance(restored, dict)
        assert restored["meta"]["episode_index"] == 0


def test_episode_shard_writer_flush_writes_per_episode_files():
    with tempfile.TemporaryDirectory() as td:
        writer = jepa_export.EpisodeShardWriter(Path(td), episodes_per_shard=2)
        assert writer.write_episode({"meta": {"episode_index": 0}}) is None
        flush_path = writer.write_episode({"meta": {"episode_index": 1}})
        assert isinstance(flush_path, Path)
        files = writer.finalize()
        assert len(files) == 2
        restored = [jepa_export.torch.load(path) for path in files]
        indices = sorted(int(item["meta"]["episode_index"]) for item in restored)
        assert indices == [0, 1]


def test_promote_episode_shards_replaces_existing_destination():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        staging_dir = root / ".episodes_staging"
        final_dir = root / "episodes"
        staging_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "old.txt").write_text("stale", encoding="utf-8")
        (staging_dir / "episode_000000.pt").write_bytes(b"fresh")

        jepa_export._promote_episode_shards(staging_dir, final_dir)

        assert not staging_dir.exists()
        assert final_dir.is_dir()
        assert (final_dir / "episode_000000.pt").is_file()
        assert not (final_dir / "old.txt").exists()


def test_cleanup_episode_shards_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        staging_dir = root / ".episodes_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        (staging_dir / "episode_000000.pt").write_bytes(b"fresh")

        jepa_export._cleanup_episode_shards(staging_dir)
        assert not staging_dir.exists()
        jepa_export._cleanup_episode_shards(staging_dir)
        assert not staging_dir.exists()


class ExporterMemoryBoundedTests(unittest.TestCase):
    def test_incremental_metrics_match_legacy_metrics(self):
        episodes = [
            {
                "meta": {"policy": "cem_primary"},
                "images": [np.zeros((4, 4, 3), dtype=np.uint8)],
                "cem_plan": {
                    "per_step": [
                        {
                            "policy_source": "cem_mpc_wm",
                            "planner_metadata": {"wm_step_error": False, "policy_exec_error": False},
                            "latent_pred_dim": 256,
                        },
                        {
                            "policy_source": "heuristic_fallback",
                            "planner_metadata": {"wm_step_error": True, "policy_exec_error": False},
                        },
                    ]
                },
            },
            {
                "meta": {"policy": "heuristic"},
                "images": [],
                "cem_plan": {
                    "per_step": [
                        {
                            "policy_source": "smolvla",
                            "planner_metadata": {"wm_step_error": False, "policy_exec_error": True},
                        }
                    ]
                },
            },
            {
                "meta": {"policy": "cem_primary"},
                "images": [np.ones((2, 2, 3), dtype=np.uint8)],
                "cem_plan": {"per_step": ["invalid-row", {"policy_source": "cem_mpc_wm", "planner_metadata": {}}]},
            },
            {
                "meta": {"policy": "heuristic_fallback"},
                "images": None,
                "cem_plan": {},
            },
        ]

        legacy_metrics = jepa_export._compute_export_quality_metrics(episodes)
        acc = jepa_export.ExportQualityAccumulator()
        for episode in episodes:
            acc.update(episode)
        incremental_metrics = acc.to_metrics()

        assert set(legacy_metrics.keys()) == set(incremental_metrics.keys())
        for key, legacy_value in legacy_metrics.items():
            assert abs(float(legacy_value) - float(incremental_metrics[key])) < 1e-12
