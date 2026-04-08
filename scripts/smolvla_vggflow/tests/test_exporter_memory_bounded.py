import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    torch.is_tensor = lambda _: False
    torch.Tensor = object
    torch.device = lambda *_args, **_kwargs: "cpu"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.save = lambda obj, path: Path(path).write_bytes(b"stub")
    sys.modules["torch"] = torch


MODULE = Path(__file__).resolve().parents[1] / "jepa_cem_paired_pushv3_export.py"
SPEC = importlib.util.spec_from_file_location("jepa_export_memory_bounded", MODULE)
jepa_export = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = jepa_export
SPEC.loader.exec_module(jepa_export)


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
    if torch.is_tensor(out):
        assert int(out.numel()) == 512
    else:
        arr = np.asarray(out)
        assert int(arr.size) == 512


def test_episode_shard_writer_writes_episode_file():
    with tempfile.TemporaryDirectory() as td:
        writer = jepa_export.EpisodeShardWriter(Path(td), episodes_per_shard=1)
        writer.write_episode(
            {
                "meta": {"episode_index": 0},
                "state": [[0.0, 1.0, 2.0]],
                "actions": [[0.0, 0.0, 0.0, 0.0]],
                "images": [],
            }
        )
        writer.finalize()
        written_files = [p for p in Path(td).iterdir() if p.is_file()]
        assert written_files
