"""IO-level round-trip tests for save_safetensors / load_safetensors.

Mirrors tests/checkpoint/test_checkpoint.py but for the safetensors codec.
"""

import json
import struct

import pytest
import torch

from rlcade.checkpoint.checkpoint import Checkpoint
from rlcade.checkpoint.safetensors import load_safetensors, save_safetensors


def _make_state(step=42):
    return {
        "step": step,
        "actor": {"layer.weight": torch.randn(3, 3), "layer.bias": torch.randn(3)},
        "critic": {"head.weight": torch.randn(1, 4)},
        "optimizer": {"state": "ignored"},  # non-tensor dict — must be dropped
        "grad_scaler": None,
    }


class TestSafetensorsLocal:
    def test_save_load_roundtrip(self, tmp_path):
        url = str(tmp_path / "model.safetensors")
        original = _make_state(step=42)
        save_safetensors(original, url, step=42)

        loaded, step = load_safetensors(url, device=torch.device("cpu"))
        assert step == 42
        assert set(loaded.keys()) == {"actor", "critic"}
        for name in ("actor", "critic"):
            for k, v in original[name].items():
                assert torch.equal(v, loaded[name][k]), f"{name}.{k} mismatch"

    def test_step_metadata_preserved(self, tmp_path):
        url = str(tmp_path / "model.safetensors")
        save_safetensors(_make_state(), url, step=12345)
        _, step = load_safetensors(url, device=torch.device("cpu"))
        assert step == 12345

    def test_non_tensor_entries_dropped(self, tmp_path):
        url = str(tmp_path / "model.safetensors")
        state = {
            "step": 1,
            "actor": {"w": torch.randn(2, 2)},
            "optimizer": {"state": "junk"},
            "grad_scaler": None,
        }
        save_safetensors(state, url, step=1)
        loaded, _ = load_safetensors(url, device=torch.device("cpu"))
        assert set(loaded.keys()) == {"actor"}

    def test_format_mismatch_raises(self, tmp_path):
        url = str(tmp_path / "model.safetensors")
        save_safetensors(_make_state(), url, step=0)
        # Rewrite the metadata format string in place.
        with open(url, "rb") as f:
            blob = f.read()
        (header_len,) = struct.unpack("<Q", blob[:8])
        header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
        header["__metadata__"]["format"] = "garbage-v0"
        new_header = json.dumps(header).encode("utf-8")
        new_blob = struct.pack("<Q", len(new_header)) + new_header + blob[8 + header_len :]
        with open(url, "wb") as f:
            f.write(new_blob)

        with pytest.raises(ValueError, match="unrecognized format"):
            load_safetensors(url, device=torch.device("cpu"))

    def test_load_routes_to_device(self, tmp_path):
        url = str(tmp_path / "model.safetensors")
        save_safetensors(_make_state(), url, step=0)
        loaded, _ = load_safetensors(url, device=torch.device("cpu"))
        for sd in loaded.values():
            for v in sd.values():
                assert v.device.type == "cpu"

    def test_nested_module_fqn_preserved(self, tmp_path):
        """Only the FIRST '.' is the model boundary; nested FQNs survive."""
        url = str(tmp_path / "model.safetensors")
        state = {
            "step": 0,
            "actor": {
                "conv.0.weight": torch.randn(2, 2),
                "conv.1.bias": torch.randn(2),
            },
        }
        save_safetensors(state, url, step=0)
        loaded, _ = load_safetensors(url, device=torch.device("cpu"))
        assert set(loaded["actor"].keys()) == {"conv.0.weight", "conv.1.bias"}

    def test_save_creates_parent_dirs(self, tmp_path):
        url = str(tmp_path / "a" / "b" / "model.safetensors")
        save_safetensors(_make_state(), url, step=0)
        assert Checkpoint(url).exists()


class TestSafetensorsS3:
    def test_save_load_roundtrip(self, s3_url):
        url = f"{s3_url}/model.safetensors"
        ckpt = Checkpoint(url)
        try:
            save_safetensors(_make_state(), url, step=7)
            loaded, step = load_safetensors(url, device=torch.device("cpu"))
            assert step == 7
            assert "actor" in loaded
        finally:
            ckpt.delete()
