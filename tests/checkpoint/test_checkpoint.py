# tests/checkpoint/test_checkpoint.py
import torch
import pytest

from rlcade.checkpoint.checkpoint import Checkpoint


class TestCheckpointLocal:
    def test_save_and_reader_roundtrip(self, tmp_path):
        url = str(tmp_path / "ckpt.pt")
        ckpt = Checkpoint(url)
        state = {"step": 42, "weights": torch.randn(3, 3)}
        ckpt.save(state)
        with ckpt.reader() as f:
            loaded = torch.load(f, weights_only=True)
        assert loaded["step"] == 42
        assert torch.equal(loaded["weights"], state["weights"])

    def test_writer_and_reader_roundtrip(self, tmp_path):
        url = str(tmp_path / "ckpt.pt")
        ckpt = Checkpoint(url)
        state = {"step": 7, "w": torch.randn(2, 2)}
        with ckpt.writer() as f:
            torch.save(state, f)
        with ckpt.reader() as f:
            loaded = torch.load(f, weights_only=True)
        assert loaded["step"] == 7
        assert torch.equal(loaded["w"], state["w"])

    def test_reader_returns_readable_stream(self, tmp_path):
        url = str(tmp_path / "ckpt.pt")
        ckpt = Checkpoint(url)
        ckpt.save({"step": 0})
        with ckpt.reader() as f:
            assert hasattr(f, "read")

    def test_writer_returns_writable_stream(self, tmp_path):
        url = str(tmp_path / "ckpt.pt")
        ckpt = Checkpoint(url)
        with ckpt.writer() as f:
            assert hasattr(f, "write")
            torch.save({"step": 0}, f)

    def test_exists_true(self, tmp_path):
        url = str(tmp_path / "ckpt.pt")
        ckpt = Checkpoint(url)
        ckpt.save({"step": 0})
        assert ckpt.exists()

    def test_exists_false(self, tmp_path):
        url = str(tmp_path / "missing.pt")
        ckpt = Checkpoint(url)
        assert not ckpt.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        url = str(tmp_path / "a" / "b" / "ckpt.pt")
        ckpt = Checkpoint(url)
        ckpt.save({"step": 0})
        assert ckpt.exists()

    def test_reader_nonexistent_raises(self, tmp_path):
        url = str(tmp_path / "nope.pt")
        ckpt = Checkpoint(url)
        with pytest.raises(FileNotFoundError):
            with ckpt.reader() as f:
                pass

    def test_url_property(self, tmp_path):
        url = str(tmp_path / "ckpt.pt")
        ckpt = Checkpoint(url)
        assert ckpt.url == url

    def test_writer_is_atomic_on_failure(self, tmp_path):
        """An exception mid-write must leave the final path untouched and clean up .tmp."""
        url = str(tmp_path / "ckpt.pt")
        ckpt = Checkpoint(url)
        ckpt.save({"step": 1})

        with pytest.raises(RuntimeError, match="boom"):
            with ckpt.writer() as f:
                f.write(b"partial")
                raise RuntimeError("boom")

        # Original file preserved.
        with ckpt.reader() as f:
            loaded = torch.load(f, weights_only=True)
        assert loaded["step"] == 1
        # .tmp cleaned up.
        assert not (tmp_path / "ckpt.pt.tmp").exists()

    def test_writer_atomic_first_write_leaves_no_file_on_failure(self, tmp_path):
        """If the very first write fails, no file (partial or otherwise) should exist."""
        url = str(tmp_path / "ckpt.pt")
        ckpt = Checkpoint(url)

        with pytest.raises(RuntimeError):
            with ckpt.writer() as f:
                f.write(b"partial")
                raise RuntimeError("boom")

        assert not (tmp_path / "ckpt.pt").exists()
        assert not (tmp_path / "ckpt.pt.tmp").exists()


class TestCheckpointS3:
    def test_save_and_reader_roundtrip(self, s3_url):
        url = f"{s3_url}/ckpt.pt"
        ckpt = Checkpoint(url)
        state = {"step": 10, "w": torch.randn(2, 2)}
        ckpt.save(state)
        with ckpt.reader() as f:
            loaded = torch.load(f, weights_only=True)
        assert loaded["step"] == 10
        assert torch.equal(loaded["w"], state["w"])

    def test_exists(self, s3_url):
        url = f"{s3_url}/ckpt_exists.pt"
        ckpt = Checkpoint(url)
        ckpt.delete()
        assert not ckpt.exists()
        ckpt.save({"step": 0})
        assert ckpt.exists()
        ckpt.delete()
