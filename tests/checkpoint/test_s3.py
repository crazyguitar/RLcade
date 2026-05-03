from unittest import mock

import pytest

from rlcade.checkpoint import S3FileSystem


def _make_fs(s3_url: str) -> S3FileSystem:
    """Parse s3://bucket/prefix and return an S3FileSystem."""
    parts = s3_url[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return S3FileSystem(bucket, prefix)


class TestS3FileSystem:
    @pytest.fixture()
    def fs(self, s3_url):
        return _make_fs(s3_url)

    def test_factory_returns_s3(self, fs):
        assert isinstance(fs, S3FileSystem)

    def test_write_and_read(self, fs):
        fs.write("test_rw.txt", b"hello s3")
        assert fs.read("test_rw.txt") == b"hello s3"

    def test_exists(self, fs):
        fs.write("test_exists.txt", b"data")
        assert fs.exists("test_exists.txt")
        assert not fs.exists("no_such_key_abc123.txt")

    def test_open_read(self, fs):
        fs.write("test_open.bin", b"\xde\xad")
        with fs.open("test_open.bin", "rb") as f:
            assert f.read() == b"\xde\xad"

    def test_open_write(self, fs):
        with fs.open("test_open_w.bin", "wb") as f:
            f.write(b"via open")
        assert fs.read("test_open_w.bin") == b"via open"

    def test_list(self, fs):
        fs.write("test_list/a.txt", b"a")
        fs.write("test_list/b.txt", b"b")
        result = fs.list("test_list")
        assert "test_list/a.txt" in result
        assert "test_list/b.txt" in result

    def test_repr(self, fs):
        assert "S3FileSystem" in repr(fs)


class TestS3FileSystemExistsErrors:
    """exists() must distinguish 'not found' from permission/server errors."""

    @pytest.fixture()
    def fs(self):
        with mock.patch("boto3.client") as mock_client, mock.patch("boto3.Session"):
            mock_client.return_value = mock.MagicMock()
            yield S3FileSystem("my-bucket")

    def _client_error(self, *, status: int, code: str):
        from botocore.exceptions import ClientError

        return ClientError(
            {"Error": {"Code": code, "Message": ""}, "ResponseMetadata": {"HTTPStatusCode": status}},
            "HeadObject",
        )

    def test_returns_false_on_404(self, fs):
        fs._s3.head_object.side_effect = self._client_error(status=404, code="404")
        assert fs.exists("missing.pt") is False

    def test_returns_false_on_no_such_key(self, fs):
        fs._s3.head_object.side_effect = self._client_error(status=404, code="NoSuchKey")
        assert fs.exists("missing.pt") is False

    def test_raises_on_403(self, fs):
        from botocore.exceptions import ClientError

        fs._s3.head_object.side_effect = self._client_error(status=403, code="AccessDenied")
        with pytest.raises(ClientError):
            fs.exists("forbidden.pt")

    def test_raises_on_500(self, fs):
        from botocore.exceptions import ClientError

        fs._s3.head_object.side_effect = self._client_error(status=500, code="InternalError")
        with pytest.raises(ClientError):
            fs.exists("any.pt")


class TestS3FileSystemStreaming:
    """Test torch.save/load through S3FileSystem.open() streams."""

    @pytest.fixture()
    def fs(self, s3_url):
        return _make_fs(s3_url)

    def test_open_write_then_read_stream(self, fs):
        import torch

        state = {"step": 99, "w": torch.randn(4, 4)}
        with fs.open("stream_test.pt", "wb") as f:
            torch.save(state, f)
        with fs.open("stream_test.pt", "rb") as f:
            loaded = torch.load(f, weights_only=True)
        assert loaded["step"] == 99
        assert torch.equal(loaded["w"], state["w"])
