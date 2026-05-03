"""Tests for rlcade.notebook HTTP server helpers."""

import socket
import urllib.request
from pathlib import Path

import pytest

from rlcade import notebook as nb


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _cleanup_servers():
    yield
    for server in nb._servers.values():
        server.shutdown()
        server.server_close()
    nb._servers.clear()


class TestFindProjectRoot:
    def test_points_to_dir_three_levels_above_module(self):
        # Module lives at <repo>/rlcade/notebook/__init__.py; repo root is three
        # parents up (notebook → rlcade → repo). This test guards against the
        # earlier bug where it pointed to rlcade/ instead.
        expected = Path(nb.__file__).resolve().parent.parent.parent
        if (expected / "pkg").exists():
            assert nb._find_project_root() == expected
        else:
            with pytest.raises(FileNotFoundError, match="pkg"):
                nb._find_project_root()

    def test_raises_when_pkg_missing(self, monkeypatch, tmp_path):
        fake_module = tmp_path / "repo" / "rlcade" / "notebook" / "__init__.py"
        fake_module.parent.mkdir(parents=True)
        fake_module.write_text("")
        monkeypatch.setattr(nb, "__file__", str(fake_module))
        with pytest.raises(FileNotFoundError, match="pkg"):
            nb._find_project_root()


class TestEnsureIndexHtml:
    def test_creates_when_missing(self, tmp_path):
        nb._ensure_index_html(tmp_path)
        assert (tmp_path / "index.html").exists()
        assert "<canvas" in (tmp_path / "index.html").read_text()

    def test_preserves_existing(self, tmp_path):
        (tmp_path / "index.html").write_text("custom")
        nb._ensure_index_html(tmp_path)
        assert (tmp_path / "index.html").read_text() == "custom"


class TestStartServer:
    def test_repeat_calls_reuse_same_server(self, tmp_path):
        port = _free_port()
        first = nb._start_server(tmp_path, port)
        second = nb._start_server(tmp_path, port)
        assert first is second

    def test_repeat_calls_do_not_raise_address_in_use(self, tmp_path):
        port = _free_port()
        nb._start_server(tmp_path, port)
        # Pre-fix, this second bind would raise OSError(EADDRINUSE).
        nb._start_server(tmp_path, port)

    def test_different_ports_create_different_servers(self, tmp_path):
        a = nb._start_server(tmp_path, _free_port())
        b = nb._start_server(tmp_path, _free_port())
        assert a is not b

    def test_different_roots_create_different_servers(self, tmp_path):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        a = nb._start_server(root_a, _free_port())
        b = nb._start_server(root_b, _free_port())
        assert a is not b

    def test_serves_files_from_root(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hi there")
        port = _free_port()
        nb._start_server(tmp_path, port)
        with urllib.request.urlopen(f"http://localhost:{port}/hello.txt") as resp:
            assert resp.read() == b"hi there"
