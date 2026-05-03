"""Checkpoint I/O -- stream-based reads/writes for local and S3 storage."""

from __future__ import annotations

import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


class Checkpoint:
    """Stream-based checkpoint I/O for a single URL (local path or ``s3://...``).

    Provides ``reader()`` / ``writer()`` context managers that yield file-like
    streams.  Agents own deserialization (``torch.load``) so they control
    ``map_location``.  A convenience ``save(state)`` wraps ``writer()`` +
    ``torch.save`` for the common write path.
    """

    def __init__(self, url: str):
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    @contextmanager
    def reader(self) -> Iterator[io.IOBase]:
        """Yield a readable binary stream for the checkpoint."""
        if self._url.startswith("s3://"):
            fs, key = self._s3_fs()
            with fs.open(key, "rb") as f:
                yield f
        else:
            path = Path(os.path.expanduser(self._url))
            with open(path, "rb") as f:
                yield f

    @contextmanager
    def writer(self) -> Iterator[io.IOBase]:
        """Yield a writable binary stream for the checkpoint.

        Local writes are atomic: data lands in a ``*.tmp`` sibling, is fsynced,
        then renamed over the destination.  An interrupt or crash mid-write
        leaves the final path untouched (and the ``.tmp`` cleaned up on
        exception).  S3 writes are atomic at the object level by boto3/MPU.
        """
        if self._url.startswith("s3://"):
            fs, key = self._s3_fs()
            with fs.open(key, "wb") as f:
                yield f
            return
        path = Path(os.path.expanduser(self._url))
        path.parent.mkdir(parents=True, exist_ok=True)
        # path.parent / (path.name + ".tmp") works for any name, including
        # dotfiles where path.with_suffix() raises on Python 3.12+.
        tmp = path.parent / (path.name + ".tmp")
        try:
            with open(tmp, "wb") as f:
                yield f
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def save(self, state: dict) -> None:
        """Convenience: serialize *state* to the checkpoint URL."""
        with self.writer() as f:
            torch.save(state, f)

    def exists(self) -> bool:
        """Check whether the checkpoint exists."""
        if self._url.startswith("s3://"):
            fs, key = self._s3_fs()
            return fs.exists(key)
        return os.path.exists(os.path.expanduser(self._url))

    def delete(self) -> None:
        """Delete the checkpoint if it exists. Idempotent."""
        if self._url.startswith("s3://"):
            fs, key = self._s3_fs()
            fs.delete(key)
            return
        path = Path(os.path.expanduser(self._url))
        path.unlink(missing_ok=True)

    def _s3_fs(self):
        from rlcade.checkpoint.s3 import S3FileSystem

        bucket, key = _parse_s3(self._url)
        return S3FileSystem(bucket, ""), key


def _parse_s3(url: str) -> tuple[str, str]:
    """Return ``(bucket, key)`` from an ``s3://bucket/key`` URL."""
    parts = url[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    if not bucket:
        raise ValueError(f"S3 URL missing bucket: {url!r}")
    if not key:
        raise ValueError(f"S3 URL missing key: {url!r}")
    return bucket, key
