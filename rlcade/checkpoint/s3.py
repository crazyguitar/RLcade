"""Amazon S3 filesystem (requires boto3)."""

from __future__ import annotations

import io
from contextlib import contextmanager
from typing import Iterator


class S3FileSystem:
    """Amazon S3 filesystem (requires boto3)."""

    def __init__(self, bucket: str, prefix: str = "", region: str = ""):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._s3 = boto3.client("s3")
        self._region = region or boto3.Session().region_name or "us-east-1"

    def _key(self, path: str) -> str:
        parts = [self.prefix, path] if self.prefix else [path]
        return "/".join(parts)

    def read(self, path: str) -> bytes:
        resp = self._s3.get_object(Bucket=self.bucket, Key=self._key(path))
        return resp["Body"].read()

    def write(self, path: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._key(path), Body=data)

    def delete(self, path: str) -> None:
        """Delete the object at *path*. Idempotent -- missing keys are not an error."""
        self._s3.delete_object(Bucket=self.bucket, Key=self._key(path))

    def exists(self, path: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._key(path))
            return True
        except ClientError as e:
            # Only "not found" means the object doesn't exist. 403/5xx must
            # propagate so a misconfigured bucket or throttled call doesn't
            # silently skip a checkpoint load and restart training from scratch.
            err = e.response.get("Error", {})
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404 or err.get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    @contextmanager
    def open(self, path: str, mode: str = "rb") -> Iterator[io.IOBase]:
        key = self._key(path)
        s3_uri = f"s3://{self.bucket}/{key}"

        # Try s3torchconnector for true streaming
        try:
            from s3torchconnector import S3Checkpoint as _S3Ckpt

            ckpt = _S3Ckpt(region=self._region)
            if "w" in mode:
                with ckpt.writer(s3_uri) as w:
                    yield w
            else:
                with ckpt.reader(s3_uri) as r:
                    yield r
            return
        except ImportError:
            pass

        # Fallback: BytesIO buffering via boto3
        if "w" in mode:
            buf = io.BytesIO()
            yield buf
            buf.seek(0)
            self.write(path, buf.read())
        else:
            yield io.BytesIO(self.read(path))

    def list(self, prefix: str = "") -> list[str]:
        full_prefix = self._key(prefix)
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix += "/"
        paginator = self._s3.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1 :]
                results.append(key)
        return results

    def __repr__(self) -> str:
        return f"S3FileSystem(s3://{self.bucket}/{self.prefix})"
