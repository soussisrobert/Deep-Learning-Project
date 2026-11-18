"""Seekable HTTP206 reader for selective standard-library ZIP/ZIP64 access.

No fallback to an unbounded full GET. Each response must match requested range,
archive total size, and stable Last-Modified metadata before any bytes are used.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import urllib.request


class AcquisitionError(ValueError):
    pass


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, size, *, budget_bytes=512 * 1024 * 1024,
                 cache_dir=None, opener=None, timeout=60):
        super().__init__()
        if not isinstance(size, int) or size <= 0 or budget_bytes <= 0:
            raise AcquisitionError("positive archive size and network budget required")
        if not url.startswith("https://"):
            raise AcquisitionError("authoritative retrieval requires explicit HTTPS URL")
        self.url, self.size, self.budget_bytes = url, size, budget_bytes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.opener = opener or urllib.request.urlopen
        self.timeout, self.position, self.network_bytes = timeout, 0, 0
        self.records, self.cache = [], []
        self.last_modified = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else (self.position + offset if whence == io.SEEK_CUR else self.size + offset if whence == io.SEEK_END else -1)
        if position < 0:
            raise AcquisitionError("invalid negative seek or whence")
        self.position = position
        return position

    def read(self, size=-1):
        if self.closed:
            raise AcquisitionError("range reader is closed")
        if self.position >= self.size or size == 0:
            return b""
        size = self.size - self.position if size is None or size < 0 else min(size, self.size - self.position)
        start, end = self.position, self.position + size
        for cached_start, payload in reversed(self.cache):
            if cached_start <= start and end <= cached_start + len(payload):
                self.position = end
                return payload[start - cached_start:end - cached_start]
        payload = self.fetch(start, end - 1)
        self.position = end
        return payload

    def fetch(self, start, end):
        """Fetch exact inclusive range, or fail BEFORE full-archive data is read."""
        if start < 0 or end < start or end >= self.size:
            raise AcquisitionError("range outside archive")
        expected = end - start + 1
        if self.network_bytes + expected > self.budget_bytes:
            raise AcquisitionError("network byte budget exceeded before request")
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                   "User-Agent": "HiFT-research-bounded-acquisition/1.0"}
        if self.last_modified:
            headers["If-Unmodified-Since"] = self.last_modified
        request = urllib.request.Request(self.url, headers=headers)
        with self.opener(request, timeout=self.timeout) as response:
            if response.status != 206:
                raise AcquisitionError(f"server ignored Range (status {response.status}); full response NOT consumed")
            content_range = response.headers.get("Content-Range", "")
            if content_range != f"bytes {start}-{end}/{self.size}":
                raise AcquisitionError(f"unexpected Content-Range: {content_range}")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise AcquisitionError("encoded range response is not byte-identical archive content")
            modified = response.headers.get("Last-Modified")
            if self.last_modified and modified != self.last_modified:
                raise AcquisitionError("archive changed Last-Modified during retrieval")
            self.last_modified = modified
            chunks, remaining = [], expected
            while remaining:
                chunk = response.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise AcquisitionError("truncated HTTP range response")
                chunks.append(chunk)
                remaining -= len(chunk)
                self.network_bytes += len(chunk)
            # Content-Range and Content-Length must both agree; no extra body read.
            if response.headers.get("Content-Length") not in {None, str(expected)}:
                raise AcquisitionError("range Content-Length mismatch")
            payload = b"".join(chunks)
            record = {"start": start, "end": end, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                      "content_range": content_range, "last_modified": modified, "url": self.url}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{start}-{end}.bin"
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise AcquisitionError("existing range-cache bytes differ")
            else:
                with path.open("xb") as stream:
                    stream.write(payload)
            record["cache_path"] = str(path.resolve())
        self.records.append(record)
        self.cache.append((start, payload))
        return payload

    def prefetch(self, start, end, *, chunk_bytes=16 * 1024 * 1024):
        """Cache a contiguous member span, bounding each request and total budget."""
        if chunk_bytes <= 0:
            raise AcquisitionError("positive prefetch chunk size required")
        if self.network_bytes + end - start + 1 > self.budget_bytes:
            raise AcquisitionError("prefetch exceeds remaining network byte budget")
        parts = []
        offset = start
        while offset <= end:
            stop = min(end, offset + chunk_bytes - 1)
            parts.append(self.fetch(offset, stop))
            offset = stop + 1
        # A merged span permits ZipFile reads crossing a request boundary.
        self.cache.append((start, b"".join(parts)))
