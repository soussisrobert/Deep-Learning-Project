"""Extract selected members of remote Hugging Face zip archives via HTTP Range requests.

Reads the zip central directory remotely and streams only members whose path contains one of
--include substrings (all members if none given), so unneeded splits are never downloaded.
zipfile verifies each member's CRC-32 while streaming; extraction is zip-slip protected.
Records the Hub LFS SHA-256/size of each archive (whole-archive hash is not recomputed in
range mode). Contents are data only. Immutable ledger.

    python scripts/acquire_hf_zip_members.py --repo vanthanh/VisDrone2019-MOT --files VisDrone2019-MOT-train.zip \
        --root data/visdrone --ledger artifacts/visdrone/MOT-ACQUISITION-001.json
"""
import argparse
import http.client
import io
import json
import shutil
import time
import urllib.request
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

BLOCK = 16*1024*1024


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file using HTTP Range requests and one cached block."""
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.block_start, self.block = -1, b""
        self.requests = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos+offset, 2: self.size+offset}[whence]
        return self.pos

    def _fetch(self, start):
        end = min(start+BLOCK, self.size)-1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError(f"server ignored Range (status {response.status})")
                    data = response.read()
                if len(data) != end-start+1:
                    raise OSError("short range read")
                self.requests += 1
                self.block_start, self.block = start, data
                return
            except (OSError, http.client.HTTPException):  # IncompleteRead is not an OSError
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 60))

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size-self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            if not self.block_start <= self.pos < self.block_start+len(self.block):
                self._fetch(self.pos - self.pos % BLOCK)
            offset = self.pos-self.block_start
            chunk = self.block[offset:offset+n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data
        return len(data)


def verified_existing(path, member):
    """True when a file left by an interrupted run already matches the member's size and CRC-32."""
    if not path.exists():
        return False
    if path.stat().st_size == member.file_size:
        crc = 0
        with path.open("rb") as stream:
            while chunk := stream.read(BLOCK):
                crc = zlib.crc32(chunk, crc)
        if crc == member.CRC:
            return True
    path.unlink()  # truncated or corrupt partial file from the interrupted run
    return False


def lfs_info(repo, path):
    body = json.dumps({"paths": [path], "expand": True}).encode()
    request = urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/paths-info/main", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    target = args.root.resolve()
    target.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger = {"schema": "hift-hf-zip-member-acquisition-v1", "repo": args.repo, "include": args.include,
              "integrity": "per-member CRC-32 verified by zipfile while streaming; Hub LFS oid recorded",
              "started_utc": datetime.now(timezone.utc).isoformat(), "archives": {}}
    for name in args.files:
        info = lfs_info(args.repo, name)
        remote = HTTPRangeFile(f"https://huggingface.co/datasets/{args.repo}/resolve/main/{name}", info["size"])
        start, count, written, skipped, resumed = time.time(), 0, 0, 0, 0
        with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=BLOCK)) as zf:
            for member in zf.infolist():
                if args.include and not any(token in member.filename for token in args.include):
                    skipped += 1; continue
                out = (target/member.filename).resolve()
                if out != target and target not in out.parents:
                    raise ValueError(f"unsafe archive path {member.filename}")
                if member.is_dir():
                    out.mkdir(parents=True, exist_ok=True); continue
                out.parent.mkdir(parents=True, exist_ok=True)
                if verified_existing(out, member):
                    count += 1; written += member.file_size; resumed += 1; continue
                with zf.open(member) as source, out.open("xb") as sink:  # CRC checked at end of read
                    shutil.copyfileobj(source, sink, BLOCK)
                count += 1; written += member.file_size
                if count % 20000 == 0:
                    print(name, count, "files", round(written/2**30, 1), "GiB", flush=True)
        ledger["archives"][name] = {"hub_size": info["size"], "hub_lfs_sha256": info.get("lfs", {}).get("oid"),
                                    "extracted_files": count, "extracted_bytes": written, "skipped_members": skipped, "resumed_verified": resumed,
                                    "range_requests": remote.requests, "seconds": time.time()-start}
        print(name, "done", count, "files", round(written/2**30, 2), "GiB", flush=True)
    ledger["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with args.ledger.open("x", encoding="utf-8") as stream:
        json.dump(ledger, stream, indent=2)


if __name__ == "__main__":
    main()
