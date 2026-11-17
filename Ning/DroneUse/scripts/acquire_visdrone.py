"""Acquire official VisDrone2019 SOT/MOT splits from the Google Drive links in github.com/VisDrone/VisDrone-Dataset.

No publisher hashes exist: each archive's SHA-256 and size are recorded, every member CRC is
verified (zipfile.testzip) before extraction, extraction is zip-slip protected, and the zip is
deleted only after verified extraction. Archive contents are data only. Immutable ledger.

    python scripts/acquire_visdrone.py --root data/visdrone --ledger artifacts/visdrone/ACQUISITION-001.json
"""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SPLITS = {  # name: (Google Drive file id, publisher-stated size)
    "SOT-train-part1": ("1a-AmQjYATzj8seXLlEm9Sx8aQmClrJka", "7.78 GB"),
    "SOT-train-part2": ("16YPyhNDQrTgW8I2HaH_HNEO-KTRA-xso", "12.59 GB"),
    "SOT-val": ("18SNAOlCJtApnG2m45ud-1e_OtGYill0D", "1.29 GB"),
    "SOT-test-dev": ("1xCiHjU4JlR9QsYtiHYy2UUd3m6NthoBC", "11.27 GB"),
    "MOT-train": ("1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh", "7.53 GB"),
    "MOT-val": ("1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu", "1.48 GB"),
    "MOT-test-dev": ("14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu", "2.14 GB"),
}
CHUNK = 8*1024*1024


def download(file_id, destination):
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    partial = destination.with_suffix(".partial")
    digest, received, start = hashlib.sha256(), 0, time.time()
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("xb") as stream:
        kind = response.headers.get("Content-Type", "")
        if "text/html" in kind:
            raise ValueError(f"Google Drive returned HTML (quota/consent page), not an archive: {file_id}")
        while chunk := response.read(CHUNK):
            stream.write(chunk); digest.update(chunk); received += len(chunk)
    partial.rename(destination)
    return {"bytes": received, "sha256": digest.hexdigest(), "seconds": time.time()-start, "content_type": kind}


def extract(archive, target):
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in {archive.name}: {bad}")
        count = 0
        for info in zf.infolist():
            out = (target/info.filename).resolve()
            if out != target and target not in out.parents:
                raise ValueError(f"unsafe archive path {info.filename}")
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True); continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, out.open("xb") as sink:
                shutil.copyfileobj(source, sink, CHUNK)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = parser.parse_args()
    root = args.root.resolve()
    (root/"zips").mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger = {"schema": "hift-visdrone-acquisition-v1", "source": "https://github.com/VisDrone/VisDrone-Dataset (official Google Drive links)",
              "publisher_hashes": "none published; SHA-256 recorded at acquisition, member CRCs verified",
              "started_utc": datetime.now(timezone.utc).isoformat(), "splits": {}}
    for name in args.splits:
        file_id, stated = SPLITS[name]
        archive = root/"zips"/f"{name}.zip"
        record = {"google_drive_id": file_id, "publisher_stated_size": stated, **download(file_id, archive)}
        record["extracted_files"] = extract(archive, root/name)
        archive.unlink()
        record["zip_deleted_after_verified_extraction"] = True
        ledger["splits"][name] = record
        print(name, record["bytes"], f"{record['seconds']:.0f}s", record["extracted_files"], "files", flush=True)
    ledger["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with args.ledger.open("x", encoding="utf-8") as stream:
        json.dump(ledger, stream, indent=2)


if __name__ == "__main__":
    main()
