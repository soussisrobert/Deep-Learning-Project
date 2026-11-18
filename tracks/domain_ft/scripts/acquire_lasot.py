"""Acquire selected official LaSOT categories from the authors' Hugging Face repo (l-lt/LaSOT).

Each zip streams to disk with SHA-256 computed on the fly and is checked against the Hub's
LFS oid and size. Only sequences in the official training_set.txt are extracted, with zip-slip
protection; zips are deleted after verified extraction to bound disk use. Writes an immutable
ledger. Archive contents are data only; nothing from them is executed.

    python scripts/acquire_lasot.py --root data/lasot --categories car person ... --ledger artifacts/lasot/ACQUISITION-001.json
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = "https://huggingface.co/datasets/l-lt/LaSOT"
API = "https://huggingface.co/api/datasets/l-lt/LaSOT/paths-info/main"
CHUNK = 8*1024*1024


def lfs_info(paths):
    body = json.dumps({"paths": paths, "expand": True}).encode()
    request = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return {row["path"]: row for row in json.load(response)}


def fetch_text(name):
    with urllib.request.urlopen(f"{REPO}/resolve/main/{name}", timeout=60) as response:
        return response.read()


def download(name, destination, expected_size, expected_sha):
    partial = destination.with_suffix(destination.suffix+".partial")
    digest, received, start = hashlib.sha256(), 0, time.time()
    with urllib.request.urlopen(f"{REPO}/resolve/main/{name}", timeout=120) as response, partial.open("xb") as stream:
        while chunk := response.read(CHUNK):
            stream.write(chunk); digest.update(chunk); received += len(chunk)
    if received != expected_size or digest.hexdigest() != expected_sha:
        raise ValueError(f"{name}: size/sha256 mismatch ({received} vs {expected_size})")
    partial.rename(destination)
    return {"bytes": received, "sha256": digest.hexdigest(), "seconds": time.time()-start}


def extract_training(archive, root, category, training):
    target = (root/"original").resolve()
    members, skipped = 0, 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            sequence = next((p for p in parts if p.startswith(category+"-")), None)
            if sequence not in training:
                skipped += 1
                continue
            out = (target/info.filename).resolve()
            if target not in out.parents:
                raise ValueError(f"unsafe archive path {info.filename}")
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, out.open("xb") as sink:
                shutil.copyfileobj(source, sink, CHUNK)
            members += 1
    return {"extracted_files": members, "skipped_entries": skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    (root/"zips").mkdir(parents=True, exist_ok=True)
    (root/"original").mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ("training_set.txt", "testing_set.txt"):
        payload = fetch_text(name)
        (root/name).write_bytes(payload)
        splits[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "sequences": len(payload.split())}
    training = set((root/"training_set.txt").read_text().split())
    testing = set((root/"testing_set.txt").read_text().split())
    if training & testing:
        raise ValueError("official train/test lists overlap")
    info = lfs_info([f"{c}.zip" for c in args.categories])
    ledger = {"schema": "hift-lasot-acquisition-v1", "source": REPO, "started_utc": datetime.now(timezone.utc).isoformat(),
              "split_files": splits, "extracted_split": "official training_set.txt only; testing_set sequences never extracted",
              "categories": {}}
    for category in args.categories:
        name = f"{category}.zip"
        expected = info[name]
        record = download(name, root/"zips"/name, expected["size"], expected["lfs"]["oid"])
        record.update(extract_training(root/"zips"/name, root, category, {s for s in training if s.startswith(category+"-")}))
        record["training_sequences"] = sorted(s for s in training if s.startswith(category+"-"))
        (root/"zips"/name).unlink()
        record["zip_deleted_after_verified_extraction"] = True
        ledger["categories"][category] = record
        print(category, record["bytes"], f"{record['seconds']:.0f}s", record["extracted_files"], "files", flush=True)
    ledger["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with args.ledger.open("x", encoding="utf-8") as stream:
        json.dump(ledger, stream, indent=2)


if __name__ == "__main__":
    sys.exit(main())
