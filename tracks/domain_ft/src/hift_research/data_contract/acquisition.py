"""Bounded official COCO TRAIN acquisition; no unbounded Range fallback."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from hift_research.data_acquisition.range_zip import HTTPRangeReader

ANNOTATIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip"
BUDGET = 128 * 1024 * 1024
AUTHORIZED_WAVE3_BUDGET = 192 * 1024 * 1024


def cached_reader(metadata, records, *, remaining_budget):
    reader = HTTPRangeReader(metadata["url"], metadata["size"], budget_bytes=remaining_budget)
    reader.last_modified = metadata["last_modified"]
    for record in records:
        if record["url"] != metadata["url"]:
            continue
        payload = Path(record["cache_path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("prior inspected cache changed")
        reader.cache.append((record["start"], payload))
    return reader


def acquire_subset(probe_path, output, data_root, *, image_limit=32, budget=AUTHORIZED_WAVE3_BUDGET):
    """Acquire complete official TRAIN instances member and preselected images.

    The <=32 source image IDs are frozen from annotation metadata BEFORE any
    image read. All object categories are retained; invalid/crowd rows quarantine.
    """
    import shutil
    import cv2
    import numpy as np
    if not 1 <= image_limit <= 32 or budget > AUTHORIZED_WAVE3_BUDGET:
        raise ValueError("authorized <=32 image /192MiB training budget exceeded")
    output, data_root = Path(output), Path(data_root)
    output.mkdir(parents=True, exist_ok=False)
    data_root.mkdir(parents=True, exist_ok=False)
    probe = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    prior = probe["bytes_received"]
    ledger = list(probe["network_ledger"])
    annotation_meta, image_meta = probe["archives"]
    reader = cached_reader(annotation_meta, ledger, remaining_budget=budget-prior)
    reader.cache_dir = output / "annotation-ranges"
    member = "annotations/instances_train2017.json"
    annotation_path = data_root / "instances_train2017.json"
    try:
        with zipfile.ZipFile(reader) as archive:
            info = archive.getinfo(member)
            if prior + info.compress_size + 1024 > budget:
                raise ValueError("complete annotation member cannot fit approved budget")
            with archive.open(info) as source, annotation_path.open("xb") as target:
                shutil.copyfileobj(source, target, length=8*1024*1024)
        ledger.extend(reader.records)
        spent = prior + reader.network_bytes
        annotation_sha = sha256_file(annotation_path)
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_by_id = {row["id"]: row for row in raw["images"]}
        grouped = {}
        quarantine = []
        for ann in raw["annotations"]:
            box = np.asarray(ann.get("bbox", []), dtype=float)
            reason = ("crowd_ambiguous_identity" if ann.get("iscrowd") else
                      "malformed_bbox" if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0 else None)
            if reason:
                quarantine.append({"annotation_id": ann["id"], "image_id": ann["image_id"], "reason": reason})
            else:
                grouped.setdefault(ann["image_id"], []).append(ann)
        selected_ids = sorted(set(grouped).intersection(image_by_id))[:image_limit]
        selection = {"schema": "hift-coco-train-selection-v1", "split": "train2017",
                     "policy": "smallest_numeric_image_ids_with_valid_noncrowd_instances",
                     "image_ids": selected_ids, "image_limit": image_limit,
                     "annotation_member": member, "annotation_sha256": annotation_sha,
                     "annotations": [ann for image_id in selected_ids for ann in grouped[image_id]],
                     "images": [image_by_id[image_id] for image_id in selected_ids],
                     "licenses": raw["licenses"], "info": raw["info"],
                     "quarantine": [row for row in quarantine if row["image_id"] in selected_ids]}
        write_exclusive_json(output / "selection_before_images.json", selection)
        del raw, image_by_id, grouped, quarantine
        image_reader = cached_reader(image_meta, ledger, remaining_budget=budget-spent)
        image_reader.cache_dir = output / "image-ranges"
        downloaded = []
        image_dir = data_root / "train2017"
        image_dir.mkdir()
        try:
            with zipfile.ZipFile(image_reader) as archive:
                projected = sum(archive.getinfo("train2017/" + row["file_name"]).compress_size + 512
                                for row in selection["images"])
                if spent + projected > budget:
                    raise ValueError("selected source images exceed approved remaining budget")
                for row in selection["images"]:
                    name = "train2017/" + row["file_name"]
                    if Path(row["file_name"]).name != row["file_name"]:
                        raise ValueError("unsafe source member name")
                    payload = archive.read(name)
                    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or list(image.shape[:2]) != [row["height"], row["width"]]:
                        raise ValueError("source JPEG decode or official dimensions mismatch")
                    path = image_dir / row["file_name"]
                    path.write_bytes(payload)
                    downloaded.append(dict(row, path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                                           archive_member=name, crc32=archive.getinfo(name).CRC, bytes=len(payload)))
        finally:
            ledger.extend(image_reader.records)
            spent += image_reader.network_bytes
        result = {"schema": "hift-coco-training-acquisition-v1", "budget_bytes": budget,
                  "bytes_received_including_probe": spent, "image_count": len(downloaded),
                  "annotation_path": str(annotation_path.resolve()), "annotation_sha256": annotation_sha,
                  "annotation_member": member, "sources": probe["archives"], "images": downloaded,
                  "selection_path": str((output / "selection_before_images.json").resolve()),
                  "selection_sha256": sha256_file(output / "selection_before_images.json"),
                  "network_ledger": ledger}
        write_exclusive_json(output / "acquisition.json", result)
        return result
    except BaseException as exc:
        write_exclusive_json(output / "failure.json", {"error": repr(exc), "network_ledger": ledger + reader.records})
        raise


def head(url):
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "HiFT-bounded-training-contract/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return {"url": url, "status": response.status,
                "size": int(response.headers["Content-Length"]),
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
                "accept_ranges": response.headers.get("Accept-Ranges")}


def inspect_archive(url, cache_dir, *, budget=BUDGET):
    metadata = head(url)
    reader = HTTPRangeReader(url, metadata["size"], budget_bytes=budget, cache_dir=cache_dir)
    archive = zipfile.ZipFile(reader)
    members = [{"name": info.filename, "compressed": info.compress_size,
                "uncompressed": info.file_size, "offset": info.header_offset,
                "crc32": info.CRC} for info in archive.infolist()]
    return metadata, reader, archive, members


def write_exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
