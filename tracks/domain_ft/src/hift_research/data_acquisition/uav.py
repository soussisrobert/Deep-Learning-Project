"""Official UAV selective acquisition and complete-segment validation.

No annotations are generated, resampled, interpolated, or repaired. Original ZIP
members are preserved and copied byte-for-byte into a documented adapter layout.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

import numpy as np
from PIL import Image

from hift_research.evaluation.assets import import_dataset, read_boxes, sha256_file, write_json_exclusive
from hift_research.evaluation.metrics import valid_boxes
from .range_zip import AcquisitionError, HTTPRangeReader

OFFICIAL_PAGE = "https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset"
SOURCES = {
    "UAV123@10fps": {
        "url": "https://drive.usercontent.google.com/download?id=0B6sQMCU1i4NbZmFlQmJBVDlLRDg&export=download&confirm=t&resourcekey=0--jsSKS1oGFidNhgMF75cSQ",
        "size": 4688528930, "prefix": "UAV123_10fps",
        "last_modified_observed": "Fri, 19 Dec 2025 23:52:28 GMT",
    },
    "UAV123": {
        "url": "https://drive.usercontent.google.com/download?id=0B6sQMCU1i4NbNGxWQzRVak5yLWs&export=download&confirm=t&resourcekey=0-IjwQcWEzP2x3ec8kXtLBpA",
        "size": 14049397769, "prefix": "UAV123",
        "last_modified_observed": "Fri, 19 Dec 2025 22:05:18 GMT",
    },
}
PILOT_SEQUENCES = ("uav6", "wakeboard7", "building5")


def parse_sequence_config(text):
    """Parse declared fields only; NEVER execute externally supplied MATLAB."""
    result = {}
    for body in re.findall(r"struct\((.*?)\)", text, flags=re.S):
        name = re.search(r"'name'\s*,\s*'([^']+)'", body)
        path = re.search(r"'path'\s*,\s*'([^']+)'", body)
        fields = {key: re.search(r"'" + key + r"'\s*,\s*(\d+)", body)
                  for key in ("startFrame", "endFrame", "nz")}
        extension = re.search(r"'ext'\s*,\s*'([^']+)'", body)
        if not name or not path or not extension or not all(fields.values()):
            raise AcquisitionError("incomplete authoritative sequence-config struct")
        name = name.group(1)
        if name in result or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise AcquisitionError("unsafe/duplicate authoritative sequence name")
        folder = path.group(1).replace("\\", "/").rstrip("/").split("/")[-1]
        start, end, digits = (int(fields[k].group(1)) for k in ("startFrame", "endFrame", "nz"))
        if start < 1 or end < start or digits < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", folder):
            raise AcquisitionError("invalid authoritative frame range/folder")
        result[name] = {"folder": folder, "start_frame": start, "end_frame": end,
                        "digits": digits, "extension": extension.group(1)}
    if not result:
        raise AcquisitionError("no sequence declarations found")
    return result


def safe_member_target(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise AcquisitionError("unsafe archive member path")
    root = Path(root).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise AcquisitionError("archive member escapes output root")
    return target


def extract_member(archive, info, root, *, max_bytes=64 * 1024 * 1024):
    if info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16):
        raise AcquisitionError("encrypted/symlink archive members forbidden")
    if info.is_dir() or info.file_size > max_bytes or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise AcquisitionError("unsupported/oversized archive member")
    target = safe_member_target(root, info.filename)
    if target.exists():
        raise AcquisitionError(f"refusing to overwrite acquired member: {target}")
    payload = archive.read(info)  # zipfile verifies the original central-directory CRC.
    if len(payload) != info.file_size:
        raise AcquisitionError("extracted member length mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
    return {"archive_member": info.filename, "path": str(target), "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "zip_crc32": f"{info.CRC:08x}",
            "zip_header_offset": info.header_offset, "zip_compressed_size": info.compress_size}


def validate_sequence(image_dir, annotation_path, config):
    """Assert exact official IDs/order/row count, decode every image, validate XYWH."""
    image_dir = Path(image_dir)
    expected = [f"{i:0{config['digits']}d}.{config['extension']}"
                for i in range(config["start_frame"], config["end_frame"] + 1)]
    actual = {p.name for p in image_dir.iterdir() if p.is_file()}
    if actual != set(expected):
        raise AcquisitionError(f"official frame ID set mismatch: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}")
    boxes = read_boxes(annotation_path)
    if len(boxes) != len(expected):
        raise AcquisitionError("official annotation/frame count mismatch")
    valid = valid_boxes(boxes)
    # Official ReadMe explicitly defines full-NaN annotations as absent/occluded.
    # Keep them aligned with image rows and score them by explicit failure policy.
    absent = np.isnan(boxes).all(axis=1)
    malformed = ~(valid | absent)
    if malformed.any() or not valid[0]:
        raise AcquisitionError(f"malformed XYWH rows {np.flatnonzero(malformed).tolist()} or invalid first-frame initialization; no rows dropped")
    frames = []
    for name in expected:
        path = image_dir / name
        try:
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise AcquisitionError(f"expected original JPEG frame: {name}")
                image.load()  # Force full pixel decoding, not merely header probing.
                if image.width <= 0 or image.height <= 0:
                    raise AcquisitionError("nonpositive decoded image dimensions")
                frames.append({"path": str(path.resolve()), "sha256": sha256_file(path),
                               "width": image.width, "height": image.height, "mode": image.mode,
                               "size_bytes": path.stat().st_size})
        except (OSError, Image.DecompressionBombError) as exc:
            raise AcquisitionError(f"frame decode failed: {path}") from exc
    return {"frame_count": len(frames), "first_frame_id": config["start_frame"],
            "last_frame_id": config["end_frame"], "frame_id_step": 1,
            "all_frames_decoded": True, "all_nonabsent_xywh_finite_positive_size": True,
            "absent_annotation_count": int(absent.sum()),
            "absent_frame_ids": [config["start_frame"] + int(i) for i in np.flatnonzero(absent)],
            "invalid_policy": "count_as_failure", "annotation_rows_dropped": 0,
            "annotation_sha256": sha256_file(annotation_path), "initial_box_annotation_native": boxes[0].tolist(),
            "box_xy_min": np.min(boxes[valid, :2], axis=0).tolist(),
            "box_wh_min": np.min(boxes[valid, 2:], axis=0).tolist(), "frames": frames}


def _load_inventory_cache(reader, inventory):
    if inventory.get("url") != reader.url or inventory.get("archive_size") != reader.size:
        raise AcquisitionError("inventory is not for the pinned official archive")
    for row in inventory.get("ranges", []):
        payload = Path(row["cache_path"]).read_bytes()
        if len(payload) != row["end"] - row["start"] + 1 or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise AcquisitionError("cached official archive range hash/length mismatch")
        reader.cache.append((row["start"], payload))


def acquire_uav_pilot(output_root, provenance_dir, *, sequences=PILOT_SEQUENCES,
                      inventory=None, budget_bytes=480 * 1024 * 1024,
                      annotation_coordinate_origin=None,
                      selection_id="WAVE2-UAV10-PILOT-001", selection_rule=None):
    """Acquire explicitly fixed complete official 10fps segments, never full ZIP.

    Optional selection metadata generalizes the existing adapter without changing
    Wave2 defaults, bytes, annotations, validation, or evaluation behavior.
    """
    source = SOURCES["UAV123@10fps"]
    output_root, provenance_dir = Path(output_root).resolve(), Path(provenance_dir).resolve()
    if output_root.exists() or provenance_dir.exists():
        raise AcquisitionError("new output and provenance directories required; do not overwrite runs")
    if len(set(sequences)) != len(sequences) or not sequences:
        raise AcquisitionError("unique explicit sequence selection required")
    provenance_dir.mkdir(parents=True)
    write_json_exclusive(provenance_dir / "selection.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "sequences": list(sequences),
        "selected_before_tracking": True, "selection_id": selection_id,
        "selection_rule": selection_rule or "fixed official short complete sequences across UAV/human sport/building categories; no model outputs used",
        "dataset": "UAV123@10fps", "network_budget_bytes": budget_bytes})
    reader = HTTPRangeReader(source["url"], source["size"], budget_bytes=budget_bytes,
                             cache_dir=provenance_dir / "ranges")
    reader.last_modified = source["last_modified_observed"]
    if inventory:
        _load_inventory_cache(reader, inventory)
    prefix = source["prefix"]
    records, validations = [], {}
    try:
        with zipfile.ZipFile(reader) as archive:
            infos = archive.infolist()
            names = {i.filename: i for i in infos}
            if len(names) != len(infos):
                raise AcquisitionError("duplicate ZIP member names")
            for suffix in ("configSeqs.m", "ReadMe.txt"):
                records.append(extract_member(archive, names[f"{prefix}/{suffix}"], output_root / "original"))
            config_path = output_root / "original" / prefix / "configSeqs.m"
            configs = parse_sequence_config(config_path.read_text(encoding="utf-8-sig"))
            ordered_infos = sorted(infos, key=lambda i: i.header_offset)
            next_offset = {a.header_offset: b.header_offset for a, b in zip(ordered_infos, ordered_infos[1:])}
            for name in sequences:
                if name not in configs:
                    raise AcquisitionError(f"sequence not declared by official config: {name}")
                config = configs[name]
                # These pilot segments are entire source folders; no ambiguous shared-video trimming.
                folder = f"{prefix}/data_seq/{prefix}/{config['folder']}/"
                selected = [names[f"{folder}{index:0{config['digits']}d}.{config['extension']}"]
                            for index in range(config["start_frame"], config["end_frame"] + 1)]
                annotation_name = f"{prefix}/anno/{prefix}/{name}.txt"
                records.append(extract_member(archive, names[annotation_name], output_root / "original"))
                start = min(i.header_offset for i in selected)
                end = next_offset.get(max(i.header_offset for i in selected))
                if end is None:
                    raise AcquisitionError("could not bound final selected ZIP member span")
                if end - start > sum(i.compress_size for i in selected) + 1024 * 1024:
                    raise AcquisitionError("selected members not contiguous; refusing unexpectedly large span")
                reader.prefetch(start, end - 1)
                for info in selected:
                    records.append(extract_member(archive, info, output_root / "original"))
                source_dir = output_root / "original" / folder
                annotation_path = output_root / "original" / annotation_name
                validation = validate_sequence(source_dir, annotation_path, config)
                validation["authoritative_config"] = config
                validations[name] = validation
                # Byte-preserving adaptation for existing evaluation import APIs.
                ready_dir = output_root / "prepared/data_seq" / name
                ready_dir.mkdir(parents=True)
                for frame in validation["frames"]:
                    shutil.copyfile(frame["path"], ready_dir / Path(frame["path"]).name)
                prepared_annotation = output_root / "prepared/anno" / f"{name}.txt"
                prepared_annotation.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(annotation_path, prepared_annotation)
                print(f"Validated complete official sequence {name}: {validation['frame_count']} frames, {reader.network_bytes} new network bytes", flush=True)
        manifest = import_dataset(output_root / "prepared", "UAV123@10fps", allow_partial=True,
                                  invalid_policy="count_as_failure", hash_frames=True)
        manifest["annotation_coordinate_origin"] = annotation_coordinate_origin
        manifest["selected_segment_membership_verified_against_original_config"] = True
        manifest["acquisition_selection_id"] = selection_id
        write_json_exclusive(provenance_dir / "dataset_manifest.json", manifest)
        report = {"schema_version": 1, "kind": "authoritative_partial_dataset_acquisition",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "UAV123@10fps",
                  "official_page": OFFICIAL_PAGE, "source": source,
                  "source_archive_full_sha256": None,
                  "source_identity_limitation": "selective authoritative HTTPS ranges, stable size/Last-Modified, ZIP CRC and selected-member SHA256; full archive not downloaded/hashed",
                  "selected_sequences": list(sequences), "network_bytes_this_run": reader.network_bytes,
                  "prior_inventory_network_bytes": inventory.get("network_bytes", 0) if inventory else 0,
                  "range_responses": reader.records, "members": records, "validation": validations,
                  "original_config_sha256": sha256_file(output_root / "original" / prefix / "configSeqs.m"),
                  "original_readme_sha256": sha256_file(output_root / "original" / prefix / "ReadMe.txt"),
                  "annotation_coordinate_origin": annotation_coordinate_origin,
                  "inference_ground_truth_boundary": "prepared dataset/validation manifest is EVALUATION ONLY; inference requires separate frames+initialization input",
                  "selection_id": selection_id,
                  "limitations": [f"{len(sequences)} complete official sequences, not full123 benchmark", "source archive may be replaced by publisher; archived selected bytes and hashes preserved", "UAV variants are overlapping not independent evidence"]}
        write_json_exclusive(provenance_dir / "acquisition.json", report)
        return report
    finally:
        write_json_exclusive(provenance_dir / "network_ledger.json", {
            "network_bytes_this_run": reader.network_bytes, "ranges": reader.records,
            "last_modified": reader.last_modified, "budget_bytes": budget_bytes})
