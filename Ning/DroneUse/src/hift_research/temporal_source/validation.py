"""Independent CPU raw-CSV/content audit of the frozen source receipt."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import zlib

import cv2
from PIL import Image

from hift_research.data_contract.screening import phash64, content_screen
from hift_research.evaluation.assets import read_json, sha256_file, write_json_exclusive


def assert_annotation_matches_source(annotation, fields, frame_id, source_row):
    """Separate scalar comparison against the literal original nine-column row."""
    if len(fields) != 9:
        raise ValueError("raw original GT row not nine columns")
    values = list(map(float, fields))
    if (annotation["frame_id"] != frame_id or values[0] != frame_id or annotation["source_row"] != source_row
            or annotation["track_id"] != str(int(values[1])) or annotation["bbox_xywh_raw"] != values[2:6]
            or annotation["mark"] != values[6] or annotation["class_id"] != values[7]
            or annotation["visibility"] != values[8] or annotation["presence"] is not bool(values[6] == 1)):
        raise ValueError("canonical annotation differs from complete original source GT")


def audit_temporal_manifest(manifest_path, output):
    source = read_json(manifest_path)
    if source.get("schema") != "hift-temporal-source-v1" or source.get("source_dataset") != "MOT17/train":
        raise ValueError("exact declared temporal source schema required")
    if source["validation"]["production_training_authorized"] is not False or source["validation"]["GPU_used"] is not False:
        raise ValueError("source cannot grant real SGD or GPU authority")
    provenance = source["source_provenance"]
    for path_key, hash_key in (("selection_path", "selection_sha256"), ("inventory_path", "inventory_sha256")):
        if sha256_file(provenance[path_key]) != provenance[hash_key]:
            raise ValueError("frozen source selection/inventory changed")
    if sha256_file(source["reservations_path"]) != source["reservations_sha256"]:
        raise ValueError("frozen eval reservations changed")
    plan, reservations = read_json(provenance["selection_path"]), read_json(source["reservations_path"])
    planned = {v["video_lineage"]: v for v in plan["videos"]}
    if len(source["videos"]) != len(planned) or {v["video_lineage"] for v in source["videos"]} != set(planned):
        raise ValueError("source lineage population changed")
    results, all_count = [], 0
    for video in source["videos"]:
        chosen = planned[video["video_lineage"]]
        if video["split"] != chosen["split"] or video["coordinate_origin"] != 1:
            raise ValueError("source split/origin differs from frozen panel")
        if [f["frame_id"] for f in video["frames"]] != [f["frame_id"] for f in chosen["frames"]]:
            raise ValueError("source selected frame population/order changed")
        gt_path = Path(video["source_annotation_path"])
        if sha256_file(gt_path) != video["source_annotation_sha256"]:
            raise ValueError("complete original annotation bytes changed")
        if f"{zlib.crc32(gt_path.read_bytes()) & 0xffffffff:08x}" != video["original_GT"]["zip_crc32"]:
            raise ValueError("complete original GT CRC mismatch")
        raw_rows, raw_frames = {}, {}
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            for row_id, fields in enumerate(csv.reader(stream), 1):
                if not fields or not any(v.strip() for v in fields):
                    continue
                if len(fields) != 9:
                    raise ValueError("complete GT schema changed")
                raw_rows[row_id] = fields
                raw_frames.setdefault(int(fields[0]), []).append(row_id)
        if len(raw_rows) != video["source_row_count"]:
            raise ValueError("complete original GT row count mismatch")
        selected_annotations, quarantined_frames, quarantined_annotations, source_only_ignored = 0, 0, 0, 0
        for frame in video["frames"]:
            path = Path(frame["path"])
            if sha256_file(path) != frame["sha256"] or path.stat().st_size != frame["byte_size"]:
                raise ValueError("original frame bytes changed")
            if f"{zlib.crc32(path.read_bytes()) & 0xffffffff:08x}" != frame["zip_crc32"]:
                raise ValueError("original JPEG CRC differs from archive")
            with Image.open(path) as image:
                image.load()
                if image.format != "JPEG" or image.size != (frame["width"], frame["height"]):
                    raise ValueError("source JPEG decode/dimensions mismatch")
            if [frame["width"], frame["height"]] != video["native_size"]:
                raise ValueError("selected frame dimensions differ from complete source seqinfo")
            decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if decoded is None or phash64(decoded) != frame["phash64"]:
                raise ValueError("source OpenCV decode/perceptual hash differs")
            screen_reason, _ = content_screen(frame["sha256"], frame["phash64"], reservations)
            if screen_reason and screen_reason not in frame["quarantine_reasons"]:
                raise ValueError("unquarantined eval reservation match")
            expected_time = (frame["frame_id"] - 1) / video["seqinfo"]["fps"]
            if frame["timestamp_seconds"] != expected_time:
                raise ValueError("source timestamp not native(frame-1)/fps")
            expected_rows = raw_frames.get(frame["frame_id"], [])
            if [a["source_row"] for a in frame["annotations"]] != expected_rows:
                raise ValueError("selected-frame source rows dropped/reordered/added")
            for annotation in frame["annotations"]:
                row_id = annotation["source_row"]
                assert_annotation_matches_source(annotation, raw_rows[row_id], frame["frame_id"], row_id)
                if annotation["identity"] != ["MOT17", video["video_lineage"], annotation["track_id"]]:
                    raise ValueError("trajectory fragment scope changed")
                quarantined_annotations += bool(annotation["quarantine_reasons"])
                source_only_ignored += annotation["mark"] == 0
            selected_annotations += len(frame["annotations"])
            quarantined_frames += bool(frame["quarantine_reasons"])
        all_count += len(video["frames"])
        results.append({"video_lineage": video["video_lineage"], "split": video["split"], "original_frames": len(video["frames"]),
                        "complete_original_GT_rows": len(raw_rows), "selected_original_annotation_rows": selected_annotations,
                        "selected_source_mark0_rows_retained": source_only_ignored, "quarantined_selected_frames": quarantined_frames,
                        "quarantined_selected_annotation_rows": quarantined_annotations, "full_source_track_hint_records": len(video["quarantine"]),
                        "source_GT_sha256": video["source_annotation_sha256"]})
    if all_count != plan["total_original_frames"] or all_count > 160:
        raise ValueError("source total image count changed")
    report = {"kind": "independent_frozen_temporal_source_CPU_audit", "status": "passed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_manifest_path": str(Path(manifest_path).resolve()),
              "source_manifest_sha256": sha256_file(manifest_path), "frames": all_count, "videos": results,
              "budget": source["budget"], "source_rows_recomputed_from_complete_original_csv": True,
              "all_raw_source_annotations_preserved_no_interpolation": True, "GPU_used": False, "SGD_steps": 0,
              "production_training_authorized": False,
              "limitations": ["FullframeSHA/pHash screens do not prove semantic/pretraining alias freedom", "Technical source audit does not accept pair eligibility, negative crop exclusion, or production training readiness", "Withinvideo author fragment IDs only; no physical-person or biometric identity claim"]}
    write_json_exclusive(output, report)
    return report
