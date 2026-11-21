"""Immutable, byte-bounded original MOT17 temporal subset acquisition."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import cv2
from PIL import Image

from hift_research.data_acquisition.range_zip import HTTPRangeReader
from hift_research.data_acquisition.uav import extract_member
from hift_research.data_contract.screening import phash64, content_screen
from hift_research.evaluation.assets import read_json, sha256_file, write_json_exclusive
from .annotations import IDENTITY_SCOPE, parse_ground_truth, parse_seqinfo, selected_frame_ids

SCHEMA = "hift-temporal-source-v1"
SOURCE_PAGE = "https://motchallenge.net/data/MOT17/"
FORMAT_SOURCE = "https://arxiv.org/html/2010.07548v2#A3"


def restore_index_cache(reader, ledger):
    """Validate reused central-directory bytes before handing them to ZipFile."""
    for item in ledger["ranges"]:
        path = Path(item["cache_path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError("immutable source index cache changed")
        payload = path.read_bytes()
        if len(payload) != item["end"] - item["start"] + 1:
            raise ValueError("source index cache length mismatch")
        reader.cache.append((item["start"], payload))


def validate_selection(plan):
    if plan.get("status") != "frozen_before_images_and_pair_outcomes" or plan.get("production_training_authorized") is not False:
        raise ValueError("frozen preflight-only source selection required")
    if plan.get("total_original_frames", 0) > 160 or plan.get("initial_global_network_budget_bytes") != 128 * 1024 * 1024:
        raise ValueError("approved initial source bounds exceeded")
    seen, splits, count = set(), set(), 0
    for video in plan["videos"]:
        lineage = video["video_lineage"]
        if lineage in seen or video["split"] not in {"train", "development"}:
            raise ValueError("source lineage cannot overlap train/development")
        seen.add(lineage)
        splits.add(video["split"])
        expected = selected_frame_ids(video["seqinfo_expected"]["totalframes"])
        if [f["frame_id"] for f in video["frames"]] != expected:
            raise ValueError("source frame IDs differ from preregistered time-window rule")
        count += len(expected)
    if count != plan["total_original_frames"] or splits != {"train", "development"}:
        raise ValueError("incomplete source split/frame plan")
    return plan


def acquire_temporal_subset(plan_path, index_ledger_path, reservations_path, data_root, output):
    plan = validate_selection(read_json(plan_path))
    index_ledger = read_json(index_ledger_path)
    if sha256_file(plan["inventory_path"]) != plan["inventory_sha256"]:
        raise ValueError("frozen source inventory changed")
    inventory = read_json(plan["inventory_path"])
    if (plan["source"] != inventory["source"] or plan["source"] != index_ledger["source"]
            or plan["source"]["url"] != "https://motchallenge.net/data/MOT17.zip"
            or plan["source"]["size"] != 5860214001):
        raise ValueError("source selection/index must bind the probed official MOT17 archive")
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if data_root.exists() or output.exists():
        raise ValueError("fresh immutable data and receipt directories required")
    reservations = read_json(reservations_path)
    if not reservations.get("frames") or not reservations.get("reserved_family_lineages"):
        raise ValueError("known evaluation content/lineage reservation screen required")
    output.mkdir(parents=True)
    data_root.mkdir(parents=True)
    source = plan["source"]
    remaining = plan["initial_global_network_budget_bytes"] - index_ledger["network_bytes"] - plan["reserved_documentation_and_probe_allowance_bytes"]
    if plan["image_span_bytes_estimate"] > remaining:
        raise ValueError("preregistered image estimate exceeds remaining global budget")

    def stable_opener(request, timeout):
        response = urllib.request.urlopen(request, timeout=timeout)
        if response.headers.get("ETag") != source["etag"]:
            response.close()
            raise ValueError("source ETag changed; no full-body fallback")
        return response

    reader = HTTPRangeReader(source["url"], source["size"], budget_bytes=remaining,
                             cache_dir=output / "range_cache", opener=stable_opener)
    reader.last_modified = source["last_modified"]
    restore_index_cache(reader, index_ledger)
    by_name = {m["name"]: m for m in inventory["members"]}
    offsets = sorted(m["header_offset"] for m in inventory["members"])
    next_offset = dict(zip(offsets, offsets[1:]))
    videos, all_hashes = [], set()
    try:
        with zipfile.ZipFile(reader) as archive:
            for chosen in plan["videos"]:
                seq = chosen["author_sequence_id"]
                prefix = f"MOT17/train/{seq}"
                original = data_root / "original"
                metadata = {}
                for suffix in ("seqinfo.ini", "gt/gt.txt"):
                    member = f"{prefix}/{suffix}"
                    info = archive.getinfo(member)
                    end = next_offset.get(info.header_offset, archive.start_dir) - 1
                    reader.prefetch(info.header_offset, end)
                    metadata[suffix] = extract_member(archive, info, original, max_bytes=16 * 1024 * 1024)
                seqinfo = parse_seqinfo(Path(metadata["seqinfo.ini"]["path"]).read_text(encoding="utf-8-sig"), seq)
                if any(seqinfo[k] != v for k, v in chosen["seqinfo_expected"].items()):
                    raise ValueError("original seqinfo differs from frozen publisher metadata")
                gt_path = Path(metadata["gt/gt.txt"]["path"])
                full_gt = parse_ground_truth(gt_path.read_text(encoding="utf-8-sig"), seqinfo)
                rows_by_frame = {}
                for row in full_gt["rows"]:
                    rows_by_frame.setdefault(row["frame_id"], []).append(row)
                frames, video_quarantine = [], list(full_gt["quarantine"])
                for frame in chosen["frames"]:
                    member = frame["archive_member"]
                    info = archive.getinfo(member)
                    if info.header_offset != frame["header_offset"] or f"{info.CRC:08x}" != frame["crc32"] or info.file_size != frame["size"]:
                        raise ValueError("selected archive member differs from frozen directory")
                    reader.prefetch(frame["header_offset"], frame["end_offset"])
                    record = extract_member(archive, info, original, max_bytes=16 * 1024 * 1024)
                    path = Path(record["path"])
                    with Image.open(path) as image:
                        if image.format != "JPEG":
                            raise ValueError("selected original frame is not JPEG")
                        image.load()
                        width, height = image.size
                    if (width, height) != (seqinfo["w"], seqinfo["h"]):
                        raise ValueError("original frame size differs from seqinfo")
                    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if decoded is None or decoded.shape[:2] != (height, width):
                        raise ValueError("second CPU decode failed/alignment differs")
                    perceptual = phash64(decoded)
                    reason, matched = content_screen(record["sha256"], perceptual, reservations)
                    if record["sha256"] in all_hashes:
                        reason = reason or "duplicate_selected_source_image_bytes"
                    all_hashes.add(record["sha256"])
                    quarantine_reasons = [reason] if reason else []
                    if reason:
                        video_quarantine.append({"frame_id": frame["frame_id"], "reason": reason,
                                                 "matched_reservation": matched})
                    alias_evidence = []
                    for detector in ("DPM", "FRCNN", "SDP"):
                        alias_name = member.replace(seq, f"{chosen['video_lineage']}-{detector}")
                        alias = by_name.get(alias_name)
                        if alias is None or alias["crc32"] != record["zip_crc32"] or alias["size"] != info.file_size:
                            raise ValueError("declared duplicate-detector frame alias does not match original CRC/size")
                        alias_evidence.append({"member": alias_name, "crc32": alias["crc32"], "size": alias["size"]})
                    annotations = rows_by_frame.get(frame["frame_id"], [])
                    for row in annotations:
                        row["identity"] = ["MOT17", chosen["video_lineage"], row["track_id"]]
                    frames.append({"frame_id": frame["frame_id"], "timestamp_seconds": (frame["frame_id"] - 1) / seqinfo["fps"],
                                   "source_image_id": f"MOT17/train/{chosen['video_lineage']}/{frame['frame_id']:06d}",
                                   "path": str(path), "sha256": record["sha256"], "phash64": perceptual,
                                   "width": width, "height": height, "zip_member": member, "zip_crc32": record["zip_crc32"],
                                   "byte_size": info.file_size, "annotations": annotations, "quarantine_reasons": quarantine_reasons,
                                   "screen_status": "quarantined" if reason else "clear_known_eval_reservations_unknown_semantic_aliases",
                                   "detector_alias_crc_size_evidence": alias_evidence})
                gt_aliases = []
                for detector in ("DPM", "FRCNN", "SDP"):
                    alias_name = f"MOT17/train/{chosen['video_lineage']}-{detector}/gt/gt.txt"
                    alias = by_name[alias_name]
                    if alias["crc32"] != metadata["gt/gt.txt"]["zip_crc32"] or alias["size"] != gt_path.stat().st_size:
                        raise ValueError("detector aliases have inconsistent original GT CRC/size")
                    gt_aliases.append({"member": alias_name, "crc32": alias["crc32"], "size": alias["size"]})
                videos.append({"video_lineage": chosen["video_lineage"], "author_sequence_id": seq,
                    "split": chosen["split"], "publisher_split": "train", "seqinfo": seqinfo,
                    "native_size": [seqinfo["w"], seqinfo["h"]], "coordinate_origin": 1, "bbox_format": "xywh",
                    "annotationboxconversion": "x-1,y-1,w,h", "identity_semantics": "source_verified_video_track",
                    "identity_scope": IDENTITY_SCOPE, "known_collection": chosen["known_collection"],
                    "reserved_aliases": chosen["reserved_aliases"],
                    "source_annotation_path": str(gt_path), "source_annotation_sha256": sha256_file(gt_path),
                    "original_GT": metadata["gt/gt.txt"], "original_seqinfo": metadata["seqinfo.ini"],
                    "source_row_count": full_gt["source_row_count"], "source_track_count": full_gt["source_track_count"],
                    "full_gt_missing_annotation_intervals": full_gt["missing_annotation_intervals"],
                    "quarantine": video_quarantine, "frames": frames, "gt_alias_crc_size_evidence": gt_aliases})
                print(f"Validated{chosen['video_lineage']} {chosen['split']}:40originalJPEGs, fullGT{full_gt['source_row_count']}rows; newbytes{reader.network_bytes}", flush=True)
        manifest = {"schema": SCHEMA, "frozen": True, "status": "frozen_preflight_only", "source_dataset": "MOT17/train",
            "created_utc": datetime.now(timezone.utc).isoformat(), "identity_scope": IDENTITY_SCOPE,
            "coordinate_origin": 1, "bbox_format": "xywh", "annotationboxconversion": "x-1,y-1,w,h",
            "source_provenance": {"publisher_page": SOURCE_PAGE, "format_and_identity_paper": FORMAT_SOURCE,
                "archive": source, "full_archive_sha256": None, "selection_path": str(Path(plan_path).resolve()),
                "selection_sha256": sha256_file(plan_path), "inventory_path": plan["inventory_path"], "inventory_sha256": plan["inventory_sha256"],
                "archive_integrity": "HTTPS_exact206_stable_ETag_LastModified_total_ZIP_CRC_selected_SHA256; no_fullarchive_SHA"},
            "source_video_alias_policy": {
                "DPM_FRCNN_SDP": "same_source_video_reserved_not_three_independent_videos",
                "MOT16": "same_base_video_reserved_but_updated_GT_numeric_trackIDs_NOT_equated",
                "conservative_MOT15_clip_alias_candidates_not_hash_verified": {"MOT17-02": "Venice-2", "MOT17-09": "ADL-Rundle-6", "MOT17-10": "ADL-Rundle-8", "MOT17-05": "ETH-Pedcross2"},
                "TRAIN_collection_clustering": "02_Vimeo_recording;09_10_conservative_ADL_family_not_independent_collection_count",
                "development_reserved_known_collection": ["ETH_Ess_aess_collection", "MOT17-05_all_detectors", "MOT16-05", "MOT17-06_all_detectors", "MOT16-06", "ETH-Pedcross2", "ETH-Bahnhof", "ETH-Sunnyday"],
                "evidence": [SOURCE_PAGE, "https://motchallenge.net/data/MOT15/", "https://arxiv.org/html/2010.07548v2"],
                "scope": "catalog_recording_alias_reservations_not_global_physical_person_identity"},
            "legal": {"license": "CC BY-NC-SA 3.0", "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
                "publisher_terms": SOURCE_PAGE, "additional_original_source_conditions_may_apply": True,
                "commercial_or_redistribution_rights_certified": False},
            "reservations_path": str(Path(reservations_path).resolve()), "reservations_sha256": sha256_file(reservations_path),
            "phash": reservations.get("phash"), "videos": videos,
            "validation": {"original_frames": sum(len(v["frames"]) for v in videos), "full_original_GT_retained": True,
                "all_selected_JPEGs_fully_decoded": True, "all_selected_frame_ids_align_seqinfo_and_original_GT": True,
                "presence_semantics": "bool_original_mark_equals1_active_source_flag_not_visibility",
                "interpolation": "forbidden", "production_training_authorized": False, "GPU_used": False},
            "budget": {"global_bound_bytes": plan["initial_global_network_budget_bytes"], "index_payload_bytes": index_ledger["network_bytes"],
                "acquisition_payload_bytes": reader.network_bytes, "documentation_probe_allowance_bytes": plan["reserved_documentation_and_probe_allowance_bytes"],
                "accounted_upper_payload_bytes": index_ledger["network_bytes"] + reader.network_bytes + plan["reserved_documentation_and_probe_allowance_bytes"]},
            "limitations": ["Only160original selected frames, not complete video images; complete original GT kept", "Author IDs are withinvideo trajectory fragments, not global person or biometric identity", "Different IDs across time/video do not imply different physical people", "No crossvideo negatives; sampling must verify simultaneous distinct tracks and final augmented crop excludes template object", "Known ETH/Ess DEV recording group held out; collection/person-disjointness beyond known catalog groups not proven", "Perceptual screens do not prove absence of semantic/transcoded/pretraining aliases", "Other detector images and MOT16 aliases reserved as sourcevideo lineage, not numerictrackID-equated", "Detector alias CRC/size evidence is not SHA verification of all unacquired variant bytes"]}
        write_json_exclusive(output / "temporal_manifest.json", manifest)
        return manifest
    finally:
        write_json_exclusive(output / "network_ledger.json", {"source": source, "network_bytes": reader.network_bytes,
            "index_network_bytes_already_accounted": index_ledger["network_bytes"], "reader_remaining_budget": remaining,
            "ranges": reader.records})
