"""Frozen, fail-closed real TRAIN preflight manifest contract (no SGD authority)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = "hift-training-data-v1"
FORBIDDEN_TRAIN_NAMES = ("UAV123", "UAV20L", "UAV123@10fps", "DTB70")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_manifest(path, *, verify_files=True):
    """Return validated preflight-only manifest; never upgrades training authority.

    Immutable file hashes and canonical source-instance IDs are checked. A source
    annotation ID is not a global physical identity or legal-use certification.
    """
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("dataset") != "COCO2017" or value.get("split") != "train2017":
        raise ValueError("only explicitly identified authentic COCO train2017 accepted")
    if value.get("status") != "frozen_preflight_only" or value.get("validation", {}).get("production_training_authorized") is not False:
        raise ValueError("manifest must remain explicitly preflight-only, not a training grant")
    policy = value.get("policy", {})
    if policy.get("candidate_negative_probability") != 0.05 or policy.get("replacement_policy") != "none":
        raise ValueError("candidate negative probability .05 and no replacements required")
    if policy.get("alias_unknown") != "reject_negative" or policy.get("same_frame_positive") != "allowed_coco_still_image":
        raise ValueError("explicit alias and still-image policies required")
    if value.get("policy_sha256") != canonical_hash(policy):
        raise ValueError("frozen policy hash mismatch")
    if (policy.get("template_aug") != {"shift": 4, "scale": .05, "blur": 0., "flip": 0., "color": 1.}
            or policy.get("search_aug") != {"shift": 64, "scale": .18, "blur": .2, "flip": 0., "color": 1.}
            or policy.get("grid") != [63, 16, 11] or policy.get("input") != "BGR_CHW_float32_unnormalized"
            or policy.get("same_image_distinct_instances") != "allow_if_bbox_iou_below_0.9"):
        raise ValueError("policy contradicts pinned source/B2 input recipe")
    if not isinstance(policy.get("seed"), int) or not 0 <= policy["seed"] < 2**32-1024 or policy.get("candidate_count") != 1024:
        raise ValueError("valid deterministic seed and exactly 1024 preregistered candidates required")
    for key in ("source_provenance", "source_code_hashes", "licenses", "reservations_sha256"):
        if not value.get(key):
            raise ValueError(f"missing provenance field {key}")
    reservations = Path(value["reservations_path"])
    if verify_files and file_hash(reservations) != value["reservations_sha256"]:
        raise ValueError("frozen reservations changed")
    reserved = json.loads(reservations.read_text(encoding="utf-8"))
    source = value["source_provenance"]
    selected_path = Path(source["selection_path"])
    if verify_files and file_hash(selected_path) != source["selection_sha256"]:
        raise ValueError("source annotation selection changed")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    source_instances = {row["id"]: row for row in selected["annotations"]}
    if selected.get("split") != "train2017":
        raise ValueError("source annotations are not TRAIN")
    if verify_files:
        for filename, expected in value["source_code_hashes"].items():
            if file_hash(filename) != expected:
                raise ValueError("frozen data contract source changed")
    images, image_ids, content = value.get("images", []), set(), set()
    if not images or len(images) > 32:
        raise ValueError("bounded nonempty subset (<=32 images) required")
    for image in images:
        if image.get("split") != "train2017" or image.get("image_id") in image_ids:
            raise ValueError("duplicate/wrong-split image")
        image_ids.add(image["image_id"])
        if image.get("sha256") in content:
            raise ValueError("duplicate source image bytes across distinct records")
        content.add(image["sha256"])
        if image.get("screen_status") != "clear_known_reservations_unknown_semantic_aliases":
            raise ValueError("unscreened/overlapping source image")
        if min(image.get("width", 0), image.get("height", 0)) <= 0:
            raise ValueError("invalid image dimensions")
        if verify_files:
            if file_hash(image["path"]) != image["sha256"]:
                raise ValueError("source image hash mismatch")
            import cv2
            from .screening import phash64, content_screen
            decoded = cv2.imread(image["path"], cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape[:2] != (image["height"], image["width"]):
                raise ValueError("source image decode/dimensions changed")
            perceptual = phash64(decoded)
            if perceptual != image["phash64"] or content_screen(image["sha256"], perceptual, reserved)[0]:
                raise ValueError("source perceptual/content reservation overlap")
    ids = set()
    for instance in value.get("instances", []):
        if instance.get("image_id") not in image_ids or instance.get("annotation_id") in ids:
            raise ValueError("missing image or repeated annotation ID")
        ids.add(instance["annotation_id"])
        annotation = source_instances.get(instance["annotation_id"])
        if (annotation is None or annotation["image_id"] != instance["image_id"]
                or annotation["bbox"] != instance["bbox_xywh"]
                or annotation.get("category_id") != instance.get("category_id")):
            raise ValueError("manifest instance contradicts authenticated source annotation")
        box = np.asarray(instance.get("bbox_xywh"), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0:
            raise ValueError("malformed source bbox")
        expected = ["COCO2017/train2017", str(instance["image_id"]), str(instance["annotation_id"])]
        if instance.get("identity") != expected or instance.get("identity_scope") != "annotation_instance_within_source_image":
            raise ValueError("identity must use canonical source annotation/image IDs")
        if instance.get("iscrowd") != 0 or instance.get("quarantine_reasons"):
            raise ValueError("ambiguous/crowd/quarantined instance cannot be active")
    if not ids:
        raise ValueError("no active canonical instances")
    return value
