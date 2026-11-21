"""Freeze a bounded official training subset and policy before preflight results."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import cv2
from .schema import SCHEMA, canonical_hash, file_hash, validate_training_manifest
from .screening import freeze_reservations, phash64, content_screen
from .source_ops import ROOT
from .acquisition import write_exclusive_json

SOURCE_FILES = [
    "third_party/HiFT/experiments/config.yaml", "third_party/HiFT/pysot/datasets/dataset.py",
    "third_party/HiFT/pysot/datasets/augmentation.py", "third_party/HiFT/training_dataset/coco/par_crop.py",
    "third_party/HiFT/training_dataset/coco/gen_json.py", "third_party/HiFT/pysot/utils/bbox.py",
    "src/hift_research/corrections/positive_targets.py", "src/hift_research/corrections/pair_targets.py",
    "src/hift_research/data_contract/pipeline.py", "src/hift_research/data_contract/schema.py",
    "src/hift_research/data_contract/source_ops.py", "src/hift_research/data_contract/screening.py",
    "src/hift_research/data_contract/freeze.py", "src/hift_research/data_contract/preflight.py",
    "src/hift_research/data_contract/acquisition.py", "src/hift_research/data_contract/torch_adapter.py",
    "scripts/preflight_training_data.py",
    "scripts/acquire_training_subset.py",
]


def freeze_training(acquisition_path, evaluation_manifests, lineages, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    acquisition_path = Path(acquisition_path)
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    selection_path = Path(acquisition["selection_path"])
    if file_hash(selection_path) != acquisition["selection_sha256"]:
        raise ValueError("acquisition preselection changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["split"] != "train2017":
        raise ValueError("validation/evaluation split is forbidden for training preflight")
    reservation_path = output / "reservations.json"
    reserved = freeze_reservations(evaluation_manifests, lineages, reservation_path)
    counts = Counter(row["sha256"] for row in acquisition["images"])
    images, quarantined = [], []
    for row in acquisition["images"]:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("acquired image changed before freeze")
        image = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        perceptual = phash64(image)
        reason, matched = content_screen(row["sha256"], perceptual, reserved)
        if counts[row["sha256"]] > 1:
            reason = "duplicate_train_source_content"
        if reason:
            quarantined.append({"image_id": row["id"], "reason": reason,
                                "matched_reserved_sequence": matched["sequence"] if matched else None})
            continue
        images.append({"image_id": row["id"], "path": row["path"], "sha256": row["sha256"],
                       "phash64": perceptual, "width": row["width"], "height": row["height"],
                       "split": "train2017", "license_id": row["license"],
                       "source_coco_url": row["coco_url"], "source_flickr_url": row.get("flickr_url"),
                       "screen_status": "clear_known_reservations_unknown_semantic_aliases"})
    allowed = {row["image_id"] for row in images}
    instances, duplicate_ids = [], Counter(row["id"] for row in selection["annotations"])
    for row in selection["annotations"]:
        if row["image_id"] not in allowed:
            continue
        if duplicate_ids[row["id"]] > 1:
            quarantined.append({"annotation_id": row["id"], "reason": "repeated_source_annotation_id"})
            continue
        instances.append({"annotation_id": row["id"], "image_id": row["image_id"],
                          "category_id": row["category_id"], "bbox_xywh": row["bbox"], "iscrowd": row["iscrowd"],
                          "identity": ["COCO2017/train2017", str(row["image_id"]), str(row["id"])],
                          "identity_scope": "annotation_instance_within_source_image", "quarantine_reasons": []})
    policy = {"seed": 20261023, "candidate_count": 1024, "candidate_negative_probability": .05,
              "replacement_policy": "none", "alias_unknown": "reject_negative",
              "same_frame_positive": "allowed_coco_still_image", "negative_cross_image": "reject_physical_alias_unknown",
              "same_image_distinct_instances": "allow_if_bbox_iou_below_0.9",
              "template_image_sampling": "uniform_active_images_then_uniform_source_instances",
              "temporal_positive_sampling": "same_source_annotation_frame000000_only",
              "preprocess": {"crop": "pinned_crop_like_SiamFC", "instance_size": 511, "context": .5,
                             "padding": "original_image_channel_mean", "jpeg_quality": 95},
              "template_aug": {"shift": 4, "scale": .05, "blur": 0., "flip": 0., "color": 1.},
              "search_aug": {"shift": 64, "scale": .18, "blur": .2, "flip": 0., "color": 1.},
              "gray": 0., "input": "BGR_CHW_float32_unnormalized", "grid": [63, 16, 11],
              "compatibility": "local_numpy_float_alias_only", "weight64": "retained_unchanged"}
    value = {"schema": SCHEMA, "dataset": "COCO2017", "split": "train2017", "status": "frozen_preflight_only",
             "source_provenance": {"acquisition_path": str(acquisition_path.resolve()),
                                   "acquisition_sha256": file_hash(acquisition_path),
                                   "selection_path": str(selection_path.resolve()),
                                   "selection_sha256": acquisition["selection_sha256"],
                                   "annotation_path": acquisition["annotation_path"],
                                   "annotation_sha256": acquisition["annotation_sha256"],
                                   "official_origin": acquisition["sources"][0]["url"]},
             "source_code_hashes": {str((ROOT/path).resolve()): file_hash(ROOT/path) for path in SOURCE_FILES},
             "licenses": selection["licenses"], "license_status": "source_terms_recorded_not_legal_opinion",
             "reservations_path": str(reservation_path.resolve()), "reservations_sha256": file_hash(reservation_path),
             "policy": policy, "policy_sha256": canonical_hash(policy), "images": images, "instances": instances,
             "quarantine": selection["quarantine"] + quarantined,
             "representation": {"dataset_count": 1, "still_image_video_equivalents": len(images),
                                "temporal_tracking_video_count": 0, "annotation_instances": len(instances),
                                "VID_GOT_YTBB": "not_registered_no_source_assets_in_bounded_scope"},
             "validation": {"production_training_authorized": False,
                            "known_eval_content_screen": "exact_sha256_and_phash64_lte4",
                            "cross_dataset_semantic_alias_freedom": "unknown_not_certified",
                            "no_real_sgd_or_gpu": True}}
    manifest_path = output / "training_manifest.json"
    write_exclusive_json(manifest_path, value)
    validate_training_manifest(manifest_path)
    return manifest_path
