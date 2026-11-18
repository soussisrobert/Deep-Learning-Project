"""Deterministic real-data candidate path: discovery->pair->crop->aug->B2 targets.

CPU arrays only. No torch import, model forward, optimizer, or SGD.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np

from hift_research.corrections.positive_targets import CorrectedAnchorTarget, robust_pair_targets
from .schema import validate_training_manifest, file_hash
from .source_ops import source_operations, jpeg_crop511, preprocess_geometry


@contextmanager
def numpy_seed(seed):
    """Serial replay only; temporarily isolate pinned source's global np.random."""
    old = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(old)


def array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def box_iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax+aw, bx+bw)-max(ax, bx))*max(0, min(ay+ah, by+bh)-max(ay, by))
    return intersection/(aw*ah+bw*bh-intersection)


class RealPairPipeline:
    def __init__(self, frozen_manifest_path, reservations_path=None):
        self.manifest_path = Path(frozen_manifest_path)
        self.manifest = validate_training_manifest(self.manifest_path)
        self.manifest_sha256 = file_hash(self.manifest_path)
        if reservations_path is not None and file_hash(reservations_path) != self.manifest["reservations_sha256"]:
            raise ValueError("reservations differ from frozen manifest")
        self.ops = source_operations()
        self.images = {row["image_id"]: row for row in self.manifest["images"]}
        self.grouped = {image_id: [] for image_id in sorted(self.images)}
        for instance in self.manifest["instances"]:
            self.grouped[instance["image_id"]].append(instance)
        if any(not rows for rows in self.grouped.values()):
            raise ValueError("active source image without usable instances")
        self.image_ids = sorted(self.grouped)
        for rows in self.grouped.values():
            rows.sort(key=lambda row: row["annotation_id"])
        self.anchor = CorrectedAnchorTarget(SimpleNamespace(TRAIN=SimpleNamespace(
            SEARCH_SIZE=287, OUTPUTFEATURE_SIZE=11, LABEL_RANGE=4)))
        self.template_aug = self.ops.AuditedAugmentation(4, .05, 0., 0., 1.)
        self.search_aug = self.ops.AuditedAugmentation(64, .18, .2, 0., 1.)
        self.crop_cache = {}
        self.decoded = {}

    def _random_instance(self):
        image_id = self.image_ids[np.random.randint(0, len(self.image_ids))]
        rows = self.grouped[image_id]
        return rows[np.random.randint(0, len(rows))]

    def _prepared(self, instance):
        ann_id = instance["annotation_id"]
        if ann_id not in self.crop_cache:
            image_id = instance["image_id"]
            if image_id not in self.decoded:
                image = cv2.imread(self.images[image_id]["path"], cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("source image disappeared/cannot decode")
                self.decoded[image_id] = image
            crop, jpeg = jpeg_crop511(self.decoded[image_id], instance["bbox_xywh"], self.ops)
            self.crop_cache[ann_id] = crop, hashlib.sha256(jpeg).hexdigest()
        return self.crop_cache[ann_id]

    def candidate(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("candidate index must be a nonnegative integer")
        seed = self.manifest["policy"]["seed"] + index
        record = {"candidate_index": index, "seed": seed, "manifest_sha256": self.manifest_sha256,
                  "replacement": None, "replacement_count": 0, "status": "rejected"}
        with numpy_seed(seed):
            negative = bool(np.random.random() < .05)
            template = self._random_instance()
            search = self._random_instance() if negative else template
            record.update(candidate_negative=negative, template_annotation_id=template["annotation_id"],
                          search_annotation_id=search["annotation_id"], template_image_id=template["image_id"],
                          search_image_id=search["image_id"], frame_interval=0 if search["image_id"] == template["image_id"] else None,
                          temporal_motion_source_px=0 if not negative else None,
                          repeated_frame=template["image_id"] == search["image_id"],
                          dataset="COCO2017/train2017", pair_type="negative_candidate" if negative else "same_frame_positive",
                          template_identity=template["identity"], search_identity=search["identity"],
                          template_identity_key=f"COCO2017/train2017/image:{template['image_id']}/annotation:{template['annotation_id']}",
                          search_identity_key=f"COCO2017/train2017/image:{search['image_id']}/annotation:{search['annotation_id']}",
                          template_source_bbox_xywh=template["bbox_xywh"], search_source_bbox_xywh=search["bbox_xywh"],
                          template_preprocess_geometry=preprocess_geometry(template["bbox_xywh"]),
                          search_preprocess_geometry=preprocess_geometry(search["bbox_xywh"]),
                          template_augmented_bbox=None, search_augmented_bbox=None,
                          template_augmentation_geometry=None, search_augmentation_geometry=None,
                          identity_scope="annotation_instance_within_source_image")
            if negative:
                if template["annotation_id"] == search["annotation_id"]:
                    record["reason"] = "same_instance_negative_collision"
                    return record, None
                if template["image_id"] != search["image_id"]:
                    record["reason"] = "cross_image_physical_alias_unknown"
                    return record, None
                if box_iou_xywh(template["bbox_xywh"], search["bbox_xywh"]) >= .9:
                    record["reason"] = "ambiguous_overlapping_annotation_instances"
                    return record, None
            try:
                template_crop, template_crop_sha = self._prepared(template)
                search_crop, search_crop_sha = self._prepared(search)
                def centered_box(instance, crop):
                    x, y, w, h = instance["bbox_xywh"]
                    return self.ops.get_bbox(None, crop, [x, y, x+w, y+h])
                template_box = centered_box(template, template_crop)
                search_box = centered_box(search, search_crop)
                template_array, template_augmented_bbox = self.template_aug(template_crop, template_box, 127, gray=False)
                search_array, bbox = self.search_aug(search_crop, search_box, 287, gray=False)
                record.update(template_augmented_bbox=list(map(float, template_augmented_bbox)),
                              search_augmented_bbox=list(map(float, bbox)),
                              template_centered_crop_bbox=list(map(float, template_box)),
                              search_centered_crop_bbox=list(map(float, search_box)),
                              template_augmentation_geometry=self.template_aug.last_crop_geometry,
                              search_augmentation_geometry=self.search_aug.last_crop_geometry)
                labels = robust_pair_targets(self.anchor, bbox, 11, is_negative=negative,
                                             template_identity=tuple(template["identity"]),
                                             search_identity=tuple(search["identity"]))
                arrays = dict(zip(("label_cls1", "labelxff", "labelcls2", "weightxff"), labels))
                arrays.update(template=template_array.transpose(2, 0, 1).astype(np.float32),
                              search=search_array.transpose(2, 0, 1).astype(np.float32),
                              bbox=np.asarray(bbox, dtype=np.float64))
                if any(not np.isfinite(array).all() for array in arrays.values()):
                    raise ValueError("nonfinite model input/targets")
                expected_shapes = {"template": (3,127,127), "search": (3,287,287), "bbox": (4,),
                                   "label_cls1": (1,11,11), "labelxff": (4,11,11),
                                   "labelcls2": (1,11,11), "weightxff": (1,11,11)}
                expected_types = {"template": np.float32, "search": np.float32, "bbox": np.float64,
                                  "label_cls1": np.float64, "labelxff": np.float32,
                                  "labelcls2": np.float64, "weightxff": np.float64}
                if any(arrays[key].shape != shape or arrays[key].dtype != expected_types[key]
                       for key, shape in expected_shapes.items()):
                    raise ValueError("model input/target shapes or dtypes mismatch")
                weight = arrays["weightxff"]
                cls1, cls2 = arrays["label_cls1"], arrays["labelcls2"]
                if np.any(weight < 0) or not np.isin(cls1, [-2, -1, 0, 1]).all() or np.any((cls2 < 0) | (cls2 > 1)):
                    raise ValueError("target domain violation")
                if negative and (np.any(weight) or np.any(cls1) or np.any(cls2)):
                    raise ValueError("negative supervision mask violation")
                if not negative and (not np.any(weight > 0) or not np.any(cls1 == 1) or not np.any(cls2 > 0)):
                    raise ValueError("positive target has empty supervision support")
                source_box = search["bbox_xywh"]
                source_image = self.images[search["image_id"]]
                sx, sy, sw, sh = source_box
                source_crop_side = record["search_preprocess_geometry"]["source_crop_side_px"]
                visible_width = max(0., min(sx+sw, source_image["width"])-max(0., sx))
                visible_height = max(0., min(sy+sh, source_image["height"])-max(0., sy))
                centered = np.asarray(search_box)
                index_box = np.int32(np.minimum(10, np.maximum(0, (np.asarray(bbox)-63)/16)))
                collapsed = bool(index_box[0] == index_box[2] and index_box[1] == index_box[3])
                record.update(status="accepted", reason=None, array_hashes={key: array_digest(value) for key, value in arrays.items()},
                              template_crop511_jpeg_sha256=template_crop_sha, search_crop511_jpeg_sha256=search_crop_sha,
                              source_object_size_px=source_box[2:], source_image_size_px=[source_image["width"], source_image["height"]],
                              source_crop511_side_px=source_crop_side, source_to_crop511_scale=510/source_crop_side,
                              input_contract={key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key,value in arrays.items()},
                              label_domains={"cls1": [-2,-1,0,1], "cls2": [0,1], "weightxff": "finite_nonnegative"},
                              genuine_visible_object_support_px2=visible_width*visible_height,
                              centered_crop_object_size_px=[float(centered[2]-centered[0]), float(centered[3]-centered[1])],
                              augmented_object_size_px=[float(bbox.x2-bbox.x1), float(bbox.y2-bbox.y1)],
                              regression_support_cells=int(np.count_nonzero(weight)),
                              cls1_positive_cells=int(np.count_nonzero(cls1 == 1)),
                              cls1_background_cells=int(np.count_nonzero(cls1 == 0)),
                              cls1_ignored_cells=int(np.count_nonzero(cls1 < 0)),
                              cls2_positive_cells=int(np.count_nonzero(cls2)),
                              max_regression_weight=float(weight.max()), weight64_cells=int(np.count_nonzero(weight == 64)),
                              regression_weights_above_1_5=int(np.count_nonzero(weight > 1.5)),
                              collapsed_support_fallback=collapsed and not negative)
                return record, arrays
            except ValueError as exc:
                record["reason"] = "target_or_input_rejected:" + str(exc)
                return record, None
