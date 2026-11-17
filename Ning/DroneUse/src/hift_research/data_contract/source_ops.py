"""Execute pinned preprocessing/augmentation bodies with a LOCAL np.float alias.

No upstream file, global numpy attribute, import path, or model is modified.
"""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party/HiFT"


class NumpyCompat:
    def __getattr__(self, name):
        return float if name == "float" else getattr(np, name)


def symbols(relative, names, namespace):
    path = UPSTREAM / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if len(nodes) != len(names):
        raise ValueError("pinned source symbols changed")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def source_operations():
    bbox_path = UPSTREAM / "pysot/utils/bbox.py"
    bbox = {}
    exec(compile(bbox_path.read_text(encoding="utf-8"), str(bbox_path), "exec"), bbox)
    crop = symbols("training_dataset/coco/par_crop.py",
                   {"crop_hwc", "pos_s_2_bbox", "crop_like_SiamFC"}, {"np": NumpyCompat(), "cv2": cv2})
    namespace = {"np": NumpyCompat(), "cv2": cv2,
                 **{key: bbox[key] for key in ("Corner", "Center", "corner2center", "center2corner")}}
    augmentation = symbols("pysot/datasets/augmentation.py", {"Augmentation"}, namespace)["Augmentation"]
    tree = ast.parse((UPSTREAM / "pysot/datasets/dataset.py").read_text(encoding="utf-8"))
    dataset = next(node for node in tree.body if getattr(node, "name", None) == "TrkDataset")
    method = next(node for node in dataset.body if getattr(node, "name", None) == "_get_bbox")
    get_bbox_ns = {"np": np, "cfg": SimpleNamespace(TRAIN=SimpleNamespace(EXEMPLAR_SIZE=127)),
                   "Center": bbox["Center"], "center2corner": bbox["center2corner"]}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<pinned-_get_bbox>", "exec"), get_bbox_ns)
    class AuditedAugmentation(augmentation):
        def _crop_roi(self, image, box, out_sz, padding=(0, 0, 0)):
            x1, y1, x2, y2 = map(float, box)
            a, b = (out_sz-1)/(x2-x1), (out_sz-1)/(y2-y1)
            self.last_crop_geometry = {"input_shape": list(image.shape), "crop_xyxy": [x1, y1, x2, y2],
                                       "output_size": out_sz, "affine_2x3": [[a, 0., -a*x1], [0., b, -b*y1]],
                                       "padding": list(padding)}
            return super()._crop_roi(image, box, out_sz, padding)
    return SimpleNamespace(crop=crop["crop_like_SiamFC"], Augmentation=augmentation,
                           AuditedAugmentation=AuditedAugmentation,
                           get_bbox=get_bbox_ns["_get_bbox"], Corner=bbox["Corner"])


def preprocess_geometry(bbox_xywh):
    x, y, w, h = bbox_xywh
    sz = float(np.sqrt((h+.5*(w+h))*(w+.5*(w+h))))
    label_scale = 127/sz
    side = sz + 2*((511-127)/2/label_scale)
    cx, cy = x+w/2, y+h/2
    corners = [cx-side/2, cy-side/2, cx+side/2, cy+side/2]
    a = 510/side
    return {"source_crop_xyxy": corners, "source_crop_side_px": side, "output_size": 511,
            "affine_2x3": [[a, 0., -a*corners[0]], [0., a, -a*corners[1]]],
            "actual_source_to_crop_pixel_scale": a, "ideal_source_to_label_scale": label_scale,
            "retained_label_to_pixel_scale_ratio": label_scale/a,
            "padding": "original_image_channel_mean", "jpeg_quality": 95}


def jpeg_crop511(image, bbox_xywh, operations=None):
    operations = operations or source_operations()
    x, y, w, h = bbox_xywh
    corners = [x, y, x+w, y+h]
    _, crop = operations.crop(image, corners, instanc_size=511, padding=np.mean(image, axis=(0, 1)))
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("source-style crop JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != (511, 511, 3):
        raise ValueError("source-style crop JPEG decoding failed")
    return decoded, encoded.tobytes()
