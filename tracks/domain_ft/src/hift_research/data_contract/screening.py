"""Known eval lineage/content screens; perceptual similarity is not identity proof."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from .schema import file_hash
from .acquisition import write_exclusive_json


def phash64(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def content_screen(sha256, perceptual_hash, reservations, threshold=4):
    for row in reservations["frames"]:
        if sha256 == row["sha256"]:
            return "reserved_eval_exact_content", row
        if (int(perceptual_hash, 16) ^ int(row["phash64"], 16)).bit_count() <= threshold:
            return "reserved_eval_perceptual_near_match", row
    return None, None


def freeze_reservations(manifest_paths, lineages, output):
    frames, sources = [], []
    for path in manifest_paths:
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        root = Path(manifest["root"])
        sources.append({"path": str(path.resolve()), "sha256": file_hash(path), "dataset": manifest["dataset"]})
        for sequence in manifest["sequences"]:
            declared = sequence["frame_sha256"]
            if len(declared) != len(sequence["frame_paths"]):
                raise ValueError("reserved eval hashes/frame alignment invalid")
            for relative, expected in zip(sequence["frame_paths"], declared):
                image_path = root / relative
                sha = file_hash(image_path)
                if sha != expected:
                    raise ValueError("reserved eval source content changed")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("reserved frame failed CPU decode")
                frames.append({"dataset": manifest["dataset"], "sequence": sequence["name"],
                               "path": str(image_path.resolve()), "sha256": sha, "phash64": phash64(image)})
    value = {"schema": "hift-training-eval-reservations-v1", "source_manifests": sources,
             "reserved_family_lineages": sorted(set(lineages)), "frames": frames,
             "phash": {"implementation": "gray_AREA32_DCT8_median_excluding_DC_bits64", "quarantine_distance_lte": 4},
             "unknown_cross_dataset_semantic_aliases": "unresolved_not_certified_absent"}
    write_exclusive_json(output, value)
    return value
