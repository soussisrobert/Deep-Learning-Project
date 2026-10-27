"""Extra raw-tree evaluation datasets, registered at runtime (evaluation/assets.py stays frozen).

Importing this module adds each dataset's expected sequence count to assets.DATASET_COUNTS so the
unchanged validator and scripts/evaluate.py (via scripts/evaluate_extra.py) accept its manifests.
"""
from pathlib import Path

from . import assets
from .assets import (UPSTREAM_SHA, ValidationError, _clean_boxes, _image_paths, read_boxes, resolve_frame,
                     safe_name, sha256_file, validate_dataset_manifest)
from .metrics import boxes_array, valid_boxes

# name: (expected sequence count, sequence directory, annotation directory) relative to the dataset root
EXTRA_LAYOUTS = {"VisDrone2019-SOT-test-dev": (35, "sequences", "annotations")}
for _name, (_count, _, _) in EXTRA_LAYOUTS.items():
    assets.DATASET_COUNTS.setdefault(_name, _count)


def import_extra_dataset(root, dataset, *, invalid_policy="count_as_failure", hash_frames=True):
    """Same manifest schema as assets.import_dataset for a raw `<seq dir>/<name>/*.jpg` + `<anno dir>/<name>.txt` tree."""
    root = Path(root).resolve()
    if dataset not in EXTRA_LAYOUTS or not root.is_dir():
        raise ValidationError("unknown extra dataset or nonexistent dataset root")
    count, sequence_dir, annotation_dir = EXTRA_LAYOUTS[dataset]
    sequences, annotation_hashes = [], {}
    for directory in sorted(p for p in (root/sequence_dir).iterdir() if p.is_dir()):
        name = safe_name(directory.name)
        annotation = root/annotation_dir/(name+".txt")
        if not annotation.is_file():
            raise ValidationError(f"missing annotation: {annotation}")
        boxes = read_boxes(annotation)
        annotation_hashes[annotation.relative_to(root).as_posix()] = sha256_file(annotation)
        sequences.append({"name": name, "frame_paths": [p.relative_to(root).as_posix() for p in _image_paths(directory)],
                          "ground_truth": _clean_boxes(boxes), "init_box": _clean_boxes(boxes[:1])[0]})
    if {p.stem for p in (root/annotation_dir).glob("*.txt")} != {r["name"] for r in sequences}:
        raise ValidationError("annotation/image sequence-name sets differ")
    manifest = {"schema_version": 1, "kind": "dataset", "dataset": dataset, "root": str(root),
                "protocol": "ope_all_frames_sequence_macro", "upstream_sha": UPSTREAM_SHA,
                "completeness": "expected_count_only", "expected_sequence_count": count, "invalid_policy": invalid_policy,
                "frame_content_identity": "sha256" if hash_frames else "paths_and_sizes_only",
                "annotation_sha256": annotation_hashes, "sequences": sequences, "official_membership_verified": False}
    validate_dataset_manifest(manifest)
    for sequence in sequences:  # same post-processing as assets.import_dataset
        boxes = boxes_array(sequence["ground_truth"])
        sequence["ground_truth"] = _clean_boxes(boxes)
        sequence["invalid_ground_truth_frames"] = int((~valid_boxes(boxes)).sum())
        sequence["frame_sizes"] = [resolve_frame(root, p).stat().st_size for p in sequence["frame_paths"]]
        if hash_frames:
            sequence["frame_sha256"] = [sha256_file(resolve_frame(root, p)) for p in sequence["frame_paths"]]
    return manifest
