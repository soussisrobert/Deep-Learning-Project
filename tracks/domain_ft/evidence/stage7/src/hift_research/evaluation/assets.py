"""Explicit local-only asset imports; no network and no checkpoint unpickling."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil

from .metrics import ValidationError, boxes_array, valid_boxes

UPSTREAM_SHA = "992aa7c1605360d607af84cb804b93cd8ee81caf"
DATASET_COUNTS = {"UAV123": 123, "UAV123@10fps": 123, "UAV20L": 20, "DTB70": 70}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValidationError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique)


def write_json_exclusive(path, value):
    """Never replace an existing manifest/result; validate JSON before creating."""
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", name) or name in {".", ".."}:
        raise ValidationError(f"unsafe sequence name: {name!r}")
    return name


def resolve_frame(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValidationError("frame paths must use relative POSIX spelling")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative or not path.parts:
        raise ValidationError(f"unsafe frame path: {relative!r}")
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValidationError(f"frame escapes dataset root: {relative}")
    if not target.is_file() or target.stat().st_size == 0:
        raise ValidationError(f"missing/empty frame: {target}")
    return target


def read_boxes(path):
    """Comma or whitespace Nx4 only; blank/malformed rows are not dropped."""
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            stripped = line.strip()
            fields = [part.strip() for part in stripped.split(",")] if "," in stripped else stripped.split()
            if len(fields) != 4 or any(not field for field in fields):
                raise ValidationError(f"{path}:{number}: expected four xywh fields")
            try:
                rows.append([float(x) for x in fields])
            except ValueError as exc:
                raise ValidationError(f"{path}:{number}: nonnumeric row") from exc
    return boxes_array(rows, str(path))


def _clean_boxes(boxes):
    return [[float(x) if math.isfinite(x) else None for x in row] for row in boxes]


def _image_paths(directory):
    if not directory.is_dir():
        raise ValidationError(f"missing image directory: {directory}")
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    # Numeric order without guessing sampling frequency or segment boundaries.
    def key(path):
        parts = [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]
        return parts, path.name  # Deterministic tie-break for 1.jpg versus 01.jpg.
    return sorted(files, key=key)


def import_dataset(root, dataset, *, allow_partial=False, invalid_policy="error", hash_frames=False):
    """Build a manifest from toolkit UAV123 JSON or explicitly prepared raw trees.

    UAV123 JSON is mandatory: raw shared UAV folders alone cannot establish the
    official 123 segment boundaries. Other raw trees must already contain the
    official variant sampling/sequence segmentation; no temporal resampling here.
    """
    root = Path(root).resolve()
    if dataset not in DATASET_COUNTS or not root.is_dir():
        raise ValidationError("unknown dataset or nonexistent dataset root")
    if invalid_policy not in {"error", "count_as_failure"}:
        raise ValidationError("unknown invalid-frame policy")
    metadata_path = root / "UAV123.json"
    sequences, annotation_hashes = [], {}
    if dataset == "UAV123":
        if not metadata_path.is_file():
            raise ValidationError("UAV123 requires toolkit-compatible UAV123.json with exact segment frame lists")
        metadata = read_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ValidationError("UAV123.json must map sequence names to metadata")
        annotation_hashes[metadata_path.name] = sha256_file(metadata_path)
        for name, row in sorted(metadata.items()):
            if not isinstance(row, dict) or not {"img_names", "gt_rect", "init_rect"} <= row.keys():
                raise ValidationError(f"{name}: incomplete UAV123 metadata")
            sequences.append({"name": name, "frame_paths": row["img_names"],
                              "ground_truth": row["gt_rect"], "init_box": row["init_rect"]})
    else:
        base = root if dataset == "DTB70" else root / "data_seq"
        if not base.is_dir():
            raise ValidationError(f"missing sequence directory: {base}")
        for directory in sorted(p for p in base.iterdir() if p.is_dir()):
            name = safe_name(directory.name)
            image_dir = directory / "img" if dataset == "DTB70" else directory
            annotation = directory / "groundtruth_rect.txt" if dataset == "DTB70" else root / "anno" / (name + ".txt")
            if not annotation.is_file():
                raise ValidationError(f"missing annotation: {annotation}")
            boxes = read_boxes(annotation)
            annotation_hashes[annotation.relative_to(root).as_posix()] = sha256_file(annotation)
            sequences.append({"name": name, "frame_paths": [p.relative_to(root).as_posix() for p in _image_paths(image_dir)],
                              "ground_truth": _clean_boxes(boxes), "init_box": _clean_boxes(boxes[:1])[0]})
        if dataset != "DTB70":
            annotations = {p.stem for p in (root / "anno").glob("*.txt")}
            if annotations != {r["name"] for r in sequences}:
                raise ValidationError("annotation/image sequence-name sets differ")
    manifest = {"schema_version": 1, "kind": "dataset", "dataset": dataset, "root": str(root),
                "protocol": "ope_all_frames_sequence_macro", "upstream_sha": UPSTREAM_SHA,
                "completeness": "partial" if allow_partial else "expected_count_only",
                "expected_sequence_count": DATASET_COUNTS[dataset], "invalid_policy": invalid_policy,
                "frame_content_identity": "sha256" if hash_frames else "paths_and_sizes_only",
                "annotation_sha256": annotation_hashes, "sequences": sequences,
                "official_membership_verified": False}
    validate_dataset_manifest(manifest)
    for sequence in sequences:
        boxes = boxes_array(sequence["ground_truth"])
        sequence["ground_truth"] = _clean_boxes(boxes)
        sequence["invalid_ground_truth_frames"] = int((~valid_boxes(boxes)).sum())
        sequence["frame_sizes"] = [resolve_frame(root, p).stat().st_size for p in sequence["frame_paths"]]
        if hash_frames:
            sequence["frame_sha256"] = [sha256_file(resolve_frame(root, p)) for p in sequence["frame_paths"]]
    return manifest


def validate_dataset_manifest(manifest, *, verify_files=True):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("kind") != "dataset":
        raise ValidationError("unsupported dataset manifest")
    dataset = manifest.get("dataset")
    if dataset not in DATASET_COUNTS or manifest.get("protocol") != "ope_all_frames_sequence_macro":
        raise ValidationError("unknown dataset/protocol")
    policy = manifest.get("invalid_policy")
    if policy not in {"error", "count_as_failure"}:
        raise ValidationError("manifest missing explicit invalid-frame policy")
    if manifest.get("completeness") not in {"partial", "expected_count_only"}:
        raise ValidationError("manifest missing explicit completeness")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValidationError("dataset has no sequences")
    if manifest["completeness"] != "partial" and len(sequences) != DATASET_COUNTS[dataset]:
        raise ValidationError(f"expected {DATASET_COUNTS[dataset]} sequences, got {len(sequences)}; use explicit partial import for fixtures")
    names = set()
    if not isinstance(manifest.get("root"), str) or not Path(manifest["root"]).is_absolute():
        raise ValidationError("manifest root must be an explicit absolute local path")
    root = Path(manifest["root"]).resolve()
    if verify_files:
        for annotation, expected_hash in manifest.get("annotation_sha256", {}).items():
            if sha256_file(resolve_frame(root, annotation)) != expected_hash:
                raise ValidationError(f"annotation hash changed: {annotation}")
    for row in sequences:
        if not isinstance(row, dict):
            raise ValidationError("each sequence must be a metadata object")
        name = safe_name(row.get("name"))
        if name.casefold() in names:
            raise ValidationError(f"duplicate/case-colliding sequence: {name}")
        names.add(name.casefold())
        boxes = boxes_array(row.get("ground_truth"), f"{name} GT")
        valid = valid_boxes(boxes)
        if policy == "error" and not valid.all():
            raise ValidationError(f"{name}: invalid GT frames {list(map(int, (~valid).nonzero()[0]))}")
        initial = boxes_array([row.get("init_box")], f"{name} initialization")
        if not valid_boxes(initial).all() or not valid[0] or not (initial[0] == boxes[0]).all():
            raise ValidationError(f"{name}: initialization must be valid and exactly equal first GT row")
        frames = row.get("frame_paths")
        if not isinstance(frames, list) or len(frames) != len(boxes) or any(not isinstance(x, str) for x in frames):
            raise ValidationError(f"{name}: frame/GT count mismatch or malformed paths")
        if len(set(x.casefold() for x in frames)) != len(frames):
            raise ValidationError(f"{name}: duplicate/case-colliding frame paths")
        for field in ("frame_sizes", "frame_sha256"):
            if field in row and len(row[field]) != len(frames):
                raise ValidationError(f"{name}: {field} length mismatch")
        if verify_files:
            for index, frame in enumerate(frames):
                target = resolve_frame(root, frame)
                if "frame_sizes" in row and target.stat().st_size != row["frame_sizes"][index]:
                    raise ValidationError(f"{name}: frame size changed: {frame}")
                if "frame_sha256" in row and sha256_file(target) != row["frame_sha256"][index]:
                    raise ValidationError(f"{name}: frame hash changed: {frame}")
    return manifest


def import_checkpoint(source, destination, *, expected_sha256):
    """Copy only an explicitly hash-pinned local file; do not deserialize it."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValidationError("expected_sha256 must contain 64 hex digits")
    if not source.is_file() or not source.stat().st_size:
        raise ValidationError("checkpoint source missing/empty")
    digest = sha256_file(source)
    if digest != expected_sha256.lower():
        raise ValidationError("checkpoint SHA-256 does not match expected hash")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha256_file(destination) != digest:
        raise ValidationError("copied checkpoint hash mismatch")
    return {"schema_version": 1, "kind": "checkpoint", "path": str(destination),
            "source": str(source), "sha256": digest, "size_bytes": destination.stat().st_size,
            "state_dict_validated": False, "official_identity_verified": False,
            "warning": "Hash integrity is not authorship or model-key compatibility; inference must strictly validate keys."}
