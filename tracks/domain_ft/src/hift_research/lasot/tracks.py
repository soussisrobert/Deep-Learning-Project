"""Single-object track loaders for the aerial-vehicle/pedestrian domain.

A track is {"name", "video", "source", "category", "frames": [Path], "boxes": float[N,4] xywh,
"valid": int indices}; MOT tracks also carry "frame_ids" (absolute video frame of each row; boxes of
non-valid rows are kept, zero when unannotated). Valid frames are visible, unoccluded-enough and of positive size; pair
sampling uses only valid frames. Negatives must come from a different "video" (MOT tracks from
the same video can appear inside each other's search crops).

Class maps (domain: car/van, truck/bus, pedestrian/people):
  VisDrone-MOT: 1 pedestrian, 2 people, 4 car, 5 van, 6 truck, 9 bus (score==1, occlusion<2, truncation==0)
  UAVDT MOT-D:  1 car, 2 truck, 3 bus (out-of-view==1 'no-out'; occlusion!=2, since the official README codes
                occlusion as 1 no, 2 LARGE, 3 medium, 4 small -- not ordered by severity)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

VISDRONE_DOMAIN = {1: "pedestrian", 2: "people", 4: "car", 5: "van", 6: "truck", 9: "bus"}
UAVDT_DOMAIN = {1: "car", 2: "truck", 3: "bus"}


def _flags(path, count):
    if not path.is_file():
        return np.zeros(count, dtype=bool)
    values = np.array([int(v) for v in path.read_text().replace("\n", ",").split(",") if v.strip()], dtype=int)
    if values.shape != (count,):
        raise ValueError(f"flag count mismatch {path}")
    return values.astype(bool)


def _sized(boxes, minimum):
    return np.isfinite(boxes).all(1) & (boxes[:, 2] >= minimum) & (boxes[:, 3] >= minimum)


def lasot_track(directory, minimum=2):
    directory = Path(directory)
    boxes = np.loadtxt(directory/"groundtruth.txt", delimiter=",", dtype=float, ndmin=2)
    frames = sorted((directory/"img").glob("*.jpg"))
    if boxes.shape[1] != 4 or len(frames) != len(boxes):
        raise ValueError(f"frame/annotation mismatch {directory}")
    hidden = _flags(directory/"full_occlusion.txt", len(boxes)) | _flags(directory/"out_of_view.txt", len(boxes))
    valid = np.flatnonzero(~hidden & _sized(boxes, minimum))
    return {"name": directory.name, "video": "LaSOT/"+directory.name, "source": "LaSOT",
            "category": directory.name.rsplit("-", 1)[0], "frames": frames, "boxes": boxes, "valid": valid}


def visdrone_sot_track(sequence_dir, annotation, minimum=2):
    sequence_dir = Path(sequence_dir)
    boxes = np.loadtxt(annotation, delimiter=",", dtype=float, ndmin=2)[:, :4]
    frames = sorted(sequence_dir.glob("*.jpg"))
    if len(frames) != len(boxes):
        raise ValueError(f"frame/annotation mismatch {sequence_dir}")
    return {"name": sequence_dir.name, "video": "VisDrone-SOT/"+sequence_dir.name, "source": "VisDrone-SOT",
            "category": "unlabeled", "frames": frames, "boxes": boxes, "valid": np.flatnonzero(_sized(boxes, minimum))}


def _mot_tracks(video, frames_by_id, rows, *, source, classes, min_length, minimum):
    """rows: [frame(1-based), track_id, x, y, w, h, keep(bool), class]; one track per id (per-frame majority class)."""
    tracks = []
    for track_id in np.unique(rows[:, 1]).astype(int):
        mine = rows[rows[:, 1] == track_id]
        category = int(np.bincount(mine[:, 7].astype(int)).argmax())
        if category not in classes:
            continue
        first, last = int(mine[:, 0].min()), int(mine[:, 0].max())
        span = np.arange(first, last+1)
        if any(f not in frames_by_id for f in span):
            continue
        boxes = np.full((len(span), 4), np.nan)
        keep = np.zeros(len(span), dtype=bool)
        for row in mine:
            i = int(row[0])-first
            boxes[i] = row[2:6]
            keep[i] = bool(row[6])
        valid = np.flatnonzero(keep & _sized(np.nan_to_num(boxes, nan=-1), minimum))
        if len(valid) < min_length:
            continue
        tracks.append({"name": f"{video}#{track_id}", "video": f"{source}/{video}", "source": source,
                       "category": classes[category], "frames": [frames_by_id[f] for f in span],
                       "boxes": np.nan_to_num(boxes, nan=0.0), "valid": valid, "frame_ids": span})
    return tracks


def visdrone_mot_tracks(sequence_dir, annotation, *, min_length=20, minimum=4):
    sequence_dir = Path(sequence_dir)
    frames_by_id = {int(p.stem): p for p in sequence_dir.glob("*.jpg")}
    raw = np.loadtxt(annotation, delimiter=",", dtype=float, ndmin=2)
    keep = (raw[:, 6] == 1) & (raw[:, 9] < 2) & (raw[:, 8] == 0)
    rows = np.column_stack([raw[:, 0], raw[:, 1], raw[:, 2:6], keep, raw[:, 7]])
    return _mot_tracks(sequence_dir.name, frames_by_id, rows, source="VisDrone-MOT", classes=VISDRONE_DOMAIN,
                       min_length=min_length, minimum=minimum)


def uavdt_tracks(sequence_dir, annotation, *, min_length=20, minimum=4):
    sequence_dir = Path(sequence_dir)
    frames_by_id = {int(p.stem.removeprefix("img")): p for p in sequence_dir.glob("img*.jpg")}
    raw = np.loadtxt(annotation, delimiter=",", dtype=float, ndmin=2)
    keep = (raw[:, 6] == 1) & (raw[:, 7] != 2)  # drop partially out-of-view and large occlusion (code 2)
    rows = np.column_stack([raw[:, 0], raw[:, 1], raw[:, 2:6], keep, raw[:, 8]])
    return _mot_tracks(sequence_dir.name, frames_by_id, rows, source="UAVDT", classes=UAVDT_DOMAIN,
                       min_length=min_length, minimum=minimum)
