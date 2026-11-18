"""Original MOT17 nine-column GT and seqinfo parsing, never inferred identity."""
from __future__ import annotations

import configparser
import csv
import math
import re

IDENTITY_SCOPE = "author_trajectory_fragment_within_source_video"


def parse_seqinfo(text, expected_name=None):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text)
    row = parser["Sequence"]
    name = row["name"]
    if expected_name is not None and name != expected_name:
        raise ValueError("original seqinfo name differs from source member")
    integers = {key: int(row[source]) for key, source in
                (("fps", "frameRate"), ("w", "imWidth"), ("h", "imHeight"), ("totalframes", "seqLength"))}
    if min(integers.values()) <= 0 or row["imDir"] != "img1" or row["imExt"].lower() != ".jpg":
        raise ValueError("invalid original sequence image/rate/dimension metadata")
    return {"name": name, **integers, "imdir": "img1", "extension": ".jpg"}


def selected_frame_ids(totalframes, window_length=20):
    """Preregistered20%/70% start positions, independent of GT/pair/model outcomes."""
    if totalframes < 2 * window_length or window_length <= 1:
        raise ValueError("sufficient complete video and nonzero temporal gaps required")
    starts = [1 + (totalframes - window_length) * numerator // 10 for numerator in (2, 7)]
    selected = [i for start in starts for i in range(start, start + window_length)]
    if len(set(selected)) != 2 * window_length or min(selected) < 1 or max(selected) > totalframes:
        raise ValueError("predeclared windows overlap or exceed source sequence")
    return selected


def native_to_zero_origin(box):
    if len(box) != 4 or not all(math.isfinite(x) for x in box) or min(box[2:]) <= 0:
        raise ValueError("finite positive original xywh required")
    return [box[0] - 1, box[1] - 1, box[2], box[3]]


def parse_ground_truth(text, seqinfo):
    """Retain every original finite source row and separately flag unsafe identities.

    MOT16/17 GT has9columns, unlike the generic10column submission format.
    Presence is bool(mark==1), an active author flag, NOT a visibility/class test
    or physical-absence claim. Full-trajectory gaps are reported, never
    interpolated or declared new people.
    """
    rows, groups, frame_ids = [], {}, {}
    quarantine = []
    for line, fields in enumerate(csv.reader(text.splitlines()), 1):
        if not fields or not any(v.strip() for v in fields):
            continue
        if len(fields) != 9:
            raise ValueError(f"MOT17 GT must have exactly9columns at source row{line}")
        try:
            values = [float(v) for v in fields]
        except ValueError as exc:
            raise ValueError(f"nonnumeric original GT row{line}") from exc
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"nonfinite original GT row{line}; no repairs")
        for index in (0, 1, 6, 7):
            if values[index] != int(values[index]):
                raise ValueError(f"integer source field required at row{line}")
        frame, track, mark, class_id = (int(values[i]) for i in (0, 1, 6, 7))
        if not 1 <= frame <= seqinfo["totalframes"]:
            raise ValueError(f"invalid original frame at row{line}")
        box = values[2:6]
        reasons = []
        if track < 1:
            reasons.append("nonpositive_source_track_id")
        if min(box[2:]) <= 0:
            reasons.append("nonpositive_source_box")
        if mark not in (0, 1):
            reasons.append("unknown_mark")
        if not 1 <= class_id <= 12:
            reasons.append("unknown_class")
        if not 0 <= values[8] <= 1:
            reasons.append("unknown_visibility")
        key = (frame, str(track))
        if key in frame_ids:
            quarantine.append({"track_id": str(track), "reason": "duplicate_frame_track_source_rows",
                               "source_rows": [frame_ids[key], line]})
        frame_ids[key] = line
        row = {"frame_id": frame, "track_id": str(track), "bbox_xywh_raw": box,
               "mark": mark, "class_id": class_id, "visibility": values[8], "presence": bool(mark == 1),
               "presence_semantics": "bool_original_mark_equals1_active_source_flag_not_visibility",
               "source_row": line, "identity_scope": IDENTITY_SCOPE, "quarantine_reasons": reasons}
        rows.append(row)
        groups.setdefault(str(track), []).append(row)
    if not rows:
        raise ValueError("complete original GT cannot be empty")
    gaps = []
    for track, observations in sorted(groups.items()):
        ordered = sorted(observations, key=lambda r: (r["frame_id"], r["source_row"]))
        if len({r["class_id"] for r in ordered}) > 1:
            quarantine.append({"track_id": track, "reason": "class_change_identity_ambiguity_hint"})
        for first, second in zip(ordered, ordered[1:]):
            gap = second["frame_id"] - first["frame_id"]
            if gap > 1:
                gaps.append({"track_id": track, "observed_endpoints": [first["frame_id"], second["frame_id"]],
                             "missing_intermediate_annotation_frames": gap - 1, "interpolated": False})
            if gap == 1 and min(first["bbox_xywh_raw"][2:] + second["bbox_xywh_raw"][2:]) > 0:
                a, b = first["bbox_xywh_raw"], second["bbox_xywh_raw"]
                distance = math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2)
                if distance > 5 * max(min(a[2:]), min(b[2:])):
                    quarantine.append({"track_id": track, "reason": "large_one_frame_geometric_jump_hint_not_physical_identity_proof",
                                       "observed_endpoints": [first["frame_id"], second["frame_id"]]})
    bad_tracks = {r["track_id"] for r in quarantine}
    for row in rows:
        if row["track_id"] in bad_tracks:
            row["quarantine_reasons"].append("full_source_track_ambiguity_hint")
    return {"rows": rows, "source_track_count": len(groups), "source_row_count": len(rows),
            "quarantine": quarantine, "missing_annotation_intervals": gaps,
            "identity_scope": IDENTITY_SCOPE,
            "identity_limitations": "Different IDs across time/video are NOT proof of distinct physical people; no biometric identity inference."}
