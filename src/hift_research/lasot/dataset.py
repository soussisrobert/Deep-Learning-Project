"""Indexed LaSOT training pairs for B2 training, built from the audited source crop/aug/target ops.

Positive: same sequence, both frames visible (not fully occluded / out of view, positive size),
gap 1..max_gap frames. Negative (probability negative_probability): search from a DIFFERENT
sequence, so the template object cannot appear; dense-zero targets via robust_pair_targets.
Distractor negative (optional, probability distractor_probability, stage-2 P2): search centred on ANOTHER
same-group track of the SAME video at a frame within max_gap, accepted only when the template object's
annotated box at that frame lies entirely outside the 511-px source crop window; otherwise the pair falls
back to an ordinary different-video negative. distractor_probability=0 draws no extra random numbers.
Each index uses its own NumPy seed (seed+index), so a pair is reproducible per index.
Rejected candidates (e.g. no positive support) return None and are counted by the caller.
LaSOT boxes are used as published (1-based convention not shifted; <=1 px at 511 crop scale).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from torch.utils.data import Dataset

from hift_research.corrections.positive_targets import CorrectedAnchorTarget, robust_pair_targets
from hift_research.data_contract.pipeline import numpy_seed
from hift_research.data_contract.source_ops import jpeg_crop511, source_operations

TEMPLATE_AUG = {"shift": 4.0, "scale": 0.05, "blur": 0.0, "flip": 0.0, "color": 1.0}
SEARCH_AUG = {"shift": 64.0, "scale": 0.18, "blur": 0.2, "flip": 0.0, "color": 1.0}


def read_flags(path, count):
    if not path.is_file():
        return np.zeros(count, dtype=bool)
    values = np.array([int(v) for v in path.read_text().replace("\n", ",").split(",") if v.strip()], dtype=int)
    if values.shape != (count,):
        raise ValueError(f"flag count mismatch {path}")
    return values.astype(bool)


def load_sequence(directory):
    boxes = np.loadtxt(directory/"groundtruth.txt", delimiter=",", dtype=float, ndmin=2)
    if boxes.shape[1] != 4:
        raise ValueError(f"bad groundtruth {directory}")
    frames = sorted((directory/"img").glob("*.jpg"))
    if len(frames) != len(boxes):
        raise ValueError(f"frame/annotation count mismatch {directory}")
    hidden = read_flags(directory/"full_occlusion.txt", len(boxes)) | read_flags(directory/"out_of_view.txt", len(boxes))
    valid = ~hidden & np.isfinite(boxes).all(1) & (boxes[:, 2] > 1) & (boxes[:, 3] > 1)
    return {"name": directory.name, "video": "LaSOT/"+directory.name, "frames": frames, "boxes": boxes,
            "valid": np.flatnonzero(valid)}


def template_outside_crop(search_box, template_box):
    """True when template_box (xywh, positive size) lies entirely outside the 511-px source crop window centred
    on search_box (context size sqrt((w+p)(h+p)), p=(w+h)/2, scaled 511/127), which bounds every augmented
    287-px search region."""
    x, y, w, h = map(float, search_box)
    tx, ty, tw, th = map(float, template_box)
    if not (w > 0 and h > 0 and tw > 0 and th > 0):
        return False
    p = (w+h)/2
    half = np.sqrt((w+p)*(h+p))*511/127/2
    cx, cy = x+w/2, y+h/2
    return tx >= cx+half or tx+tw <= cx-half or ty >= cy+half or ty+th <= cy-half


class TrackPairDataset(Dataset):
    """Pairs over generic tracks (see tracks.py). Negatives always come from a different video."""
    def __init__(self, tracks, *, length, seed, negative_probability=0.05, max_gap=100, source_weights=None,
                 search_scale=None, distractor_probability=0.0):
        # search_scale: optional override of SEARCH_AUG["scale"] (default 0.18, the source pysot value)
        self.search_aug = dict(SEARCH_AUG, scale=search_scale) if search_scale is not None else dict(SEARCH_AUG)
        if not 0 <= self.search_aug["scale"] < 1:
            raise ValueError("search scale augmentation must be in [0, 1)")
        self.sequences = [t for t in tracks if len(t["valid"]) >= 2]
        # Optional: pick a source by weight, then a track uniformly within it (default: uniform over tracks).
        self._source_tracks = None
        if source_weights:
            sources = sorted(source_weights)
            members = [np.array([i for i, t in enumerate(self.sequences) if t["source"] == s]) for s in sources]
            if any(not len(m) for m in members) or any(t["source"] not in source_weights for t in self.sequences):
                raise ValueError("every weighted source needs tracks and every track needs a weighted source")
            weights = np.array([source_weights[s] for s in sources], dtype=float)
            self._source_tracks, self._source_p = members, weights/weights.sum()
        if len({t["video"] for t in self.sequences}) < 2:
            raise ValueError("at least two usable videos required (negatives come from another video)")
        if length < 1 or not 0 <= negative_probability < 1 or max_gap < 1:
            raise ValueError("invalid length/negative_probability/max_gap")
        if not 0 <= distractor_probability < 1:
            raise ValueError("distractor_probability must be in [0, 1)")
        self.distractor_probability = float(distractor_probability)
        self._same_video = None
        if self.distractor_probability:
            from hift_research.evaluation.validation import group_of
            buckets = {}
            for i, t in enumerate(self.sequences):
                if "frame_ids" in t and group_of(t) != "unlabeled":
                    buckets.setdefault((t["video"], group_of(t)), []).append(i)
            self._same_video = [np.array([], dtype=int) for _ in self.sequences]
            for members in buckets.values():
                for i in members:
                    self._same_video[i] = np.array([j for j in members if j != i], dtype=int)
        self.length, self.seed = length, seed
        self.negative_probability, self.max_gap = negative_probability, max_gap
        videos = [t["video"] for t in self.sequences]
        self._other_video = [np.array([j for j, v in enumerate(videos) if v != video]) for video in videos]
        self._ops = None

    def __len__(self):
        return self.length

    def _setup(self):
        if self._ops is None:
            self._ops = source_operations()
            self._anchor = CorrectedAnchorTarget(SimpleNamespace(TRAIN=SimpleNamespace(SEARCH_SIZE=287, OUTPUTFEATURE_SIZE=11, LABEL_RANGE=4)))
            order = ("shift", "scale", "blur", "flip", "color")  # positional, as in the temporal pipeline
            self._template_aug = self._ops.AuditedAugmentation(*(TEMPLATE_AUG[k] for k in order))
            self._search_aug = self._ops.AuditedAugmentation(*(self.search_aug[k] for k in order))

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_ops", "_anchor", "_template_aug", "_search_aug"):
            state.pop(key, None)
        state["_ops"] = None
        return state

    def _crop(self, sequence, frame):
        image = cv2.imread(str(sequence["frames"][frame]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot decode {sequence['frames'][frame]}")
        x, y, w, h = sequence["boxes"][frame]
        crop, _ = jpeg_crop511(image, [x, y, w, h], self._ops)
        return crop, self._ops.get_bbox(None, crop, [x, y, x+w, y+h])

    def _distractor(self, first, tf):
        """(track index, frame) of a same-video same-group distractor whose crop excludes the template object."""
        a = self.sequences[first]
        candidates = self._same_video[first]
        if not len(candidates):
            return None
        absolute = int(a["frame_ids"][tf])
        for _ in range(3):
            j = int(candidates[int(np.random.randint(len(candidates)))])
            b = self.sequences[j]
            valid = b["valid"]
            near = valid[np.abs(np.asarray(b["frame_ids"])[valid]-absolute) <= self.max_gap]
            if not len(near):
                continue
            sf = int(np.random.choice(near))
            ai = int(b["frame_ids"][sf])-int(a["frame_ids"][0])
            if not 0 <= ai < len(a["boxes"]):
                continue  # template object's location unknown at that frame
            if template_outside_crop(b["boxes"][sf], a["boxes"][ai]):
                return j, sf
        return None

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        self._setup()
        with numpy_seed(self.seed+int(index)):
            negative = bool(np.random.random() < self.negative_probability)
            distractor = bool(self.distractor_probability and np.random.random() < self.distractor_probability)
            if self._source_tracks is None:
                first = int(np.random.randint(len(self.sequences)))
            else:
                pool = self._source_tracks[int(np.random.choice(len(self._source_tracks), p=self._source_p))]
                first = int(pool[int(np.random.randint(len(pool)))])
            template_sequence = self.sequences[first]
            tf = int(np.random.choice(template_sequence["valid"]))
            picked = self._distractor(first, tf) if distractor else None
            negative = negative or distractor
            if picked is not None:
                search_sequence, sf = self.sequences[picked[0]], picked[1]
            elif negative:
                candidates = self._other_video[first]  # for one-track-per-video data: same draw as other+(other>=first)
                search_sequence = self.sequences[int(candidates[int(np.random.randint(len(candidates)))])]
                sf = int(np.random.choice(search_sequence["valid"]))
            else:
                search_sequence = template_sequence
                valid = template_sequence["valid"]
                near = valid[(np.abs(valid-tf) <= self.max_gap) & (valid != tf)]
                if not len(near):
                    return {"index": index, "negative": negative, "reason": "no_visible_partner"}, None
                sf = int(np.random.choice(near))
            record = {"index": index, "negative": negative, "template": [template_sequence["name"], tf],
                      "search": [search_sequence["name"], sf]}
            if distractor:
                record["distractor"] = picked is not None
            try:
                template_crop, tbox = self._crop(template_sequence, tf)
                search_crop, sbox = self._crop(search_sequence, sf)
                t, _ = self._template_aug(template_crop, tbox, 127, gray=False)
                s, sa = self._search_aug(search_crop, sbox, 287, gray=False)
                labels = robust_pair_targets(self._anchor, sa, 11, is_negative=negative,
                                             template_identity=("LaSOT", template_sequence["name"], "0"),
                                             search_identity=("LaSOT", search_sequence["name"], "0"))
            except (ValueError, cv2.error) as exc:
                return {**record, "reason": str(exc)}, None
        arrays = dict(zip(("label_cls1", "labelxff", "labelcls2", "weightxff"), labels))
        arrays.update(template=t.transpose(2, 0, 1).astype(np.float32), search=s.transpose(2, 0, 1).astype(np.float32))
        if not all(np.isfinite(v).all() for v in arrays.values()):
            return {**record, "reason": "nonfinite_arrays"}, None
        if not negative and not (np.any(arrays["label_cls1"] == 1) and np.any(arrays["weightxff"] > 0)):
            return {**record, "reason": "empty_positive_support"}, None
        return record, arrays


class LaSOTPairDataset(TrackPairDataset):
    """LaSOT training-split sequences under root/original (official zips hold sequence folders directly)."""
    def __init__(self, root, sequences, **kwargs):
        super().__init__([load_sequence(Path(root)/"original"/name) for name in sorted(sequences)], **kwargs)


def collate_pairs(items):
    """Stack accepted pairs into the engine's CPU model-input layout; count rejections."""
    import torch
    from hift_research.training.wave4_layout import canonicalize_model_inputs
    accepted = [arrays for _, arrays in items if arrays is not None]
    records = [record for record, _ in items]
    if not accepted:
        return {"records": records, "inputs": None}
    stacked = {key: torch.from_numpy(np.stack([a[key] for a in accepted])) for key in accepted[0]}
    values = canonicalize_model_inputs(stacked)
    return {"records": records, "template": values["template"], "search": values["search"],
            "targets": {key: values[key] for key in ("label_cls1", "labelcls2", "labelxff", "weightxff")}}
