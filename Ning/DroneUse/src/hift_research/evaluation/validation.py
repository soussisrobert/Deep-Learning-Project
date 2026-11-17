"""Tracking validation for model selection on the aerial domain; the sealed test sets are never touched.

V1 = VisDrone2019-SOT-val sequences (whole sequences).
V2 = single-object episodes cut from VisDrone2019-MOT-val tracks, class-stratified (vehicle / person).
V3 = episodes from UAVDT videos held out of training (``uavdt_holdout``; the trainer excludes them).
Neither val split is trained on (train_b2_domain.py trains SOT train_part1/2, MOT train, UAVDT minus V3).

An episode starts at the first valid frame whose box is at least ``min_init`` px on both sides, runs over
consecutive frames for at most ``max_frames`` and is scored only on frames with valid ground truth
(occluded/out-of-view frames are tracked but not scored). Scoring reuses metrics.evaluate_sequence.
Selection is deterministic: a fixed-seed permutation per group, independent of any model result.
"""
from __future__ import annotations

import numpy as np

from .metrics import aggregate_sequences, evaluate_sequence

VEHICLE = {"car", "van", "truck", "bus"}
PERSON = {"pedestrian", "people"}
VALIDATION_SEED = 20261030


def group_of(track):
    category = track["category"]
    if category in VEHICLE:
        return "vehicle"
    if category in PERSON:
        return "person"
    return "unlabeled"


def uavdt_holdout(video_names, fraction=0.1, seed=VALIDATION_SEED):
    """Deterministic whole-video holdout (sorted names, seeded permutation); at least one video when fraction > 0."""
    names = sorted(set(video_names))
    if not 0 <= fraction < 1:
        raise ValueError("holdout fraction must be in [0, 1)")
    if fraction == 0 or not names:
        return []
    count = max(1, round(fraction*len(names)))
    order = np.random.default_rng(seed).permutation(len(names))
    return sorted(names[i] for i in order[:count])


def make_episode(track, *, max_frames, min_init=10, min_scored=20):
    """Episode dict or None when the track has no usable start / too few scorable frames."""
    boxes, valid = np.asarray(track["boxes"], dtype=float), np.asarray(track["valid"], dtype=int)
    big = [i for i in valid if boxes[i, 2] >= min_init and boxes[i, 3] >= min_init]
    if not big:
        return None
    start = int(big[0])
    stop = min(len(track["frames"]), start+max_frames)
    scored = valid[(valid >= start) & (valid < stop)]-start
    if len(scored) < min_scored:
        return None
    return {"name": track["name"], "video": track["video"], "source": track["source"], "group": group_of(track),
            "category": track["category"], "frames": list(track["frames"][start:stop]),
            "boxes": boxes[start:stop], "scored": scored, "init_bbox": boxes[start].tolist()}


def select_episodes(tracks, *, per_group, max_frames, seed=VALIDATION_SEED, max_per_video=None):
    """Fixed-seed, group-stratified episode selection; per_group maps group -> count (None = all)."""
    pools = {}
    for track in sorted(tracks, key=lambda t: t["name"]):
        episode = make_episode(track, max_frames=max_frames)
        if episode is not None and episode["group"] in per_group:
            pools.setdefault(episode["group"], []).append(episode)
    chosen = []
    for group in sorted(pools):
        pool, limit = pools[group], per_group[group]
        order = np.random.default_rng([seed, len(group)]).permutation(len(pool))
        used = {}
        for i in order:
            if limit is not None and sum(1 for e in chosen if e["group"] == group) >= limit:
                break
            video = pool[i]["video"]
            if max_per_video is not None and used.get(video, 0) >= max_per_video:
                continue
            used[video] = used.get(video, 0)+1
            chosen.append(pool[i])
    return chosen


def score_episode(episode, predictions):
    """Metrics on scored (valid-GT) frames only; the initialization frame (index 0) is always scored."""
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(episode["frames"]), 4):
        raise ValueError("prediction count differs from episode length")
    idx = episode["scored"]
    return evaluate_sequence(episode["boxes"][idx], predictions[idx], invalid_policy="count_as_failure")


def summarize(results, episodes):
    """Per subset (V1/V2/V3) and per group sequence-macro AUC, plus the primary metric M =
    mean(V1, mean(V2 vehicle, V2 person), V3) over the subsets that are present."""
    by_name = {e["name"]: e for e in episodes}
    out, subsets = {}, {}
    for (subset, name), metrics in results.items():
        subsets.setdefault(subset, {})[name] = metrics
    for subset, rows in sorted(subsets.items()):
        entry = {"all": aggregate_sequences(rows)}
        for group in ("vehicle", "person", "unlabeled"):
            mine = {n: m for n, m in rows.items() if by_name[n]["group"] == group}
            if mine:
                entry[group] = aggregate_sequences(mine)
        out[subset] = {k: {"count": v["sequence_count"], "success_auc": v["success_auc"],
                           "precision_at_20px": v["precision_at_20px"]} for k, v in entry.items()}
    parts = []
    if "V1" in out:
        parts.append(out["V1"]["all"]["success_auc"])
    if "V2" in out:
        groups = [out["V2"][g]["success_auc"] for g in ("vehicle", "person") if g in out["V2"]]
        parts.append(float(np.mean(groups)))
    if "V3" in out:
        parts.append(out["V3"]["all"]["success_auc"])
    out["M"] = float(np.mean(parts)) if parts else None
    out["M_definition"] = "mean(V1 AUC, mean(V2 vehicle AUC, V2 person AUC), V3 AUC) over present subsets"
    return out
