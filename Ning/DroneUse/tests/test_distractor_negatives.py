import cv2
import numpy as np
import pytest

from hift_research.lasot.dataset import TrackPairDataset, template_outside_crop
from hift_research.lasot.tracks import visdrone_mot_tracks


def _mot_video(tmp_path, name, rows, frames=40):
    video = tmp_path/name
    video.mkdir()
    rng = np.random.default_rng(len(name))
    for i in range(1, frames+1):
        cv2.imwrite(str(video/f"{i:07d}.jpg"), rng.integers(0, 255, (480, 1600, 3), dtype=np.uint8))
    annotation = tmp_path/f"{name}.txt"
    annotation.write_text("\n".join(rows))
    return visdrone_mot_tracks(video, annotation, min_length=20)


def _rows(track_id, category, x, frames=range(1, 41), w=30, h=20):
    return [f"{f},{track_id},{x},{200},{w},{h},1,{category},0,0" for f in frames]


def test_crop_window_geometry():
    box = [500, 200, 30, 20]  # context sqrt(55*45)~49.7 -> 511-px window half ~100 px around (515, 210)
    assert template_outside_crop(box, [800, 200, 30, 20])
    assert not template_outside_crop(box, [560, 200, 30, 20])
    assert not template_outside_crop(box, [800, 200, 0, 20])  # unknown/unannotated template box


def test_tracks_carry_absolute_frame_ids(tmp_path):
    tracks = _mot_video(tmp_path, "uav01", _rows(1, 4, 100, frames=range(5, 41)))
    assert tracks[0]["frame_ids"][0] == 5 and len(tracks[0]["frame_ids"]) == len(tracks[0]["frames"])


def test_zero_probability_is_the_unchanged_sampler(tmp_path):
    tracks = (_mot_video(tmp_path, "uav01", _rows(1, 4, 100) + _rows(2, 4, 1200))
              + _mot_video(tmp_path, "uav02", _rows(1, 4, 300)))
    a = TrackPairDataset(tracks, length=30, seed=7, negative_probability=0.3)
    b = TrackPairDataset(tracks, length=30, seed=7, negative_probability=0.3, distractor_probability=0.0)
    assert [a[i][0] for i in range(30)] == [b[i][0] for i in range(30)]


def test_distractors_are_same_video_same_group_and_exclude_the_template(tmp_path):
    far = _mot_video(tmp_path, "uav01", _rows(1, 4, 100) + _rows(2, 5, 1200) + _rows(3, 1, 1400))  # car, van, pedestrian
    near = _mot_video(tmp_path, "uav03", _rows(1, 4, 100) + _rows(2, 4, 150))   # second car inside the crop window
    other = _mot_video(tmp_path, "uav02", _rows(1, 4, 300))
    tracks = far+near+other
    by_name = {t["name"]: t for t in tracks}
    data = TrackPairDataset(tracks, length=200, seed=3, negative_probability=0.0, distractor_probability=0.9)
    records = [data[i][0] for i in range(200)]
    picked = [r for r in records if r.get("distractor")]
    assert picked and all(r["negative"] for r in records if "distractor" in r)
    for r in picked:
        t, s = by_name[r["template"][0]], by_name[r["search"][0]]
        assert t["video"] == s["video"] and t["name"] != s["name"]
        assert {t["category"], s["category"]} <= {"car", "van"}       # same group (vehicle), never the pedestrian
        assert t["video"] != "VisDrone-MOT/uav03"                      # overlapping pair is always refused
    fallback = [r for r in records if r.get("distractor") is False]
    assert all(by_name[r["template"][0]]["video"] != by_name[r["search"][0]]["video"] for r in fallback)
    with pytest.raises(ValueError):
        TrackPairDataset(tracks, length=5, seed=0, distractor_probability=1.0)
