import cv2
import numpy as np

from hift_research.lasot.dataset import TrackPairDataset
from hift_research.lasot.tracks import uavdt_tracks, visdrone_mot_tracks


def _video(root, name, frames=30, prefix=""):
    directory = root/name
    directory.mkdir(parents=True)
    rng = np.random.default_rng(len(name))
    for i in range(1, frames+1):
        cv2.imwrite(str(directory/f"{prefix}{i:07d}.jpg"), rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    return directory


def _visdrone_rows(track_id, category, frames, occlusion=0):
    return [f"{f},{track_id},{100+f},{80},{30},{20},1,{category},0,{occlusion}" for f in frames]


def test_visdrone_mot_keeps_domain_classes_and_clean_frames(tmp_path):
    video = _video(tmp_path, "uav01")
    rows = (_visdrone_rows(1, 4, range(1, 31))                      # car: kept
            + _visdrone_rows(2, 1, range(1, 31), occlusion=2)       # pedestrian, all heavily occluded: dropped
            + _visdrone_rows(3, 3, range(1, 31))                    # bicycle: out of domain
            + _visdrone_rows(4, 9, range(1, 11)))                   # bus, too short: dropped
    annotation = tmp_path/"uav01.txt"
    annotation.write_text("\n".join(rows))
    tracks = visdrone_mot_tracks(video, annotation, min_length=20)
    assert [t["category"] for t in tracks] == ["car"]
    assert tracks[0]["video"] == "VisDrone-MOT/uav01" and len(tracks[0]["valid"]) == 30


def test_uavdt_out_of_view_and_occlusion_filtered(tmp_path):
    video = _video(tmp_path, "M0101", prefix="img")
    # occlusion codes (official README): 1 none, 2 LARGE, 3 medium, 4 small; frames 26-30 partially out of view
    codes = {0: 1, 1: 2, 2: 3, 3: 4}
    rows = [f"{f},7,{50+f},60,25,15,{1 if f <= 25 else 2},{codes[f % 4]},1" for f in range(1, 31)]
    annotation = tmp_path/"M0101_gt_whole.txt"
    annotation.write_text("\n".join(rows))
    tracks = uavdt_tracks(video, annotation, min_length=5)
    assert len(tracks) == 1 and tracks[0]["category"] == "car"
    assert set(tracks[0]["valid"]) == {f-1 for f in range(1, 26) if codes[f % 4] != 2}  # medium/small occlusion kept


def test_negatives_come_from_another_video(tmp_path):
    video_a, video_b = _video(tmp_path, "a"), _video(tmp_path, "b")
    rows = _visdrone_rows(1, 4, range(1, 31)) + _visdrone_rows(2, 5, range(1, 31))  # two tracks in video a
    (tmp_path/"a.txt").write_text("\n".join(rows))
    (tmp_path/"b.txt").write_text("\n".join(_visdrone_rows(1, 6, range(1, 31))))
    tracks = visdrone_mot_tracks(video_a, tmp_path/"a.txt") + visdrone_mot_tracks(video_b, tmp_path/"b.txt")
    assert len(tracks) == 3
    video_of = {t["name"]: t["video"] for t in tracks}
    dataset = TrackPairDataset(tracks, length=200, seed=3, negative_probability=0.5, max_gap=5)
    negatives = 0
    for index in range(200):
        record, _ = dataset[index]
        if record["negative"]:
            negatives += 1
            assert video_of[record["template"][0]] != video_of[record["search"][0]]
    assert negatives > 50
