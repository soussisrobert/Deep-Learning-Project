import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from hift_research.evaluation.metrics import evaluate_sequence
from hift_research.evaluation.validation import (make_episode, score_episode, select_episodes, summarize,
                                                 uavdt_holdout)

ROOT = Path(__file__).resolve().parents[1]


def _track(name, video, category, n=60, size=20.0, valid=None):
    boxes = np.tile([10.0, 10.0, size, size], (n, 1))
    boxes[:, 0] += np.arange(n)
    return {"name": name, "video": video, "source": "VisDrone-MOT", "category": category,
            "frames": [Path(f"{name}_{i}.jpg") for i in range(n)], "boxes": boxes,
            "valid": np.arange(n) if valid is None else np.asarray(valid)}


def test_uavdt_holdout_is_deterministic_and_order_free():
    names = [f"M{i:04d}" for i in range(50)]
    a = uavdt_holdout(names, 0.1)
    assert a == uavdt_holdout(list(reversed(names)), 0.1) and len(a) == 5 and a == sorted(a)
    assert uavdt_holdout(names, 0.0) == [] and len(uavdt_holdout(names[:3], 0.1)) == 1
    with pytest.raises(ValueError):
        uavdt_holdout(names, 1.0)


def test_episode_starts_at_first_large_valid_frame_and_caps_length():
    track = _track("t", "v", "car", n=100)
    track["boxes"][:5, 2] = 6  # too small to initialize
    episode = make_episode(track, max_frames=40)
    assert episode["init_bbox"] == track["boxes"][5].tolist() and len(episode["frames"]) == 40
    assert episode["scored"][0] == 0 and episode["scored"][-1] == 39 and episode["group"] == "vehicle"


def test_episode_scores_only_valid_frames_and_rejects_short():
    valid = [0, 1, 2] + list(range(30, 60))
    episode = make_episode(_track("t", "v", "pedestrian", valid=valid), max_frames=100)
    assert list(episode["scored"]) == valid and episode["group"] == "person"
    assert make_episode(_track("s", "v", "car", valid=range(10)), max_frames=100) is None


def test_unscored_frames_do_not_affect_metrics():
    valid = [0, 1, 2] + list(range(30, 60))
    episode = make_episode(_track("t", "v", "car", valid=valid), max_frames=100)
    predictions = episode["boxes"].copy()
    predictions[3:30] = [500.0, 500.0, 5.0, 5.0]  # garbage on unscored (occluded) frames
    result = score_episode(episode, predictions)
    reference = evaluate_sequence(episode["boxes"][valid], episode["boxes"][valid], invalid_policy="count_as_failure")
    assert result["success_auc"] == reference["success_auc"]
    with pytest.raises(ValueError):
        score_episode(episode, predictions[:-1])


def test_selection_is_stratified_deterministic_and_caps_per_video():
    tracks = [_track(f"v{v}#{i}", f"VisDrone-MOT/v{v}", cat) for v in range(4) for i in range(6)
              for cat in (["car"] if i % 2 else ["people"])]
    a = select_episodes(tracks, per_group={"vehicle": 5, "person": 5}, max_frames=50, max_per_video=2)
    b = select_episodes(list(reversed(tracks)), per_group={"vehicle": 5, "person": 5}, max_frames=50, max_per_video=2)
    assert [e["name"] for e in a] == [e["name"] for e in b]
    for group in ("vehicle", "person"):
        mine = [e for e in a if e["group"] == group]
        assert len(mine) == 5
        assert max(sum(1 for e in mine if e["video"] == v) for v in {e["video"] for e in mine}) <= 2


def test_summary_primary_metric():
    eps = [make_episode(_track(n, "v", c), max_frames=50) for n, c in (("a", "car"), ("b", "people"), ("c", "car"))]
    perfect = lambda e: score_episode(e, e["boxes"])
    results = {("V2", "a"): perfect(eps[0]), ("V2", "b"): perfect(eps[1]), ("V3", "c"): perfect(eps[2])}
    out = summarize(results, eps)
    auc = out["V2"]["vehicle"]["success_auc"]
    assert out["V2"]["person"]["count"] == 1 and out["M"] == pytest.approx(auc)


def _trainer():
    spec = importlib.util.spec_from_file_location("train_b2_domain", ROOT/"scripts/train_b2_domain.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT/"scripts"))
    spec.loader.exec_module(module)
    return module


def test_trainer_excludes_uavdt_holdout(tmp_path):
    import cv2
    uavdt = tmp_path/"uavdt"
    for name in ("M0101", "M0201"):
        video = uavdt/"UAV-benchmark-M"/name
        video.mkdir(parents=True)
        for i in range(1, 26):
            cv2.imwrite(str(video/f"img{i:06d}.jpg"), np.zeros((40, 60, 3), np.uint8))
        gt = uavdt/"UAV-benchmark-MOTD_v1.0/GT"
        gt.mkdir(parents=True, exist_ok=True)
        (gt/f"{name}_gt_whole.txt").write_text("\n".join(f"{f},1,5,5,10,10,1,1,1" for f in range(1, 26)))
    trainer = _trainer()
    every = trainer.domain_tracks("train", tmp_path, ["uavdt"])
    kept = trainer.domain_tracks("train", tmp_path, ["uavdt"], uavdt_holdout=["M0201"])
    assert {t["video"] for t in every} == {"UAVDT/M0101", "UAVDT/M0201"}
    assert {t["video"] for t in kept} == {"UAVDT/M0101"}


def test_finetune_frozen_backbone_mode():
    import torch
    trainer = _trainer()
    if not (ROOT/"artifacts/checkpoints/general_model.pth").exists():
        pytest.skip("official checkpoint not present")
    frozen, _, _, modes = trainer.build_official_finetune("corrected", 2, 1e-3, 1e-4, backbone="frozen")
    assert not any(p.requires_grad for p in frozen.backbone.parameters())
    assert all(not m.training for m in frozen.backbone.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    assert all(p.requires_grad for p in frozen.grader.parameters())
    trainable, _, _, _ = trainer.build_official_finetune("corrected", 2, 1e-3, 1e-4)
    assert any(p.requires_grad for p in trainable.backbone.parameters())
    with pytest.raises(ValueError):
        trainer.build_official_finetune("corrected", 2, 1e-3, 1e-4, backbone="partial")
