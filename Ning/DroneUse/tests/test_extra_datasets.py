import cv2
import numpy as np
import pytest

from hift_research.evaluation import assets
from hift_research.evaluation.assets import ValidationError
from hift_research.evaluation.extra_datasets import EXTRA_LAYOUTS, import_extra_dataset


def _tree(root, names, frames=3):
    for name in names:
        (root/"sequences"/name).mkdir(parents=True)
        for i in range(1, frames+1):
            cv2.imwrite(str(root/"sequences"/name/f"img{i:07d}.jpg"), np.zeros((20, 30, 3), np.uint8))
        (root/"annotations").mkdir(exist_ok=True)
        (root/"annotations"/f"{name}.txt").write_text("\n".join("1,2,5,6" for _ in range(frames)))


def test_registration_does_not_alter_frozen_benchmarks():
    assert assets.DATASET_COUNTS["VisDrone2019-SOT-test-dev"] == EXTRA_LAYOUTS["VisDrone2019-SOT-test-dev"][0] == 35
    assert {k: assets.DATASET_COUNTS[k] for k in ("UAV123", "UAV123@10fps", "UAV20L", "DTB70")} == \
        {"UAV123": 123, "UAV123@10fps": 123, "UAV20L": 20, "DTB70": 70}


def test_complete_count_is_enforced_and_names_must_match(tmp_path):
    _tree(tmp_path, ["uav0000001_00000_s", "uav0000002_00000_s"])
    with pytest.raises(ValidationError, match="expected 35 sequences"):
        import_extra_dataset(tmp_path, "VisDrone2019-SOT-test-dev")
    (tmp_path/"annotations"/"orphan.txt").write_text("1,2,5,6")
    with pytest.raises(ValidationError):
        import_extra_dataset(tmp_path, "VisDrone2019-SOT-test-dev")
    with pytest.raises(ValidationError):
        import_extra_dataset(tmp_path, "UAV123")
