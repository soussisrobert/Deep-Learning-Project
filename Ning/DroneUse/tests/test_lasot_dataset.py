from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from hift_research.lasot.dataset import LaSOTPairDataset, collate_pairs, load_sequence
from hift_research.training.lifecycle import window_denominators


def _fake_sequence(root, name, frames=12, hidden=()):
    directory = root/"original"/name
    (directory/"img").mkdir(parents=True)
    rng = np.random.default_rng(len(name))
    boxes = []
    for i in range(frames):
        image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(directory/"img"/f"{i+1:08d}.jpg"), image)
        boxes.append(f"{100+i},{80+i},{40},{30}")
    (directory/"groundtruth.txt").write_text("\n".join(boxes))
    flags = ["1" if i in hidden else "0" for i in range(frames)]
    (directory/"full_occlusion.txt").write_text(",".join(flags))
    (directory/"out_of_view.txt").write_text(",".join("0" for _ in range(frames)))
    return name


@pytest.fixture
def lasot(tmp_path):
    names = [_fake_sequence(tmp_path, "cat-1", hidden={0, 1}), _fake_sequence(tmp_path, "cat-2"), _fake_sequence(tmp_path, "dog-3")]
    return tmp_path, names


def test_hidden_frames_are_never_valid(lasot):
    root, _ = lasot
    sequence = load_sequence(root/"original"/"cat-1")
    assert 0 not in sequence["valid"] and 1 not in sequence["valid"] and len(sequence["valid"]) == 10


def test_pairs_are_reproducible_and_well_formed(lasot):
    root, names = lasot
    dataset = LaSOTPairDataset(root, names, length=40, seed=7, negative_probability=0.3, max_gap=5)
    again = LaSOTPairDataset(root, names, length=40, seed=7, negative_probability=0.3, max_gap=5)
    negatives = positives = 0
    for index in range(40):
        record, arrays = dataset[index]
        record2, arrays2 = again[index]
        assert record == record2
        if arrays is None:
            continue
        assert all(np.array_equal(arrays[k], arrays2[k]) for k in arrays)
        assert arrays["template"].shape == (3, 127, 127) and arrays["search"].shape == (3, 287, 287)
        if record["negative"]:
            negatives += 1
            assert record["template"][0] != record["search"][0]
            assert not np.any(arrays["label_cls1"]) and not np.any(arrays["weightxff"]) and not np.any(arrays["labelcls2"])
        else:
            positives += 1
            assert record["template"][0] == record["search"][0]
            assert 1 <= abs(record["template"][1]-record["search"][1]) <= 5
            assert np.any(arrays["label_cls1"] == 1) and np.any(arrays["weightxff"] > 0)
    assert negatives and positives


def test_collate_matches_engine_layout(lasot):
    root, names = lasot
    dataset = LaSOTPairDataset(root, names, length=8, seed=1)
    batch = collate_pairs([dataset[i] for i in range(8)])
    assert batch["template"].dtype == torch.float32 and batch["template"].shape[1:] == (3, 127, 127)
    denominators = window_denominators([batch])
    assert denominators["bce"] == batch["targets"]["labelcls2"].numel()
    assert collate_pairs([({"index": 0}, None)])["inputs"] is None


def test_requires_two_sequences(tmp_path):
    name = _fake_sequence(tmp_path, "cat-1")
    with pytest.raises(ValueError):
        LaSOTPairDataset(tmp_path, [name], length=4, seed=0)
